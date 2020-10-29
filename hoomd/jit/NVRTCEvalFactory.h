#pragma once

#ifdef ENABLE_HIP

#include "PatchEnergyJIT.h"
#include "EvaluatorUnionGPU.cuh"
#include "hoomd/hpmc/IntegratorHPMC.h"

#include <hip/hip_runtime.h>

#ifdef __HIP_PLATFORM_NVCC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

//! uncomment to debug JIT compilation errors
//#define DEBUG_JIT

#ifdef DEBUG_JIT
#define JITIFY_PRINT_LOG 1
#define JITIFY_PRINT_LINKER_LOG 1
#define JITIFY_PRINT_LAUNCH 1
#else
#define JITIFY_PRINT_LOG 0
#define JITIFY_PRINT_LAUNCH 0
#endif
#define JITIFY_PRINT_INSTANTIATION 0
#define JITIFY_PRINT_SOURCE 0
#define JITIFY_PRINT_PTX 0
#define JITIFY_PRINT_HEADER_PATHS 0

#include "jitify.hpp"

#endif

#include <vector>
#include <map>
#include <utility>
#include <tuple>

namespace jit
{

namespace detail
{

//! Variadic helpers to forward arguments to jitify
template<typename Tuple, int... Is>
auto for_each_type_impl(std::integer_sequence<int, Is...>)
    {
    using jitify::reflection::Type;
    return std::make_tuple(Type<typename std::tuple_element<Is, Tuple>::type>()...);
    }

template<typename Tuple>
auto for_each_type()
    {
    return detail::for_each_type_impl<Tuple>(std::make_integer_sequence<int, std::tuple_size<Tuple>::value >());
    }

template<typename K, typename Tuple, int... Is, typename... Args>
auto instantiate_impl(K &&kernel, Tuple&& t, std::integer_sequence<int, Is...>, Args&&... args)
    {
    return std::forward<K>(kernel).instantiate(std::get<Is>(t)..., std::forward<Args>(args)...);
    }

template<typename K, typename Tuple, typename...Args>
auto instantiate(K&& kernel, Tuple&& t, Args&&... args)
    {
    return instantiate_impl(std::forward<K>(kernel), std::forward<Tuple>(t),
        std::make_integer_sequence<int, std::tuple_size<typename std::decay<Tuple>::type>{} >(),
        std::forward<Args>(args)...);
    }

} // end namespace detail

//! Evaluate patch energies via runtime generated code, GPU version
/*! This class encapsulates a JIT compiled kernel and provides the API necessary to query kernel
    parameters and launch the kernel into a stream.

    Additionally, it allows access to pointers alpha_iso and alpha_union defined at global scope.
 */
class PYBIND11_EXPORT NVRTCEvalFactory
    {
    public:
        //! Constructor
        NVRTCEvalFactory(std::shared_ptr<const ExecutionConfiguration> exec_conf,
                       const std::string& code,
                       const std::string& kernel_name,
                       const std::vector<std::string>& options)
            : m_exec_conf(exec_conf), m_kernel_name(kernel_name)
            {
            // instantiate jitify cache
            #ifdef __HIP_PLATFORM_NVCC__
            m_cache.resize(this->m_exec_conf->getNumActiveGPUs());
            #endif

            compileGPU(code, kernel_name, options);
            }

        ~NVRTCEvalFactory()
            { }

        //! Return the maximum number of threads per block for this kernel
        /* \param idev the logical GPU id
           \param args template parameter values

           \tparam TArgs Kernel template parameter types
           \tparam Args Types of non-type kernel template arguments
         */
        template<typename... TArgs, typename... Args>
        unsigned int getKernelMaxThreads(unsigned int idev, Args&&... args)
            {
            int max_threads = 0;

            #ifdef __HIP_PLATFORM_NVCC__
            auto types = detail::for_each_type<std::tuple<TArgs...> >();
            CUresult custatus = cuFuncGetAttribute(&max_threads,
                CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                detail::instantiate(m_program[idev].kernel(m_kernel_name),
                    types, std::forward<Args>(args)...));
            char *error;
            if (custatus != CUDA_SUCCESS)
                {
                cuGetErrorString(custatus, const_cast<const char **>(&error));
                throw std::runtime_error("cuFuncGetAttribute: "+std::string(error));
                }
            #endif

            return max_threads;
            }

        //! Return the shared size usage in bytes for this kernel
        /* \param idev the logical GPU id

           \tparam TArgs Kernel template parameter types
           \tparam Args Kernel template parameter values
         */
        template<typename... TArgs, typename... Args>
        unsigned int getKernelSharedSize(unsigned int idev, Args&&... args)
            {
            int shared_size = 0;

            #ifdef __HIP_PLATFORM_NVCC__
            auto types = detail::for_each_type<std::tuple<TArgs...> >();
            CUresult custatus = cuFuncGetAttribute(&shared_size,
                CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                detail::instantiate(m_program[idev].kernel(m_kernel_name),
                    types, std::forward<Args>(args)...));
            char *error;
            if (custatus != CUDA_SUCCESS)
                {
                cuGetErrorString(custatus, const_cast<const char **>(&error));
                throw std::runtime_error("cuFuncGetAttribute: "+std::string(error));
                }
            #endif

            return shared_size;
            }

        //! Asynchronously launch the JIT kernel
        /*! \param idev logical GPU id to launch on
            \param grid The grid dimensions
            \param threads The thread block dimensions
            \param sharedMemBytes The size of the dynamic shared mem allocation
            \param hStream stream to execute on

            \tparam TArgs Kernel template parameter types
            \tparam Args Kernel template parameter values
         */
        #ifdef __HIP_PLATFORM_NVCC__
        template<typename... TArgs, typename... Args>
        jitify::KernelLauncher configureKernel(unsigned int idev,
            dim3 grid, dim3 threads, unsigned int sharedMemBytes, cudaStream_t hStream,
            Args&&... args)
            {
            cudaSetDevice(m_exec_conf->getGPUIds()[idev]);

            auto types = detail::for_each_type<std::tuple<TArgs...> >();
            return detail::instantiate(m_program[idev].kernel(m_kernel_name),
                    types, std::forward<Args>(args)...)
                    .configure(grid, threads, sharedMemBytes, hStream);
            }
        #endif

        template<typename... TArgs, typename T, typename... Args>
        void setGlobalVariable(unsigned int idev, const std::string& var,
            T value, cudaStream_t stream, Args&&... args)
            {
            #ifdef __HIP_PLATFORM_NVCC__
            auto types = detail::for_each_type<std::tuple<TArgs...> >();

            CUdeviceptr ptr = detail::instantiate(m_program[idev].kernel(m_kernel_name),
                types, std::forward<Args>(args)...)
                .get_global_ptr(var.c_str());

            // copy the value to the device
            char *error;
            CUresult custatus = cuMemcpyHtoDAsync(ptr, &value, sizeof(T), stream);
            if (custatus != CUDA_SUCCESS)
                {
                cuGetErrorString(custatus, const_cast<const char **>(&error));
                throw std::runtime_error("cuMemcpyHtoDAsync: "+std::string(error));
                }
            #endif
            }

    private:
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< The exceuction configuration
        const std::string m_kernel_name;                     //!< The name of the __global__ function

        //! Helper function for RTC
        void compileGPU(const std::string& code,
            const std::string& kernel_name,
            const std::vector<std::string>& options);

        #ifdef __HIP_PLATFORM_NVCC__
        std::vector<jitify::JitCache> m_cache;          //!< jitify kernel cache, one per GPU
        std::vector<jitify::Program> m_program;         //!< The kernel object, one per GPU
        #endif
    };
#endif

} // end namespace jit
