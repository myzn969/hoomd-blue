#pragma once

#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/jit/NVRTCEvalFactory.h"

#include "hoomd/jit/PatchEnergyJITGPU.h"
#include "hoomd/jit/PatchEnergyJITUnionGPU.h"

#include <string>
#include <memory>

#ifdef ENABLE_HIP

#include <hip/hip_runtime.h>

//! A wrapper around the GPU factory object.
/* JIT kernel implementations hold a reference to the JITKernel and provide launch
   interfaces with specific kernel argument types.

   In the future, the internal NVRTCEvalFactory may have a sibling for ROCm RTC frameworks.
*/

namespace jit
{

template<class JIT>
class JITKernelBase
    {
    public:
        JITKernelBase(std::shared_ptr<const ExecutionConfiguration>& exec_conf,
                  const std::string& kernel_code,
                  const std::string& kernel_name,
                  std::shared_ptr<JIT> jit)
            : m_factory(exec_conf, kernel_code + jit->getEvaluatorCode(), kernel_name, jit->getCompilerOptions()),
              m_jit(jit)
            { }

        #ifdef __HIP_PLATFORM_NVCC__
        NVRTCEvalFactory& getFactory()
            {
            return m_factory;
            }
        #endif

        //! Return the specialiazation of the PatchEnergy class
        std::shared_ptr<JIT> getJIT()
            {
            return m_jit;
            }

    protected:
        #ifdef __HIP_PLATFORM_NVCC__
        NVRTCEvalFactory m_factory;  //!< The run-time-compilation (RTC) object
        #endif

        std::shared_ptr<JIT> m_jit; //!< The associated JIT potential
    };

template<class JIT>
class JITKernel {};

template<>
class JITKernel<PatchEnergyJITGPU> : public JITKernelBase<PatchEnergyJITGPU>
    {
    using JIT = PatchEnergyJITGPU;

    public:
        JITKernel(std::shared_ptr<const ExecutionConfiguration>& exec_conf,
                  const std::string& kernel_code,
                  const std::string& kernel_name,
                  std::shared_ptr<JIT> jit)
            : JITKernelBase<JIT>(exec_conf, kernel_code, kernel_name, jit)
            { }

        template<typename... TArgs, typename... Args>
        void setup(hipStream_t stream, Args&&... args)
            {
            this->m_factory.setGlobalVariable<TArgs...>("alpha_iso",
                this->m_jit->getAlpha(), stream,
                std::forward<Args>(args)...);
            }
    };

template<>
class JITKernel<PatchEnergyJITUnionGPU> : public JITKernelBase<PatchEnergyJITUnionGPU>
    {
    using JIT = PatchEnergyJITUnionGPU;

    public:
        JITKernel(std::shared_ptr<const ExecutionConfiguration>& exec_conf,
                  const std::string& kernel_code,
                  const std::string& kernel_name,
                  std::shared_ptr<JIT> jit)
            : JITKernelBase<JIT>(exec_conf, kernel_code, kernel_name, jit)
            { }

        template<typename... TArgs, typename... Args>
        void setup(hipStream_t stream, Args&&... args)
            {
            this->m_factory.setGlobalVariable<TArgs...>("alpha_iso",
                this->m_jit->getAlpha(), stream,
                std::forward<Args>(args)...);

            this->m_factory.setGlobalVariable<TArgs...>("alpha_union",
                this->m_jit->getAlphaUnion(), stream,
                std::forward<Args>(args)...);

            this->m_factory.setGlobalVariable<TArgs...>("jit::d_union_params",
                &this->m_jit->getDeviceParams().front(), stream,
                std::forward<Args>(args)...);

            this->m_factory.setGlobalVariable<TArgs...>("jit::d_rcut_union",
                this->m_jit->getRcutUnion(), stream,
                std::forward<Args>(args)...);
            }
    };

} // end namespace jit
#endif
