#pragma once

#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/jit/NVRTCEvalFactory.h"

#include <string>
#include <memory>

#ifdef ENABLE_HIP

//! A wrapper around the GPU factory object.
/* Actual JIT kernel implementations hold a reference to the JITKernel and provide launch interfaces with specific kernel
   argument types.

   In the future, the internal NVRTCEvalFactory may have a sibling for ROCm RTC frameworks.

   \tparam JIT the JIT potential class associated with this kernel
*/

namespace jit
{

template<class JIT>
class PYBIND11_EXPORT JITKernel
    {
    public:
        JITKernel(std::shared_ptr<const ExecutionConfiguration>& exec_conf,
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

        //! (Lazily) initialize the kernel data structures
        template<class T>
        void setup()
            {
            m_jit->template setupGPUKernel<T>(m_factory);
            }

    protected:
        #ifdef __HIP_PLATFORM_NVCC__
        NVRTCEvalFactory m_factory;  //!< The run-time-compilation (RTC) object
        #endif

        std::shared_ptr<JIT> m_jit; //!< The associated JIT potential
    };

} // end namespace jit
#endif
