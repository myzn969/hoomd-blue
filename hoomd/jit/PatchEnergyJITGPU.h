#ifndef _PATCH_ENERGY_JIT_GPU_H_
#define _PATCH_ENERGY_JIT_GPU_H_

#ifdef ENABLE_HIP

#include "PatchEnergyJIT.h"
#include "NVRTCEvalFactory.h"
#include <pybind11/stl.h>

#include <vector>

//! Evaluate patch energies via runtime generated code, GPU version
class PYBIND11_EXPORT PatchEnergyJITGPU : public PatchEnergyJIT
    {
    public:
        //! Constructor
        PatchEnergyJITGPU(std::shared_ptr<ExecutionConfiguration> exec_conf, const std::string& llvm_ir, Scalar r_cut,
                       const unsigned int array_size,
                       const std::string& code,
                       const std::vector<std::string>& options)
            : PatchEnergyJIT(exec_conf, llvm_ir, r_cut, array_size)
            {
            m_options = options;
            m_eval_code = code;
            }

        //! Return the list of options passed to the RTC
        std::vector<std::string> getCompilerOptions() const
            {
            return m_options;
            }

        std::string getEvaluatorCode() const
            {
            return m_eval_code;
            }

    protected:
        std::unique_ptr<Autotuner> m_tuner_narrow_patch;     //!< Autotuner for the narrow phase

    private:
        std::string m_eval_code;                    //!< Code snippet for evaluator function
        std::vector<std::string> m_options;         //!< List of compiler flags
    };

void export_PatchEnergyJITGPU(pybind11::module &m);

#endif // ENABLE_HIP
#endif // _PATCH_ENERGY_JIT_GPU_H_
