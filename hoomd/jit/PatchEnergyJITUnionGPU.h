#pragma once

#ifdef ENABLE_HIP

#include "PatchEnergyJITUnion.h"
#include "NVRTCEvalFactory.h"
#include "hoomd/managed_allocator.h"
#include "EvaluatorUnionGPU.cuh"

#include <vector>

//! Evaluate patch energies via runtime generated code, GPU version
class PYBIND11_EXPORT PatchEnergyJITUnionGPU : public PatchEnergyJITUnion
    {
    public:
        //! Constructor
        PatchEnergyJITUnionGPU(std::shared_ptr<SystemDefinition> sysdef,
            std::shared_ptr<ExecutionConfiguration> exec_conf,
            const std::string& llvm_ir_iso, Scalar r_cut_iso,
            const unsigned int array_size_iso,
            const std::string& llvm_ir_union, Scalar r_cut_union,
            const unsigned int array_size_union,
            const std::string& code,
            const std::vector<std::string>& options)
            : PatchEnergyJITUnion(sysdef, exec_conf, llvm_ir_iso, r_cut_iso, array_size_iso, llvm_ir_union, r_cut_union, array_size_union),
              m_d_union_params(this->m_sysdef->getParticleData()->getNTypes(),
                jit::union_params_t(), managed_allocator<jit::union_params_t>(this->m_exec_conf->isCUDAEnabled()))
            {
            m_options = options;
            m_eval_code = code;
            }

        virtual ~PatchEnergyJITUnionGPU() {}

        //! Set the per-type constituent particles
        /*! \param type The particle type to set the constituent particles for
            \param rcut The maximum cutoff over all constituent particles for this type
            \param types The type IDs for every constituent particle
            \param positions The positions
            \param orientations The orientations
            \param leaf_capacity Number of particles in OBB tree leaf
         */
        virtual void setParam(unsigned int type,
            pybind11::list types,
            pybind11::list positions,
            pybind11::list orientations,
            pybind11::list diameters,
            pybind11::list charges,
            unsigned int leaf_capacity=4);

        //! Method to be called when number of types changes
        virtual void slotNumTypesChange()
            {
            PatchEnergyJITUnion::slotNumTypesChange();
            unsigned int ntypes = this->m_sysdef->getParticleData()->getNTypes();
            m_d_union_params.resize(ntypes);
            m_need_to_initialize = true;
            }

        #ifdef __HIP_PLATFORM_NVCC__
        //! Set up the GPU kernel
        /*! \param kernel The jit kernel factory
         */
        template<class T>
        void setupGPUKernel(NVRTCEvalFactory& kernel)
            {
            if (m_need_to_initialize)
                {
                kernel.setAlphaPtr<T>(&this->m_alpha.front());
                kernel.setAlphaUnionPtr<T>(&this->m_alpha_union.front());
                kernel.setUnionParamsPtr<T>(&this->m_d_union_params.front());
                kernel.setRCutUnion<T>(this->m_rcut_union);
                }

            m_need_to_initialize = false;
            }
        #endif

        //! Return the per-type parameters
        const std::vector<jit::union_params_t, managed_allocator<jit::union_params_t> >& getDeviceParams() const
            {
            return m_d_union_params;
            }

        //! Return the list of options passed to the RTC
        const std::vector<std::string>& getCompilerOptions() const
            {
            return m_options;
            }

        std::string getEvaluatorCode() const
            {
            return m_eval_code;
            }

    protected:
        std::unique_ptr<Autotuner> m_tuner_narrow_patch;     //!< Autotuner for the narrow phase

        std::vector<jit::union_params_t, managed_allocator<jit::union_params_t> > m_d_union_params;   //!< Parameters for each particle type on GPU

    private:
        std::vector<std::string> m_options;         //!< List of compiler flags
        std::string m_eval_code;                    //!< Evaluator code snippet
        bool m_need_to_initialize = true;                    //!< True if sizes of parameter data structures have changed
    };

void export_PatchEnergyJITUnionGPU(pybind11::module &m);

#endif // ENABLE_HIP
