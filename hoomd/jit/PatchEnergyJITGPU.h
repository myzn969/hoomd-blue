#ifndef _PATCH_ENERGY_JIT_GPU_H_
#define _PATCH_ENERGY_JIT_GPU_H_

#ifdef ENABLE_HIP

#include "PatchEnergyJIT.h"
#include "GPUEvalFactory.h"
#include <pybind11/stl.h>

#include <vector>

#include "hoomd/Autotuner.h"

//! Evaluate patch energies via runtime generated code, GPU version
template<class Shape>
class PYBIND11_EXPORT PatchEnergyJITGPU : public PatchEnergyJIT<Shape>
    {
    public:
        //! Constructor
        PatchEnergyJITGPU(std::shared_ptr<ExecutionConfiguration> exec_conf, const std::string& llvm_ir, Scalar r_cut,
                       const unsigned int array_size,
                       const std::string& code,
                       const std::string& kernel_name,
                       const std::vector<std::string>& options,
                       const std::string& cuda_devrt_library_path,
                       unsigned int compute_arch)
            : PatchEnergyJIT<Shape>(exec_conf, llvm_ir, r_cut, array_size),
              m_gpu_factory(exec_conf, code, kernel_name, options, cuda_devrt_library_path, compute_arch)
            {
            m_gpu_factory.setAlphaPtr<Shape>(&this->m_alpha.front());

            // tuning params for patch narrow phase
            std::vector<unsigned int> valid_params_patch;
            const unsigned int narrow_phase_max_threads_per_eval = this->m_exec_conf->dev_prop.warpSize;
            auto& launch_bounds = m_gpu_factory.getLaunchBounds();
            for (auto cur_launch_bounds: launch_bounds)
                {
                for (unsigned int group_size=1; group_size <= cur_launch_bounds; group_size*=2)
                    {
                    for (unsigned int eval_threads=1; eval_threads <= narrow_phase_max_threads_per_eval; eval_threads *= 2)
                        {
                        if ((cur_launch_bounds % (group_size*eval_threads)) == 0)
                            valid_params_patch.push_back(cur_launch_bounds*1000000 + group_size*100 + eval_threads);
                        }
                    }
                }

            m_tuner_narrow_patch.reset(new Autotuner(valid_params_patch, 5, 100000, "hpmc_narrow_patch", this->m_exec_conf));
            }

        //! Asynchronously launch the JIT kernel
        /*! \param args Kernel arguments
            \param hStream stream to execute on
            */
        virtual void computePatchEnergyGPU(const typename hpmc::PatchEnergy<Shape>::gpu_args_t& args, hipStream_t hStream);

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            m_tuner_narrow_patch->setPeriod(period);
            m_tuner_narrow_patch->setEnabled(enable);
            }

    protected:
        std::unique_ptr<Autotuner> m_tuner_narrow_patch;     //!< Autotuner for the narrow phase

    private:
        GPUEvalFactory m_gpu_factory;                       //!< JIT implementation
    };

//! Kernel driver for kernel::hpmc_narrow_phase_patch
template<class Shape>
void PatchEnergyJITGPU<Shape>::computePatchEnergyGPU(const typename hpmc::PatchEnergy<Shape>::gpu_args_t& args, hipStream_t hStream)
    {
    #ifdef __HIP_PLATFORM_NVCC__
    assert(args.d_postype);
    assert(args.d_orientation);

    unsigned int param = m_tuner_narrow_patch->getParam();
    unsigned int block_size = param/1000000;
    unsigned int req_tpp = (param%1000000)/100;
    unsigned int eval_threads = param % 100;

    this->m_exec_conf->beginMultiGPU();
    m_tuner_narrow_patch->begin();

    // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
    unsigned int run_block_size = std::min(block_size, m_gpu_factory.getKernelMaxThreads<Shape>(0, eval_threads, block_size)); // fixme GPU 0

    unsigned int tpp = std::min(req_tpp,run_block_size);
    while (eval_threads*tpp > run_block_size || run_block_size % (eval_threads*tpp) != 0)
        {
        tpp--;
        }
    auto& devprop = this->m_exec_conf->dev_prop;
    tpp = std::min((unsigned int) devprop.maxThreadsDim[2], tpp); // clamp blockDim.z

    unsigned int n_groups = run_block_size/(tpp*eval_threads);

    unsigned int max_queue_size = n_groups*tpp;

    const unsigned int min_shared_bytes = args.num_types * sizeof(Scalar);

    unsigned int shared_bytes = n_groups * (4*sizeof(unsigned int) + 2*sizeof(Scalar4) + 2*sizeof(Scalar3) + 2*sizeof(Scalar))
        + max_queue_size * 2 * sizeof(unsigned int)
        + min_shared_bytes;

    if (min_shared_bytes >= devprop.sharedMemPerBlock)
        throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of particle types or size of shape parameters");

    unsigned int kernel_shared_bytes = m_gpu_factory.getKernelSharedSize<Shape>(0, eval_threads, block_size); //fixme GPU 0
    while (shared_bytes + kernel_shared_bytes >= devprop.sharedMemPerBlock)
        {
        run_block_size -= devprop.warpSize;
        if (run_block_size == 0)
            throw std::runtime_error("Insufficient shared memory for HPMC kernel");

        tpp = std::min(req_tpp, run_block_size);
        while (eval_threads*tpp > run_block_size || run_block_size % (eval_threads*tpp) != 0)
            {
            tpp--;
            }

        tpp = std::min((unsigned int) devprop.maxThreadsDim[2], tpp); // clamp blockDim.z

        n_groups = run_block_size / (tpp*eval_threads);
        max_queue_size = n_groups*tpp;

        shared_bytes = n_groups * (4*sizeof(unsigned int) + 2*sizeof(Scalar4) + 2*sizeof(Scalar3) + 2*sizeof(Scalar))
            + max_queue_size * 2 * sizeof(unsigned int)
            + min_shared_bytes;
        }

    dim3 thread(eval_threads, n_groups, tpp);

    auto& gpu_partition = args.gpu_partition;

    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;
        const unsigned int num_blocks = (nwork + n_groups - 1)/n_groups;

        dim3 grid(num_blocks, 1, 1);

        unsigned int max_extra_bytes = 0;
        unsigned int N_old = args.N + args.N_ghost;

        // configure the kernel
        auto launcher = m_gpu_factory.configureKernel<Shape>(idev, grid, thread, shared_bytes, hStream, eval_threads, block_size);

        CUresult res = launcher(args.d_postype,
            args.d_orientation,
            args.d_trial_postype,
            args.d_trial_orientation,
            args.d_charge,
            args.d_diameter,
            args.d_excell_idx,
            args.d_excell_size,
            args.excli,
            args.d_nlist_old,
            args.d_energy_old,
            args.d_nneigh_old,
            args.d_nlist_new,
            args.d_energy_new,
            args.d_nneigh_new,
            args.maxn,
            args.num_types,
            args.box,
            args.ghost_width,
            args.cell_dim,
            args.ci,
            N_old,
            args.N,
            args.r_cut_patch,
            args.d_additive_cutoff,
            args.d_overflow,
            args.d_reject_out_of_cell,
            max_queue_size,
            range.first,
            nwork,
            max_extra_bytes);

        if (res != CUDA_SUCCESS)
            {
            char *error;
            cuGetErrorString(res, const_cast<const char **>(&error));
            throw std::runtime_error("Error launching NVRTC kernel: "+std::string(error));
            }
        }

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_narrow_patch->end();
    this->m_exec_conf->endMultiGPU();
    #endif
    }

template<class Shape>
void export_PatchEnergyJITGPU(pybind11::module &m, const std::string& name);

#ifdef __EXPORT_IMPL__
template<class Shape>
void export_PatchEnergyJITGPU(pybind11::module &m, const std::string& name)
    {
    pybind11::class_<PatchEnergyJITGPU<Shape>, PatchEnergyJIT<Shape>,
                     std::shared_ptr<PatchEnergyJITGPU<Shape> > >(m, name.c_str())
            .def(pybind11::init< std::shared_ptr<ExecutionConfiguration>,
                                 const std::string&, Scalar,
                                 const unsigned int,
                                 const std::string&,
                                 const std::string&,
                                 const std::vector<std::string>&,
                                 const std::string&,
                                 unsigned int >())
            ;
    }
#endif
#endif // ENABLE_HIP
#endif // _PATCH_ENERGY_JIT_GPU_H_
