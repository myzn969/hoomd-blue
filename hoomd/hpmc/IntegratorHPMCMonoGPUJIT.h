#pragma once

#ifdef ENABLE_HIP

#include <memory>

#include "hoomd/jit/PatchEnergyJITGPU.h"
#include "hoomd/jit/PatchEnergyJITUnionGPU.h"
#include "hoomd/jit/JITKernel.h"

namespace hpmc
{

namespace gpu
{

//! Wraps arguments to kernel::narow_phase_patch functions
struct hpmc_patch_args_t
    {
    //! Construct a hpmc_patch_args_t
    hpmc_patch_args_t(Scalar4 *_d_postype,
                Scalar4 *_d_orientation,
                Scalar4 *_d_trial_postype,
                Scalar4 *_d_trial_orientation,
                const Index3D& _ci,
                const uint3& _cell_dim,
                const Scalar3& _ghost_width,
                const unsigned int _N,
                const unsigned int _N_ghost,
                const unsigned int _num_types,
                const BoxDim& _box,
                const unsigned int *_d_excell_idx,
                const unsigned int *_d_excell_size,
                const Index2D& _excli,
                const Scalar _r_cut_patch,
                const Scalar *_d_additive_cutoff,
                unsigned int *_d_nlist_old,
                unsigned int *_d_nneigh_old,
                float *_d_energy_old,
                unsigned int *_d_nlist_new,
                unsigned int *_d_nneigh_new,
                float *_d_energy_new,
                const unsigned int _maxn,
                unsigned int *_d_overflow,
                const Scalar *_d_charge,
                const Scalar *_d_diameter,
                const unsigned int *_d_reject_out_of_cell,
                const GPUPartition& _gpu_partition,
                const unsigned int _block_size,
                const unsigned int _tpp,
                const unsigned int _eval_threads,
                const hipStream_t _stream,
                const unsigned int *_d_tuner_params)
                : d_postype(_d_postype),
                  d_orientation(_d_orientation),
                  d_trial_postype(_d_trial_postype),
                  d_trial_orientation(_d_trial_orientation),
                  ci(_ci),
                  cell_dim(_cell_dim),
                  ghost_width(_ghost_width),
                  N(_N),
                  N_ghost(_N_ghost),
                  num_types(_num_types),
                  box(_box),
                  d_excell_idx(_d_excell_idx),
                  d_excell_size(_d_excell_size),
                  excli(_excli),
                  r_cut_patch(_r_cut_patch),
                  d_additive_cutoff(_d_additive_cutoff),
                  d_nlist_old(_d_nlist_old),
                  d_nneigh_old(_d_nneigh_old),
                  d_energy_old(_d_energy_old),
                  d_nlist_new(_d_nlist_new),
                  d_nneigh_new(_d_nneigh_new),
                  d_energy_new(_d_energy_new),
                  maxn(_maxn),
                  d_overflow(_d_overflow),
                  d_charge(_d_charge),
                  d_diameter(_d_diameter),
                  d_reject_out_of_cell(_d_reject_out_of_cell),
                  gpu_partition(_gpu_partition),
                  block_size(_block_size),
                  tpp(_tpp),
                  eval_threads(_eval_threads),
                  stream(_stream),
                  d_tuner_params(_d_tuner_params)
        { }

    Scalar4 *d_postype;               //!< postype array
    Scalar4 *d_orientation;           //!< orientation array
    Scalar4 *d_trial_postype;         //!< New positions (and type) of particles
    Scalar4 *d_trial_orientation;     //!< New orientations of particles
    const Index3D& ci;                //!< Cell indexer
    const uint3& cell_dim;            //!< Cell dimensions
    const Scalar3& ghost_width;       //!< Width of the ghost layer
    const unsigned int N;             //!< Number of particles
    const unsigned int N_ghost;       //!< Number of ghost particles
    const unsigned int num_types;     //!< Number of particle types
    const BoxDim& box;                //!< Current simulation box
    const unsigned int *d_excell_idx;       //!< Expanded cell list
    const unsigned int *d_excell_size;//!< Size of expanded cells
    const Index2D& excli;             //!< Excell indexer
    const Scalar r_cut_patch;        //!< Global cutoff radius
    const Scalar *d_additive_cutoff; //!< Additive contribution to cutoff per type
    unsigned int *d_nlist_old;       //!< List of neighbor particle indices, in old configuration of particle i
    unsigned int *d_nneigh_old;      //!< Number of neighbors
    float* d_energy_old;             //!< Evaluated energy terms for every neighbor
    unsigned int *d_nlist_new;       //!< List of neighbor particle indices, in new configuration of particle i
    unsigned int *d_nneigh_new;      //!< Number of neighbors
    float* d_energy_new;             //!< Evaluated energy terms for every neighbor
    const unsigned int maxn;         //!< Max number of neighbors
    unsigned int *d_overflow;        //!< Overflow condition
    const Scalar *d_charge;          //!< Particle charges
    const Scalar *d_diameter;        //!< Particle diameters
    const unsigned int *d_reject_out_of_cell;   //!< Flag if a particle move has been rejected a priori
    const GPUPartition& gpu_partition; //!< split particles among GPUs
    const unsigned int block_size;   //!< Kernel block size
    const unsigned int tpp;          //!< Kernel threads per particle
    const unsigned int eval_threads; //!< Kernel evaluator function threads
    const hipStream_t stream;        //!< Kernel stream
    const unsigned int *d_tuner_params; //!< Tuner parameters on device
    };

//! A common interface for patch energy evaluations on the GPU
template<class Shape>
class JITNarrowPhase
    {
    public:
        JITNarrowPhase(std::shared_ptr<const ExecutionConfiguration>& exec_conf)
            {
            #ifdef __HIP_PLATFORM_NVCC__
            for (unsigned int i = 32; i <= (unsigned int) exec_conf->dev_prop.maxThreadsPerBlock; i *= 2)
               m_launch_bounds.push_back(i);
            #endif
            }

        //! Launch the kernel
        /*! \params args Kernel arguments
         */
        virtual void operator()(const hpmc_patch_args_t& args) = 0;

    protected:
        std::vector<unsigned int> m_launch_bounds; // predefined kernel launchb bounds
    };

template<class Shape, class JIT>
class JITNarrowPhaseImpl {};

//! Narrow phase kernel for simple point-like interactions
template<class Shape>
class JITNarrowPhaseImpl<Shape, PatchEnergyJITGPU> : public JITNarrowPhase<Shape>
    {
    public:
        using JIT = PatchEnergyJITGPU;

        const std::string kernel_code = R"(
            #include "hoomd/hpmc/IntegratorHPMCMonoGPUJIT.inc"
        )";
        const std::string kernel_name = "hpmc::gpu::kernel::hpmc_narrow_phase_patch";

        JITNarrowPhaseImpl(std::shared_ptr<const ExecutionConfiguration>& exec_conf,
                       std::shared_ptr<JIT> jit)
            : JITNarrowPhase<Shape>(exec_conf), m_exec_conf(exec_conf),
              m_kernel(exec_conf, kernel_code, kernel_name, jit)
            { }

        virtual void operator()(const hpmc_patch_args_t& args)
            {
            #ifdef __HIP_PLATFORM_NVCC__
            assert(args.d_postype);
            assert(args.d_orientation);

            unsigned int block_size = args.block_size;
            unsigned int req_tpp = args.tpp;
            unsigned int eval_threads = args.eval_threads;

            unsigned int bounds = 0;
            for (auto b: this->m_launch_bounds)
                {
                if (b >= block_size)
                    {
                    bounds = b;
                    break;
                    }
                }

            m_kernel.setup<Shape>(args.stream, eval_threads, bounds);

            // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
            unsigned int run_block_size = std::min(block_size,
                m_kernel.getFactory().getKernelMaxThreads<Shape>(0, eval_threads, bounds)); // fixme GPU 0

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

            unsigned int kernel_shared_bytes = m_kernel.getFactory()
                .getKernelSharedSize<Shape>(0, eval_threads, bounds); //fixme GPU 0
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
                auto launcher = m_kernel.getFactory()
                    .configureKernel<Shape>(idev, grid, thread, shared_bytes, args.stream,
                        eval_threads, bounds);

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
            #endif
            }

    private:
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; /// The execution configuration
        jit::JITKernel<JIT> m_kernel;                        /// The kernel object
    };

//! Narrow phase kernel for unions of points
template<class Shape>
class JITNarrowPhaseImpl<Shape, PatchEnergyJITUnionGPU> : public JITNarrowPhase<Shape>
    {
    public:
        using JIT = PatchEnergyJITUnionGPU;

        const std::string kernel_code = R"(
            #define UNION_EVAL // use union evaluator
            #include "hoomd/hpmc/IntegratorHPMCMonoGPUJIT.inc"
        )";
        const std::string kernel_name = "hpmc::gpu::kernel::hpmc_narrow_phase_patch_union";

        JITNarrowPhaseImpl(std::shared_ptr<const ExecutionConfiguration>& exec_conf,
                       std::shared_ptr<JIT> jit)
            : JITNarrowPhase<Shape>(exec_conf), m_exec_conf(exec_conf),
              m_kernel(exec_conf, kernel_code, kernel_name, jit)
            { }

        //! Launch the kernel
        virtual void operator()(const hpmc_patch_args_t& args)
            {
            #ifdef __HIP_PLATFORM_NVCC__
            assert(args.d_postype);
            assert(args.d_orientation);

            unsigned int block_size = args.block_size;
            unsigned int req_tpp = args.tpp;
            unsigned int eval_threads = args.eval_threads;

            unsigned int bounds = 0;
            for (auto b: this->m_launch_bounds)
                {
                if (b >= block_size)
                    {
                    bounds = b;
                    break;
                    }
                }
            const unsigned int *d_type_params = args.d_tuner_params+1;

            m_kernel.setup<Shape>(args.stream, eval_threads, bounds);

            unsigned int run_block_size = std::min(block_size,
                m_kernel.getFactory().getKernelMaxThreads<Shape>(0, eval_threads, bounds)); // fixme GPU 0

            unsigned int tpp = std::min(req_tpp,run_block_size);
            while (eval_threads*tpp > run_block_size || run_block_size % (eval_threads*tpp) != 0)
                {
                tpp--;
                }
            auto& devprop = this->m_exec_conf->dev_prop;
            tpp = std::min((unsigned int) devprop.maxThreadsDim[2], tpp);

            unsigned int n_groups = run_block_size/(tpp*eval_threads);
            unsigned int max_queue_size = n_groups*tpp;

            const unsigned int min_shared_bytes = args.num_types * sizeof(Scalar) +
                m_kernel.getJIT()->getDeviceParams().size()*sizeof(jit::union_params_t);

            unsigned int shared_bytes = n_groups * (4*sizeof(unsigned int) + 2*sizeof(Scalar4) + 2*sizeof(Scalar3) + 2*sizeof(Scalar))
                + max_queue_size * 2 * sizeof(unsigned int)
                + min_shared_bytes;

            if (min_shared_bytes >= devprop.sharedMemPerBlock)
                throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of particle types or size of shape parameters");

            unsigned int kernel_shared_bytes = m_kernel.getFactory()
                .getKernelSharedSize<Shape>(0, eval_threads, bounds); //fixme GPU 0
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

            // allocate some extra shared mem to store union shape parameters
            unsigned int max_extra_bytes = this->m_exec_conf->dev_prop.sharedMemPerBlock - shared_bytes - kernel_shared_bytes;

            // determine dynamically requested shared memory
            char *ptr = (char *)nullptr;
            unsigned int available_bytes = max_extra_bytes;
            for (unsigned int i = 0; i < m_kernel.getJIT()->getDeviceParams().size(); ++i)
                {
                m_kernel.getJIT()->getDeviceParams()[i].load_shared(ptr, available_bytes, d_type_params[i]);
                }
            unsigned int extra_bytes = max_extra_bytes - available_bytes;
            shared_bytes += extra_bytes;

            dim3 thread(eval_threads, n_groups, tpp);

            auto& gpu_partition = args.gpu_partition;

            for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
                {
                auto range = gpu_partition.getRangeAndSetGPU(idev);

                unsigned int nwork = range.second - range.first;
                const unsigned int num_blocks = (nwork + n_groups - 1)/n_groups;

                dim3 grid(num_blocks, 1, 1);

                unsigned int N_old = args.N + args.N_ghost;

                // configure the kernel
                auto launcher = m_kernel.getFactory()
                    .configureKernel<Shape>(idev, grid, thread, shared_bytes, args.stream,
                        eval_threads, bounds);

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
                    max_extra_bytes,
                    d_type_params);

                if (res != CUDA_SUCCESS)
                    {
                    char *error;
                    cuGetErrorString(res, const_cast<const char **>(&error));
                    throw std::runtime_error("Error launching NVRTC kernel: "+std::string(error));
                    }
                }
            #endif
            }

    private:
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; /// The execution configuration
        jit::JITKernel<JIT> m_kernel; // The kernel object
    };

} // end namespace gpu
} // end namespace hpmc

#endif
