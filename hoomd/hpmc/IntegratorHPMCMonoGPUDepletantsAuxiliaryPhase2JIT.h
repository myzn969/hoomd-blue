// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#include "IntegratorHPMCMonoGPUTypes.cuh"
#include "IntegratorHPMCMonoGPUDepletantsAuxiliaryTypes.cuh"
#include "IntegratorHPMCMonoGPUDepletants.cuh"

#include "hoomd/jit/PatchEnergyJITGPU.h"
#include "hoomd/jit/PatchEnergyJITUnionGPU.h"
#include "hoomd/jit/JITKernel.h"

namespace hpmc {

namespace gpu {

//! A common interface for patch energy evaluations on the GPU
template<class Shape>
class JITDepletantsAuxiliaryPhase2
    {
    public:
        JITDepletantsAuxiliaryPhase2(std::shared_ptr<const ExecutionConfiguration>& exec_conf)
            : m_exec_conf(exec_conf)
            {
            #ifdef __HIP_PLATFORM_NVCC__
            for (unsigned int i = 32; i <= (unsigned int) exec_conf->dev_prop.maxThreadsPerBlock; i *= 2)
               m_launch_bounds.push_back(i);
            #endif
            }

        //! Launch the kernel
        /*! \params args Kernel arguments
         */
        virtual void operator()(const hpmc_args_t& args,
            const hpmc_implicit_args_t& implicit_args,
            const hpmc_auxiliary_args_t& auxiliary_args,
            const typename Shape::param_type *params) = 0;

    protected:
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; /// The execution configuration
        std::vector<unsigned int> m_launch_bounds; // predefined kernel launchb bounds
    };

template<class Shape, class JIT>
class JITDepletantsAuxiliaryPhase2Impl {};

//! Launcher for hpmc_insert_depletants_phase2 kernel with templated launch bounds
template<class Shape>
class JITDepletantsAuxiliaryPhase2Impl<Shape, PatchEnergyJITGPU> : public JITDepletantsAuxiliaryPhase2<Shape>
    {
    public:
        using JIT = PatchEnergyJITGPU;

        const std::string kernel_code = R"(
            #include "hoomd/hpmc/Shapes.h"
            #include "hoomd/hpmc/IntegratorHPMCMonoGPUDepletantsAuxiliaryPhase2.inc"
        )";
        const std::string kernel_name = "hpmc::gpu::kernel::hpmc_insert_depletants_phase2";

        JITDepletantsAuxiliaryPhase2Impl(std::shared_ptr<const ExecutionConfiguration>& exec_conf,
                       std::shared_ptr<JIT> jit)
            : JITDepletantsAuxiliaryPhase2<Shape>(exec_conf),
              m_kernel(exec_conf, kernel_code, kernel_name, jit)
            { }

        virtual void operator()(
            const hpmc_args_t& args,
            const hpmc_implicit_args_t& implicit_args,
            const hpmc_auxiliary_args_t& auxiliary_args,
            const typename Shape::param_type *params)
            {
            // determine the maximum block size and clamp the input block size down
            unsigned int bounds = 0;
            for (auto b: this->m_launch_bounds)
                {
                if (b >= args.block_size)
                    {
                    bounds = b;
                    break;
                    }
                }

            const unsigned int eval_threads = auxiliary_args.eval_threads;
            unsigned int max_block_size = m_kernel.getFactory().getKernelMaxThreads<Shape>(0, bounds, true, eval_threads, false); // fixme GPU 0

            // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
            unsigned int block_size = std::min(args.block_size, (unsigned int)max_block_size);

            unsigned int tpp = std::min(args.tpp,block_size);
            while (eval_threads*tpp > block_size || block_size % (eval_threads*tpp) != 0)
                {
                tpp--;
                }

            tpp = std::min((unsigned int) args.devprop.maxThreadsDim[2], tpp); // clamp blockDim.z
            unsigned int n_groups = block_size / (eval_threads*tpp);

            unsigned int max_queue_size = n_groups*tpp;
            unsigned int max_depletant_queue_size = n_groups;

            const unsigned int min_shared_bytes = args.num_types * sizeof(typename Shape::param_type) +
                       args.overlap_idx.getNumElements() * sizeof(unsigned int) +
                       args.num_types * sizeof(Scalar);

            unsigned int shared_bytes = n_groups *(sizeof(Scalar4) + sizeof(Scalar3) + sizeof(unsigned int)) +
                                        max_queue_size*2*sizeof(unsigned int) +
                                        max_depletant_queue_size*(2*sizeof(unsigned int) + sizeof(float)) +
                                        n_groups*auxiliary_args.max_len*(sizeof(unsigned int) + sizeof(float)) +
                                        min_shared_bytes;

            if (min_shared_bytes >= args.devprop.sharedMemPerBlock)
                throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of particle types or size of shape parameters");

            unsigned int shared_size_bytes = m_kernel.getFactory().getKernelSharedSize<Shape>(0, bounds, true, eval_threads, false); //GPU 0
            while (shared_bytes + shared_size_bytes >= args.devprop.sharedMemPerBlock)
                {
                block_size -= args.devprop.warpSize;
                if (block_size == 0)
                    throw std::runtime_error("Insufficient shared memory for HPMC kernel");
                tpp = std::min(tpp, block_size);
                while (eval_threads*tpp > block_size || block_size % (eval_threads*tpp) != 0)
                    {
                    tpp--;
                    }

                tpp = std::min((unsigned int) args.devprop.maxThreadsDim[2], tpp); // clamp blockDim.z
                n_groups = block_size / (eval_threads*tpp);

                max_queue_size = n_groups*tpp;
                max_depletant_queue_size = n_groups;

                shared_bytes = n_groups * (sizeof(Scalar4) + sizeof(Scalar3) + sizeof(unsigned int)) +
                               max_queue_size*2*sizeof(unsigned int) +
                               max_depletant_queue_size*(2*sizeof(unsigned int) + sizeof(float)) +
                               n_groups*auxiliary_args.max_len*(sizeof(unsigned int) + sizeof(float)) +
                               min_shared_bytes;
                }


            // determine dynamically requested shared memory
            unsigned int base_shared_bytes = shared_bytes + shared_size_bytes;
            unsigned int max_extra_bytes = args.devprop.sharedMemPerBlock - base_shared_bytes;
            char *ptr = (char *) nullptr;
            unsigned int available_bytes = max_extra_bytes;
            for (unsigned int i = 0; i < args.num_types; ++i)
                {
                params[i].load_shared(ptr, available_bytes, args.d_type_params[i]);
                }
            unsigned int extra_bytes = max_extra_bytes - available_bytes;
            shared_bytes += extra_bytes;

            // setup the grid to run the kernel
            dim3 threads(eval_threads, n_groups, tpp);

            // iterate over particles of this rank only
            for (int idev = auxiliary_args.gpu_partition_rank.getNumActiveGPUs() - 1; idev >= 0; --idev)
                {
                auto range = auxiliary_args.gpu_partition_rank.getRangeAndSetGPU(idev);

                if (range.first == range.second)
                    continue;

                // setup up global scope variables
                m_kernel.setup<Shape>(idev, auxiliary_args.streams_phase2[idev], bounds, true, eval_threads, false);

                unsigned int blocks_per_particle = auxiliary_args.nwork_local[idev]/
                    (implicit_args.depletants_per_thread*n_groups*tpp) + 1;

                dim3 grid( range.second-range.first, 2*blocks_per_particle, 1);

                if (blocks_per_particle > (unsigned int ) args.devprop.maxGridSize[1])
                    {
                    grid.y = args.devprop.maxGridSize[1];
                    grid.z = 2*blocks_per_particle/args.devprop.maxGridSize[1]+1;
                    }

                assert(args.d_trial_postype);
                assert(args.d_trial_orientation);
                assert(args.d_trial_move_type);
                assert(args.d_postype);
                assert(args.d_orientation);
                assert(args.d_counters);
                assert(args.d_excell_idx);
                assert(args.d_excell_size);
                assert(args.d_check_overlaps);
                assert(implicit_args.d_implicit_count);
                assert(args.d_update_order_by_ptl);
                assert(auxiliary_args.d_tag);
                assert(auxiliary_args.d_vel);
                assert(auxiliary_args.d_trial_vel);
                assert(auxiliary_args.d_n_depletants_ntrial);
                assert(args.d_type_params);

                // configure the kernel
                auto launcher = m_kernel.getFactory()
                    .configureKernel<Shape>(idev, grid, threads, shared_bytes, auxiliary_args.streams_phase2[idev],
                        bounds, true, eval_threads, false);

                CUresult res = launcher(
                                 args.d_trial_postype,
                                 args.d_trial_orientation,
                                 args.d_trial_move_type,
                                 args.d_postype,
                                 args.d_orientation,
                                 args.d_counters + idev*args.counters_pitch,
                                 args.d_excell_idx,
                                 args.d_excell_size,
                                 args.excli,
                                 args.cell_dim,
                                 args.ghost_width,
                                 args.ci,
                                 args.N,
                                 args.num_types,
                                 args.seed,
                                 args.d_check_overlaps,
                                 args.overlap_idx,
                                 args.timestep,
                                 args.dim,
                                 args.box,
                                 args.select,
                                 params,
                                 max_queue_size,
                                 max_extra_bytes,
                                 implicit_args.depletant_type_a,
                                 implicit_args.depletant_idx,
                                 implicit_args.d_implicit_count + idev*implicit_args.implicit_counters_pitch,
                                 auxiliary_args.ntrial,
                                 auxiliary_args.d_tag,
                                 auxiliary_args.d_vel,
                                 auxiliary_args.d_trial_vel,
                                 auxiliary_args.d_deltaF_nor_nneigh,
                                 auxiliary_args.d_deltaF_nor_nlist,
                                 auxiliary_args.d_deltaF_nor_len,
                                 auxiliary_args.d_deltaF_nor_k,
                                 auxiliary_args.d_deltaF_nor_energy,
                                 auxiliary_args.d_deltaF_nor,
                                 auxiliary_args.deltaF_nor_maxlen,
                                 auxiliary_args.d_overflow_nor,
                                 implicit_args.repulsive,
                                 range.first,
                                 max_depletant_queue_size,
                                 auxiliary_args.d_n_depletants_ntrial,
                                 auxiliary_args.max_len,
                                 auxiliary_args.d_req_len,
                                 auxiliary_args.work_offset[idev],
                                 auxiliary_args.nwork_local[idev],
                                 implicit_args.max_n_depletants[idev],
                                 args.d_type_params,
                                 auxiliary_args.r_cut_patch,
                                 auxiliary_args.d_additive_cutoff,
                                 auxiliary_args.d_charge,
                                 auxiliary_args.d_diameter);

                if (res != CUDA_SUCCESS)
                    {
                    char *error;
                    cuGetErrorString(res, const_cast<const char **>(&error));
                    throw std::runtime_error("Error launching NVRTC kernel: "+std::string(error));
                    }
                }
            }

    private:
        jit::JITKernel<JIT> m_kernel; // The kernel object
    };

//! Launcher for hpmc_insert_depletants_phase2 kernel with templated launch bounds
template<class Shape>
class JITDepletantsAuxiliaryPhase2Impl<Shape, PatchEnergyJITUnionGPU> : public JITDepletantsAuxiliaryPhase2<Shape>
    {
    public:
        using JIT = PatchEnergyJITUnionGPU;

        const std::string kernel_code = R"(
            #include "hoomd/hpmc/Shapes.h"
            #include "hoomd/hpmc/IntegratorHPMCMonoGPUDepletantsAuxiliaryPhase2.inc"
        )";
        const std::string kernel_name = "hpmc::gpu::kernel::hpmc_insert_depletants_phase2";

        JITDepletantsAuxiliaryPhase2Impl(std::shared_ptr<const ExecutionConfiguration>& exec_conf,
                       std::shared_ptr<JIT> jit)
            : JITDepletantsAuxiliaryPhase2<Shape>(exec_conf),
              m_kernel(exec_conf, kernel_code, kernel_name, jit)
            { }

        virtual void operator()(
            const hpmc_args_t& args,
            const hpmc_implicit_args_t& implicit_args,
            const hpmc_auxiliary_args_t& auxiliary_args,
            const typename Shape::param_type *params)
            {
            // determine the maximum block size and clamp the input block size down
            unsigned int bounds = 0;
            for (auto b: this->m_launch_bounds)
                {
                if (b >= args.block_size)
                    {
                    bounds = b;
                    break;
                    }
                }

            const unsigned int eval_threads = auxiliary_args.eval_threads;
            unsigned int max_block_size = m_kernel.getFactory().getKernelMaxThreads<Shape>(0, bounds, true, eval_threads, true); // fixme GPU 0

            // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
            unsigned int block_size = std::min(args.block_size, (unsigned int)max_block_size);

            unsigned int tpp = std::min(args.tpp,block_size);
            while (eval_threads*tpp > block_size || block_size % (eval_threads*tpp) != 0)
                {
                tpp--;
                }

            tpp = std::min((unsigned int) args.devprop.maxThreadsDim[2], tpp); // clamp blockDim.z
            unsigned int n_groups = block_size / (eval_threads*tpp);

            unsigned int max_queue_size = n_groups*tpp;
            unsigned int max_depletant_queue_size = n_groups;

            const unsigned int min_shared_bytes = args.num_types * sizeof(typename Shape::param_type) +
                       args.overlap_idx.getNumElements() * sizeof(unsigned int) +
                       args.num_types * sizeof(jit::union_params_t) +
                       args.num_types * sizeof(Scalar);

            unsigned int shared_bytes = n_groups *(sizeof(Scalar4) + sizeof(Scalar3) + sizeof(unsigned int)) +
                                        max_queue_size*2*sizeof(unsigned int) +
                                        max_depletant_queue_size*(2*sizeof(unsigned int) + sizeof(float)) +
                                        n_groups*auxiliary_args.max_len*(sizeof(unsigned int) + sizeof(float)) +
                                        min_shared_bytes;

            if (min_shared_bytes >= args.devprop.sharedMemPerBlock)
                throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of particle types or size of shape parameters");

            unsigned int shared_size_bytes = m_kernel.getFactory().getKernelSharedSize<Shape>(0, bounds, true, eval_threads, true); //GPU 0
            while (shared_bytes + shared_size_bytes >= args.devprop.sharedMemPerBlock)
                {
                block_size -= args.devprop.warpSize;
                if (block_size == 0)
                    throw std::runtime_error("Insufficient shared memory for HPMC kernel");
                tpp = std::min(tpp, block_size);
                while (eval_threads*tpp > block_size || block_size % (eval_threads*tpp) != 0)
                    {
                    tpp--;
                    }

                tpp = std::min((unsigned int) args.devprop.maxThreadsDim[2], tpp); // clamp blockDim.z
                n_groups = block_size / (tpp*eval_threads);

                max_queue_size = n_groups*tpp;
                max_depletant_queue_size = n_groups;

                shared_bytes = n_groups * (sizeof(Scalar4) + sizeof(Scalar3) + sizeof(unsigned int)) +
                               max_queue_size*2*sizeof(unsigned int) +
                               max_depletant_queue_size*(2*sizeof(unsigned int) + sizeof(float)) +
                               n_groups*auxiliary_args.max_len*(sizeof(unsigned int) + sizeof(float)) +
                               min_shared_bytes;
                }


            // determine dynamically requested shared memory
            unsigned int base_shared_bytes = shared_bytes + shared_size_bytes;
            unsigned int max_extra_bytes = args.devprop.sharedMemPerBlock - base_shared_bytes;
            char *ptr = (char *) nullptr;
            unsigned int available_bytes = max_extra_bytes;
            for (unsigned int i = 0; i < args.num_types; ++i)
                {
                params[i].load_shared(ptr, available_bytes, args.d_type_params[i]);
                }

            // determine dynamically requested shared memory for union parameters
            for (unsigned int i = 0; i < m_kernel.getJIT()->getDeviceParams().size(); ++i)
                {
                m_kernel.getJIT()->getDeviceParams()[i].load_shared(ptr, available_bytes, args.d_type_params[i] >> Shape::getTuningBits());
                }

            unsigned int extra_bytes = max_extra_bytes - available_bytes;
            shared_bytes += extra_bytes;

            // setup the grid to run the kernel
            dim3 threads(eval_threads, n_groups, tpp);

            // iterate over particles of this rank only
            for (int idev = auxiliary_args.gpu_partition_rank.getNumActiveGPUs() - 1; idev >= 0; --idev)
                {
                auto range = auxiliary_args.gpu_partition_rank.getRangeAndSetGPU(idev);

                // setup up global scope variables
                m_kernel.setup<Shape>(idev, auxiliary_args.streams_phase2[idev], bounds, true, eval_threads, true);

                unsigned int nwork = range.second - range.first;

                // add ghosts to final range
                if (idev == (int)auxiliary_args.gpu_partition_rank.getNumActiveGPUs()-1 && auxiliary_args.add_ghosts)
                     nwork += auxiliary_args.n_ghosts;

                if (!nwork) continue;

                unsigned int blocks_per_particle = auxiliary_args.nwork_local[idev]/
                    (implicit_args.depletants_per_thread*n_groups*tpp) + 1;

                dim3 grid( nwork, 2*blocks_per_particle, 1);

                if (blocks_per_particle > (unsigned int) args.devprop.maxGridSize[1])
                    {
                    grid.y = args.devprop.maxGridSize[1];
                    grid.z = 2*blocks_per_particle/args.devprop.maxGridSize[1]+1;
                    }

                assert(args.d_trial_postype);
                assert(args.d_trial_orientation);
                assert(args.d_trial_move_type);
                assert(args.d_postype);
                assert(args.d_orientation);
                assert(args.d_counters);
                assert(args.d_excell_idx);
                assert(args.d_excell_size);
                assert(args.d_check_overlaps);
                assert(implicit_args.d_implicit_count);
                assert(args.d_update_order_by_ptl);
                assert(auxiliary_args.d_tag);
                assert(auxiliary_args.d_vel);
                assert(auxiliary_args.d_trial_vel);
                assert(auxiliary_args.d_n_depletants_ntrial);
                assert(args.d_type_params);

                // configure the kernel
                auto launcher = m_kernel.getFactory()
                    .configureKernel<Shape>(idev, grid, threads, shared_bytes, auxiliary_args.streams_phase2[idev],
                        bounds, true, eval_threads, true);

                CUresult res = launcher(
                                 args.d_trial_postype,
                                 args.d_trial_orientation,
                                 args.d_trial_move_type,
                                 args.d_postype,
                                 args.d_orientation,
                                 args.d_counters + idev*args.counters_pitch,
                                 args.d_excell_idx,
                                 args.d_excell_size,
                                 args.excli,
                                 args.cell_dim,
                                 args.ghost_width,
                                 args.ci,
                                 args.N,
                                 args.num_types,
                                 args.seed,
                                 args.d_check_overlaps,
                                 args.overlap_idx,
                                 args.timestep,
                                 args.dim,
                                 args.box,
                                 args.select,
                                 params,
                                 max_queue_size,
                                 max_extra_bytes,
                                 implicit_args.depletant_type_a,
                                 implicit_args.depletant_idx,
                                 implicit_args.d_implicit_count + idev*implicit_args.implicit_counters_pitch,
                                 auxiliary_args.ntrial,
                                 auxiliary_args.d_tag,
                                 auxiliary_args.d_vel,
                                 auxiliary_args.d_trial_vel,
                                 auxiliary_args.d_deltaF_nor_nneigh,
                                 auxiliary_args.d_deltaF_nor_nlist,
                                 auxiliary_args.d_deltaF_nor_len,
                                 auxiliary_args.d_deltaF_nor_k,
                                 auxiliary_args.d_deltaF_nor_energy,
                                 auxiliary_args.d_deltaF_nor,
                                 auxiliary_args.deltaF_nor_maxlen,
                                 auxiliary_args.d_overflow_nor,
                                 implicit_args.repulsive,
                                 range.first,
                                 max_depletant_queue_size,
                                 auxiliary_args.d_n_depletants_ntrial,
                                 auxiliary_args.max_len,
                                 auxiliary_args.d_req_len,
                                 auxiliary_args.work_offset[idev],
                                 auxiliary_args.nwork_local[idev],
                                 implicit_args.max_n_depletants[idev],
                                 args.d_type_params,
                                 auxiliary_args.r_cut_patch,
                                 auxiliary_args.d_additive_cutoff,
                                 auxiliary_args.d_charge,
                                 auxiliary_args.d_diameter);

                if (res != CUDA_SUCCESS)
                    {
                    char *error;
                    cuGetErrorString(res, const_cast<const char **>(&error));
                    throw std::runtime_error("Error launching NVRTC kernel: "+std::string(error));
                    }
                }
            }

    private:
        jit::JITKernel<JIT> m_kernel; // The kernel object
    };

} // end namespace gpu

} // end namespace hpmc
