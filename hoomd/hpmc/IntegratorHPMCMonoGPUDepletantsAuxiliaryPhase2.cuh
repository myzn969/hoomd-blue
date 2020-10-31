// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#include <hip/hip_runtime.h>

#include "IntegratorHPMCMonoGPUDepletantsAuxiliaryTypes.cuh"
#include "IntegratorHPMCMonoGPUTypes.cuh"

// include the kernel definition
#include "IntegratorHPMCMonoGPUDepletantsAuxiliaryPhase2.inc"

namespace hpmc {

namespace gpu {

#ifdef __HIP_PLATFORM_NVCC__
#define MAX_BLOCK_SIZE 1024
#define MIN_BLOCK_SIZE 256 // a reasonable minimum to limit compile time
#else
#define MAX_BLOCK_SIZE 1024
#define MIN_BLOCK_SIZE 1024 // on AMD, we do not use __launch_bounds__
#endif

#ifdef __HIPCC__
namespace kernel
{

//! Launcher for hpmc_insert_depletants_phase2 kernel with templated launch bounds
template< class Shape, unsigned int cur_launch_bounds>
void depletants_launcher_phase2(const hpmc_args_t& args,
    const hpmc_implicit_args_t& implicit_args,
    const hpmc_auxiliary_args_t& auxiliary_args,
    const typename Shape::param_type *params,
    unsigned int max_threads,
    detail::int2type<cur_launch_bounds>)
    {
    if (max_threads == cur_launch_bounds*MIN_BLOCK_SIZE)
        {
        // determine the maximum block size and clamp the input block size down
        static int max_block_size = -1;
        static hipFuncAttributes attr;
        constexpr unsigned int launch_bounds_nonzero = cur_launch_bounds > 0 ? cur_launch_bounds : 1;
        if (max_block_size == -1)
            {
            hipFuncGetAttributes(&attr,
                reinterpret_cast<const void*>(&kernel::hpmc_insert_depletants_phase2<Shape, launch_bounds_nonzero*MIN_BLOCK_SIZE>));
            max_block_size = attr.maxThreadsPerBlock;
            }

        // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
        unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);

        unsigned int tpp = min(args.tpp,block_size);
        tpp = std::min((unsigned int) args.devprop.maxThreadsDim[2], tpp); // clamp blockDim.z
        unsigned int n_groups = block_size / tpp;

        unsigned int max_queue_size = n_groups*tpp;
        unsigned int max_depletant_queue_size = n_groups;

        const unsigned int min_shared_bytes = args.num_types * sizeof(typename Shape::param_type) +
                   args.overlap_idx.getNumElements() * sizeof(unsigned int);

        unsigned int shared_bytes = n_groups *(sizeof(Scalar4) + sizeof(Scalar3) + sizeof(unsigned int)) +
                                    max_queue_size*2*sizeof(unsigned int) +
                                    max_depletant_queue_size*(sizeof(unsigned int) + sizeof(float)) +
                                    n_groups*auxiliary_args.max_len*(sizeof(unsigned int) + sizeof(float)) +
                                    min_shared_bytes;

        if (min_shared_bytes >= args.devprop.sharedMemPerBlock)
            throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of particle types or size of shape parameters");

        while (shared_bytes + attr.sharedSizeBytes >= args.devprop.sharedMemPerBlock)
            {
            block_size -= args.devprop.warpSize;
            if (block_size == 0)
                throw std::runtime_error("Insufficient shared memory for HPMC kernel");
            tpp = min(tpp, block_size);
            tpp = std::min((unsigned int) args.devprop.maxThreadsDim[2], tpp); // clamp blockDim.z
            n_groups = block_size / tpp;

            max_queue_size = n_groups*tpp;
            max_depletant_queue_size = n_groups;

            shared_bytes = n_groups * (sizeof(Scalar4) + sizeof(Scalar3) + sizeof(unsigned int)) +
                           max_queue_size*2*sizeof(unsigned int) +
                           max_depletant_queue_size*(sizeof(unsigned int) + sizeof(float)) +
                           n_groups*auxiliary_args.max_len*(sizeof(unsigned int) + sizeof(float)) +
                           min_shared_bytes;
            }


        // determine dynamically requested shared memory
        unsigned int base_shared_bytes = shared_bytes + attr.sharedSizeBytes;
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
        dim3 threads(1, n_groups, tpp);

        for (int idev = auxiliary_args.gpu_partition_rank.getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            auto range = auxiliary_args.gpu_partition_rank.getRangeAndSetGPU(idev);

            unsigned int nwork = range.second - range.first;

            // add ghosts to final range
            if (idev == (int)auxiliary_args.gpu_partition_rank.getNumActiveGPUs()-1 && auxiliary_args.add_ghosts)
                nwork += auxiliary_args.n_ghosts;

            if (!nwork) continue;

            unsigned int blocks_per_particle = auxiliary_args.nwork_local[idev]/
                (implicit_args.depletants_per_thread*n_groups*tpp) + 1;

            dim3 grid( nwork, 2*blocks_per_particle, 1);

            if (blocks_per_particle > args.devprop.maxGridSize[1])
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
            assert(auxiliary_args.d_tag);
            assert(auxiliary_args.d_vel);
            assert(auxiliary_args.d_trial_vel);
            assert(auxiliary_args.d_n_depletants_ntrial);

            hipLaunchKernelGGL((kernel::hpmc_insert_depletants_phase2<Shape, launch_bounds_nonzero*MIN_BLOCK_SIZE>),
                dim3(grid), dim3(threads), shared_bytes, auxiliary_args.streams_phase2[idev],
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
                                 auxiliary_args.gamma,
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
                                 args.d_type_params,
                                 auxiliary_args.r_cut_patch,
                                 auxiliary_args.d_additive_cutoff,
                                 auxiliary_args.d_charge,
                                 auxiliary_args.d_diameter);
            }
        }
    else
        {
        depletants_launcher_phase2<Shape>(args,
            implicit_args,
            auxiliary_args,
            params,
            max_threads,
            detail::int2type<cur_launch_bounds/2>());
        }
    }

} // end namespace kernel

//! Kernel driver for precompiled kernel::insert_depletants_phase2() without patch interaction
/*! \param args Bundled arguments
    \param implicit_args Bundled arguments related to depletants
    \param implicit_args Bundled arguments related to auxiliary variable depletants
    \param d_params Per-type shape parameters

    This templatized method is the kernel driver for HPMC update of any shape. It is instantiated for every shape at the
    bottom of this file.

    \ingroup hpmc_kernels
*/
template< class Shape >
void hpmc_depletants_auxiliary_phase2(const hpmc_args_t& args,
                                       const hpmc_implicit_args_t& implicit_args,
                                       const hpmc_auxiliary_args_t& auxiliary_args,
                                       const typename Shape::param_type *params)
    {
    // select the kernel template according to the next power of two of the block size
    unsigned int launch_bounds = MIN_BLOCK_SIZE;
    while (launch_bounds < args.block_size)
        launch_bounds *= 2;

    kernel::depletants_launcher_phase2<Shape>(args,
        implicit_args,
        auxiliary_args,
        params,
        launch_bounds,
        detail::int2type<(int)MAX_BLOCK_SIZE/MIN_BLOCK_SIZE>());
    }
#endif

#undef MAX_BLOCK_SIZE
#undef MIN_BLOCK_SIZE

} // end namespace gpu

} // end namespace hpmc
