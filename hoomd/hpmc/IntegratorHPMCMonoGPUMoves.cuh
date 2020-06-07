// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#include <hip/hip_runtime.h>
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/BoxDim.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/hpmc/Moves.h"
#include "hoomd/GPUPartition.cuh"
#include "hoomd/hpmc/HPMCCounters.h"

#include "GPUHelpers.cuh"

// base data types
#include "IntegratorHPMCMonoGPUTypes.cuh"

namespace hpmc {

namespace gpu {

#ifdef __HIPCC__
namespace kernel
{

//! Propose trial moves
template< class Shape, unsigned int dim >
__global__ void hpmc_gen_moves(const Scalar4 *d_postype,
                           const Scalar4 *d_orientation,
                           const Scalar4 *d_vel,
                           const unsigned int N,
                           const Index3D ci,
                           const uint3 cell_dim,
                           const Scalar3 ghost_width,
                           const unsigned int num_types,
                           const unsigned int seed,
                           const Scalar* d_d,
                           const Scalar* d_a,
                           const unsigned int move_ratio,
                           const unsigned int timestep,
                           const BoxDim box,
                           const unsigned int select,
                           const Scalar3 ghost_fraction,
                           const bool domain_decomposition,
                           const bool have_auxilliary_variable,
                           Scalar4 *d_trial_postype,
                           Scalar4 *d_trial_orientation,
                           Scalar4 *d_trial_vel,
                           unsigned int *d_trial_move_type,
                           unsigned int *d_reject_out_of_cell,
                           unsigned int *d_reject,
                           const typename Shape::param_type *d_params)
    {
    // load the per type pair parameters into shared memory
    HIP_DYNAMIC_SHARED( char, s_data)

    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    Scalar *s_d = (Scalar *)(s_params + num_types);
    Scalar *s_a = (Scalar *)(s_d + num_types);

    // copy over parameters one int per thread for fast loads
        {
        unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
        unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;
        unsigned int param_size = num_types*sizeof(typename Shape::param_type) / sizeof(int);

        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < param_size)
                {
                ((int *)s_params)[cur_offset + tidx] = ((int *)d_params)[cur_offset + tidx];
                }
            }

        for (unsigned int cur_offset = 0; cur_offset < num_types; cur_offset += block_size)
            {
            if (cur_offset + tidx < num_types)
                {
                s_a[cur_offset + tidx] = d_a[cur_offset + tidx];
                s_d[cur_offset + tidx] = d_d[cur_offset + tidx];
                }
            }
        }

    __syncthreads();

    // identify the particle that this thread handles
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // return early if we are not handling a particle
    if (idx >= N)
        return;

    // read in the position and orientation of our particle.
    Scalar4 postype_i = d_postype[idx];
    Scalar4 orientation_i = make_scalar4(1,0,0,0);

    unsigned int typ_i = __scalar_as_int(postype_i.w);
    Shape shape_i(quat<Scalar>(orientation_i), s_params[typ_i]);

    if (shape_i.hasOrientation())
        orientation_i = d_orientation[idx];

    shape_i.orientation = quat<Scalar>(orientation_i);

    vec3<Scalar> pos_i = vec3<Scalar>(postype_i);
    unsigned int old_cell = computeParticleCell(vec_to_scalar3(pos_i), box, ghost_width,
        cell_dim, ci, true);

    // for domain decomposition simulations, we need to leave all particles in the inactive region alone
    // in order to avoid even more divergence, this is done by setting the move_active flag
    // overlap checks are still processed, but the final move acceptance will be skipped
    bool move_active = true;
    if (domain_decomposition && !isActive(make_scalar3(postype_i.x, postype_i.y, postype_i.z), box, ghost_fraction))
        move_active = false;

    // make the move
    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::HPMCMonoTrialMove, idx, seed, select, timestep);

    // do not move particles that are outside the boundaries
    unsigned int reject = old_cell >= ci.getNumElements();

    unsigned int move_type_select = hoomd::UniformIntDistribution(0xffff)(rng);
    bool move_type_translate = !shape_i.hasOrientation() || (move_type_select < move_ratio);

    if (move_active)
        {
        if (move_type_translate)
            {
            move_translate(pos_i, rng, s_d[typ_i], dim);

            // need to reject any move that puts the particle in the inactive region
            if (domain_decomposition && !isActive(vec_to_scalar3(pos_i), box, ghost_fraction))
                move_active = false;
            }
        else
            {
            move_rotate<dim>(shape_i.orientation, rng, s_a[typ_i]);
            }
        }

    if (move_active && move_type_translate)
        {
        // check if the particle remains in its cell
        Scalar3 xnew_i = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
        unsigned int new_cell = computeParticleCell(xnew_i, box, ghost_width,
            cell_dim, ci, true);

        if (new_cell != old_cell)
            reject = 1;
        }

    if (have_auxilliary_variable)
        {
        // generate a new random auxillary variable
        unsigned int seed_i_new = hoomd::detail::generate_u32(rng);

        // store it in the velocity .x field
        Scalar4 vel = d_vel[idx];
        vel.x = __int_as_scalar(seed_i_new);
        d_trial_vel[idx] = vel;
        }

    // stash the trial move in global memory
    d_trial_postype[idx] = make_scalar4(pos_i.x, pos_i.y, pos_i.z, __int_as_scalar(typ_i));
    d_trial_orientation[idx] = quat_to_scalar4(shape_i.orientation);

    // 0==inactive, 1==translation, 2==rotation
    d_trial_move_type[idx] = move_active ? (move_type_translate ? 1 : 2) : 0;

    // initialize reject flag
    d_reject_out_of_cell[idx] = reject;
    d_reject[idx] = 0;
    }

//! Kernel to update particle data and statistics after acceptance
template<class Shape>
__global__ void hpmc_update_pdata(Scalar4 *d_postype,
                                  Scalar4 *d_orientation,
                                  Scalar4 *d_vel,
                                  hpmc_counters_t *d_counters,
                                  const unsigned int nwork,
                                  const unsigned int offset,
                                  const bool have_auxilliary_variable,
                                  const Scalar4 *d_trial_postype,
                                  const Scalar4 *d_trial_orientation,
                                  const Scalar4 *d_trial_vel,
                                  const unsigned int *d_trial_move_type,
                                  const unsigned int *d_reject,
                                  const unsigned int *d_reject_out_of_cell,
                                  const typename Shape::param_type *d_params)
    {
    // determine which update step we are handling
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // shared arrays for per type pair parameters
    __shared__ unsigned int s_translate_accept_count;
    __shared__ unsigned int s_translate_reject_count;
    __shared__ unsigned int s_rotate_accept_count;
    __shared__ unsigned int s_rotate_reject_count;

    // initialize the shared memory array for communicating overlaps
    if (threadIdx.x == 0)
        {
        s_translate_accept_count = 0;
        s_translate_reject_count = 0;
        s_rotate_accept_count = 0;
        s_rotate_reject_count = 0;
        }

    __syncthreads();

    if (idx < nwork)
        {
        idx += offset;

        unsigned int move_type = d_trial_move_type[idx];
        bool move_active = move_type > 0;
        bool move_type_translate = move_type == 1;
        bool accept = !d_reject[idx] && !d_reject_out_of_cell[idx];

        unsigned int type_i = __scalar_as_int(d_postype[idx].w);
        Shape shape_i(quat<Scalar>(), d_params[type_i]);

        bool ignore_stats = shape_i.ignoreStatistics();

        // update the data if accepted
        if (move_active)
            {
            if (accept)
                {
                // write out the updated position and orientation
                d_postype[idx] = d_trial_postype[idx];
                d_orientation[idx] = d_trial_orientation[idx];

                if (have_auxilliary_variable)
                    d_vel[idx] = d_trial_vel[idx];
                }

            if (!ignore_stats && accept && move_type_translate)
                atomicAdd(&s_translate_accept_count, 1);
            if (!ignore_stats && accept && !move_type_translate)
                atomicAdd(&s_rotate_accept_count, 1);
            if (!ignore_stats && !accept && move_type_translate)
                atomicAdd(&s_translate_reject_count, 1);
            if (!ignore_stats && !accept && !move_type_translate)
                atomicAdd(&s_rotate_reject_count, 1);
            }
        }

    __syncthreads();

    // final tally into global mem
    if (threadIdx.x == 0)
        {
        atomicAdd(&d_counters->translate_accept_count, s_translate_accept_count);
        atomicAdd(&d_counters->translate_reject_count, s_translate_reject_count);
        atomicAdd(&d_counters->rotate_accept_count, s_rotate_accept_count);
        atomicAdd(&d_counters->rotate_reject_count, s_rotate_reject_count);
        }
    }
} // end namespace kernel

//! Kernel driver for kernel::hpmc_gen_moves
template< class Shape >
void hpmc_gen_moves(const hpmc_args_t& args, const typename Shape::param_type *params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_d);
    assert(args.d_a);

    if (args.dim == 2)
        {
        // determine the maximum block size and clamp the input block size down
        static int max_block_size = -1;
        static hipFuncAttributes attr;
        if (max_block_size == -1)
            {
            hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_gen_moves<Shape,2>));
            max_block_size = attr.maxThreadsPerBlock;
            }

        // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
        unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);
        unsigned int shared_bytes = args.num_types * (sizeof(typename Shape::param_type) + 2*sizeof(Scalar));

        if (shared_bytes + attr.sharedSizeBytes >= args.devprop.sharedMemPerBlock)
            throw std::runtime_error("hpmc::kernel::gen_moves() exceeds shared memory limits");

        // setup the grid to run the kernel
        dim3 threads( block_size, 1, 1);
        dim3 grid(args.N/block_size+1,1,1);

        hipLaunchKernelGGL((kernel::hpmc_gen_moves<Shape,2>), grid, threads, shared_bytes, 0,
                                                                     args.d_postype,
                                                                     args.d_orientation,
                                                                     args.d_vel,
                                                                     args.N,
                                                                     args.ci,
                                                                     args.cell_dim,
                                                                     args.ghost_width,
                                                                     args.num_types,
                                                                     args.seed,
                                                                     args.d_d,
                                                                     args.d_a,
                                                                     args.move_ratio,
                                                                     args.timestep,
                                                                     args.box,
                                                                     args.select,
                                                                     args.ghost_fraction,
                                                                     args.domain_decomposition,
                                                                     args.have_auxilliary_variable,
                                                                     args.d_trial_postype,
                                                                     args.d_trial_orientation,
                                                                     args.d_trial_vel,
                                                                     args.d_trial_move_type,
                                                                     args.d_reject_out_of_cell,
                                                                     args.d_reject_out,
                                                                     params
                                                                );
        }
    else
        {
        // determine the maximum block size and clamp the input block size down
        static int max_block_size = -1;
        static hipFuncAttributes attr;
        if (max_block_size == -1)
            {
            hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_gen_moves<Shape,3>));
            max_block_size = attr.maxThreadsPerBlock;
            }

        // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
        unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);
        unsigned int shared_bytes = args.num_types * (sizeof(typename Shape::param_type) + 2*sizeof(Scalar));

        if (shared_bytes + attr.sharedSizeBytes >= args.devprop.sharedMemPerBlock)
            throw std::runtime_error("hpmc::kernel::gen_moves() exceeds shared memory limits");

        // setup the grid to run the kernel
        dim3 threads( block_size, 1, 1);
        dim3 grid(args.N/block_size+1,1,1);

        hipLaunchKernelGGL((kernel::hpmc_gen_moves<Shape,3>), grid, threads, shared_bytes, 0,
                                                                     args.d_postype,
                                                                     args.d_orientation,
                                                                     args.d_vel,
                                                                     args.N,
                                                                     args.ci,
                                                                     args.cell_dim,
                                                                     args.ghost_width,
                                                                     args.num_types,
                                                                     args.seed,
                                                                     args.d_d,
                                                                     args.d_a,
                                                                     args.move_ratio,
                                                                     args.timestep,
                                                                     args.box,
                                                                     args.select,
                                                                     args.ghost_fraction,
                                                                     args.domain_decomposition,
                                                                     args.have_auxilliary_variable,
                                                                     args.d_trial_postype,
                                                                     args.d_trial_orientation,
                                                                     args.d_trial_vel,
                                                                     args.d_trial_move_type,
                                                                     args.d_reject_out_of_cell,
                                                                     args.d_reject_out,
                                                                     params
                                                                );
        }
    }

//! Driver for kernel::hpmc_update_pdata()
template<class Shape>
void hpmc_update_pdata(const hpmc_update_args_t& args, const typename Shape::param_type *params)
    {
    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    static hipFuncAttributes attr;
    if (max_block_size == -1)
        {
        hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_update_pdata<Shape>));
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);
    for (int idev = args.gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = args.gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;
        const unsigned int num_blocks = nwork/block_size + 1;

        hipLaunchKernelGGL((kernel::hpmc_update_pdata<Shape>), dim3(num_blocks), dim3(block_size), 0, 0,
            args.d_postype,
            args.d_orientation,
            args.d_vel,
            args.d_counters+idev*args.counters_pitch,
            nwork,
            range.first,
            args.have_auxilliary_variable,
            args.d_trial_postype,
            args.d_trial_orientation,
            args.d_trial_vel,
            args.d_trial_move_type,
            args.d_reject,
            args.d_reject_out_of_cell,
            params);
        }
    }
#endif

} // end namespace gpu

} // end namespace hpmc
