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
#include "hoomd/WarpTools.cuh"

#include "hoomd/hpmc/HPMCCounters.h"

#include "GPUHelpers.cuh"
#include "HPMCMiscFunctions.h"

#include "IntegratorHPMCMonoGPUDepletantsAuxiliaryTypes.cuh"
#include "IntegratorHPMCMonoGPUDepletants.cuh"

#include "hoomd/jit/Evaluator.cuh"
#include "hoomd/jit/EvaluatorUnionGPU.cuh"

#ifdef __HIPCC__
namespace hpmc {

namespace gpu {

namespace kernel
{

//! Kernel for computing the depletion Metropolis-Hastings weight  (phase 1)
template< class Shape, unsigned int max_threads, bool patch = false, unsigned int eval_threads = 1, bool patch_union = false>
#ifdef __HIP_PLATFORM_NVCC__
__launch_bounds__(max_threads)
#endif
__global__ void hpmc_insert_depletants_phase1(const Scalar4 *d_trial_postype,
                                     const Scalar4 *d_trial_orientation,
                                     const unsigned int *d_trial_move_type,
                                     const Scalar4 *d_postype,
                                     const Scalar4 *d_orientation,
                                     hpmc_counters_t *d_counters,
                                     const unsigned int *d_excell_idx,
                                     const unsigned int *d_excell_size,
                                     const Index2D excli,
                                     const uint3 cell_dim,
                                     const Scalar3 ghost_width,
                                     const Index3D ci,
                                     const unsigned int N_local,
                                     const unsigned int num_types,
                                     const unsigned int seed,
                                     const unsigned int *d_check_overlaps,
                                     const Index2D overlap_idx,
                                     const unsigned int timestep,
                                     const unsigned int dim,
                                     const BoxDim box,
                                     const unsigned int select,
                                     const typename Shape::param_type *d_params,
                                     unsigned int max_queue_size,
                                     unsigned int max_extra_bytes,
                                     unsigned int depletant_type,
                                     const Index2D depletant_idx,
                                     hpmc_implicit_counters_t *d_implicit_counters,
                                     const unsigned int *d_update_order_by_ptl,
                                     const unsigned int ntrial,
                                     const unsigned int *d_tag,
                                     const Scalar4 *d_vel,
                                     const Scalar4 *d_trial_vel,
                                     unsigned int *d_deltaF_or_nneigh,
                                     unsigned int *d_deltaF_or_nlist,
                                     unsigned int *d_deltaF_or_len,
                                     float *d_deltaF_or_energy,
                                     Scalar *d_deltaF_or,
                                     const unsigned int max_len_deltaF,
                                     unsigned int *d_overflow,
                                     bool repulsive,
                                     unsigned int work_offset,
                                     unsigned int max_depletant_queue_size,
                                     const unsigned int *d_n_depletants,
                                     const unsigned int max_len,
                                     unsigned int *d_req_len,
                                     const unsigned int global_work_offset,
                                     const unsigned int n_work_local,
                                     const unsigned int max_n_depletants,
                                     const unsigned int *d_type_params,
                                     const Scalar r_cut_patch,
                                     const Scalar *d_additive_cutoff,
                                     const Scalar *d_charge,
                                     const Scalar *d_diameter
                                     )
    {
    // variables to tell what type of thread we are
    unsigned int group = threadIdx.y;
    unsigned int offset = threadIdx.z;
    unsigned int group_size = blockDim.z;
    bool master = (offset == 0) && (threadIdx.x==0);
    unsigned int n_groups = blockDim.y;

    unsigned int err_count = 0;

    // shared particle configuation
    __shared__ Scalar4 s_orientation_i_new;
    __shared__ Scalar4 s_orientation_i_old;
    __shared__ Scalar3 s_pos_i_new;
    __shared__ Scalar3 s_pos_i_old;
    __shared__ unsigned int s_type_i;
    __shared__ Scalar s_charge_i;
    __shared__ Scalar s_diameter_i;

    // shared arrays for per type pair parameters
    __shared__ unsigned int s_overlap_checks;
    __shared__ unsigned int s_overlap_err_count;

    // shared queue variables
    __shared__ unsigned int s_queue_size;
    __shared__ unsigned int s_still_searching;
    __shared__ unsigned int s_adding_depletants;
    __shared__ unsigned int s_depletant_queue_size;

    // load the per type pair parameters into shared memory
    HIP_DYNAMIC_SHARED( char, s_data)
    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    jit::union_params_t *s_union_params = (jit::union_params_t *)(s_params + num_types);

    Scalar4 *s_orientation_group = (patch_union ? (Scalar4*) (s_union_params + num_types) : (Scalar4*) (s_params + num_types));
    Scalar3 *s_pos_group = (Scalar3*)(s_orientation_group + n_groups);
    Scalar *s_additive_cutoff = (Scalar *) (s_pos_group + n_groups);
    unsigned int *s_check_overlaps = (patch ? (unsigned int *) (s_additive_cutoff + num_types) : (unsigned int *) (s_pos_group + n_groups));
    unsigned int *s_len_group = (unsigned int *) (s_check_overlaps + overlap_idx.getNumElements());
    unsigned int *s_overlap_idx_list = (unsigned int *) (s_len_group + n_groups);
    float *s_energy_list = (float *) (s_overlap_idx_list+ max_len*n_groups);
    unsigned int *s_queue_j = (unsigned int*)(s_energy_list + max_len*n_groups);
    unsigned int *s_queue_gid = (unsigned int*)(s_queue_j + max_queue_size);
    unsigned int *s_queue_didx = (unsigned int *)(s_queue_gid + max_queue_size);
    unsigned int *s_queue_itrial = (unsigned int *)(s_queue_didx + max_depletant_queue_size);
    float *s_queue_f = (float *)(s_queue_itrial + max_depletant_queue_size);

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

        unsigned int ntyppairs = overlap_idx.getNumElements();

        for (unsigned int cur_offset = 0; cur_offset < ntyppairs; cur_offset += block_size)
            {
            if (cur_offset + tidx < ntyppairs)
                {
                s_check_overlaps[cur_offset + tidx] = d_check_overlaps[cur_offset + tidx];
                }
            }

        if (patch_union)
            {
            unsigned int union_param_size = num_types*sizeof(jit::union_params_t) / sizeof(int);
            for (unsigned int cur_offset = 0; cur_offset < union_param_size; cur_offset += block_size)
                {
                if (cur_offset + tidx < union_param_size)
                    {
                    ((int *)s_union_params)[cur_offset + tidx] = ((int *)jit::d_union_params)[cur_offset + tidx];
                    }
                }
            }

        if (patch)
            {
            for (unsigned int cur_offset = 0; cur_offset < num_types; cur_offset += block_size)
                {
                if (cur_offset + tidx < num_types)
                    {
                    s_additive_cutoff[cur_offset + tidx] = d_additive_cutoff[cur_offset + tidx];
                    }
                }
            }
        }

    __syncthreads();

    // initialize extra shared mem
    char *s_extra = (char *)(s_queue_f + max_depletant_queue_size);

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, available_bytes, d_type_params[cur_type]);

    if (patch_union)
        {
        for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
            s_union_params[cur_type].load_shared(s_extra, available_bytes, d_type_params[cur_type] >> Shape::getTuningBits());
        }

    // initialize the shared memory array for communicating overlaps
    if (master && group == 0)
        {
        s_overlap_checks = 0;
        s_overlap_err_count = 0;
        }

    __syncthreads();

    // identify the particle that this block handles
    unsigned int i = blockIdx.x + work_offset;

    // load updated particle position
    if (master && group == 0)
        {
        Scalar4 postype_i_new = d_trial_postype[i];
        Scalar4 postype_i_old = d_postype[i];
        s_pos_i_new = make_scalar3(postype_i_new.x, postype_i_new.y, postype_i_new.z);
        s_pos_i_old = make_scalar3(postype_i_old.x, postype_i_old.y, postype_i_old.z);
        s_type_i = __scalar_as_int(postype_i_new.w);
        s_diameter_i = d_diameter[i];
        s_charge_i = d_charge[i];

        s_orientation_i_new = d_trial_orientation[i];
        s_orientation_i_old = d_orientation[i];
        }

    // sync so that s_pos_i_old etc. are available
    __syncthreads();

    unsigned int overlap_checks = 0;

    // find the cell this particle should be in
    unsigned int my_cell = computeParticleCell(s_pos_i_old, box, ghost_width,
        cell_dim, ci, false);

    // the order of this particle in the chain
    unsigned int update_order_i = d_update_order_by_ptl[i];
    unsigned int seed_i_old = __scalar_as_int(d_vel[i].x);
    unsigned int seed_i_new = __scalar_as_int(d_trial_vel[i].x);
    unsigned int tag_i = d_tag[i];

    if (master && group == 0)
        {
        s_depletant_queue_size = 0;
        s_adding_depletants = 1;
        }

    __syncthreads();

    // unpack the block index to find out which work element this group handles
    unsigned int block = blockIdx.z*gridDim.y + blockIdx.y;
    unsigned int grid = gridDim.y*gridDim.z;

    unsigned int local_work_idx = offset + group*group_size + n_groups*group_size*(block >> 1);
    unsigned int blocks_per_depletant = grid>>1;

    // we always have a number of blocks that is a multiple of two
    bool new_config = block & 1;

    detail::OBB obb_i;
        {
        // get shape OBB
        Shape shape_i(quat<Scalar>(!new_config ? d_orientation[i] : d_trial_orientation[i]), s_params[s_type_i]);
        obb_i = shape_i.getOBB(new_config ? vec3<Scalar>(s_pos_i_new) : vec3<Scalar>(s_pos_i_old));

        // extend by depletant radius
        Shape shape_test(quat<Scalar>(), s_params[depletant_type]);
        OverlapReal r = 0.5*shape_test.getCircumsphereDiameter();
        obb_i.lengths.x += r;
        obb_i.lengths.y += r;
        obb_i.lengths.z += r;

        if (patch)
            {
            Scalar delta = 0.5*s_additive_cutoff[s_type_i];
            delta += 0.5*s_additive_cutoff[depletant_type];

            Scalar min_obb = detail::min(obb_i.lengths.x,detail::min(obb_i.lengths.y, obb_i.lengths.z));
            Scalar max_obb = detail::max(obb_i.lengths.x,detail::max(obb_i.lengths.y, obb_i.lengths.z));

            if (r_cut_patch + delta > min_obb)
                {
                obb_i = detail::OBB(new_config ? vec3<Scalar>(s_pos_i_new) : vec3<Scalar>(s_pos_i_old),
                    detail::max(r_cut_patch + delta, max_obb));
                }
            }
        }

    /*
     * Phase 1: insert into particle i's excluded volume
     */

    unsigned int n_depletants = 0;

    while (s_adding_depletants)
        {
        // compute global indices
        unsigned int global_work_idx = global_work_offset + local_work_idx;
        unsigned int i_dep = global_work_idx % max_n_depletants;
        unsigned int i_trial = global_work_idx/max_n_depletants;

        // load random number of depletants from Poisson distribution
        unsigned int n_depletants_i = d_n_depletants[i*2*ntrial+new_config*ntrial+i_trial];

        if (i_dep == 0) n_depletants += n_depletants_i;

        while (s_depletant_queue_size < max_depletant_queue_size && local_work_idx < n_work_local)
            {
            if (i_trial < ntrial && i_dep < n_depletants_i)
                {
                // one RNG per depletant and trial insertion
                hoomd::RandomGenerator rng(hoomd::RNGIdentifier::HPMCDepletants, new_config ? seed_i_new : seed_i_old,
                    i_dep, i_trial, depletant_type);

                // filter depletants overlapping with particle i
                vec3<Scalar> pos_test = vec3<Scalar>(generatePositionInOBB(rng, obb_i, dim));

                Shape shape_test(quat<Scalar>(), s_params[depletant_type]);
                quat<Scalar> o;
                if (shape_test.hasOrientation())
                    {
                    o = generateRandomOrientation(rng, dim);
                    shape_test.orientation = o;
                    }

                Shape shape_i(quat<Scalar>(), s_params[s_type_i]);
                if (shape_i.hasOrientation())
                    shape_i.orientation = quat<Scalar>(new_config ? s_orientation_i_new : s_orientation_i_old);
                vec3<Scalar> r_ij = vec3<Scalar>(new_config ? s_pos_i_new : s_pos_i_old) - pos_test;
                overlap_checks ++;
                bool overlap_i = (s_check_overlaps[overlap_idx(s_type_i, depletant_type)]
                    && check_circumsphere_overlap(r_ij, shape_test, shape_i)
                    && test_overlap(r_ij, shape_test, shape_i, err_count));

                overlap_i = hoomd::detail::WarpReduce<bool, eval_threads>().Sum(overlap_i);

                // Mayer f-function
                Scalar f_i = 0.0;

                if (patch)
                    {
                    // compute interaction with particle i
                    OverlapReal r_cut_patch_i = r_cut_patch + 0.5*s_additive_cutoff[s_type_i];
                    r_cut_patch_i += 0.5*s_additive_cutoff[depletant_type];

                    OverlapReal rsq = dot(r_ij,r_ij);

                    float energy = 0.0f;
                    if (rsq <= r_cut_patch_i*r_cut_patch_i)
                        {
                        // evaluate energy
                        if (patch_union)
                            {
                            energy = jit::eval_union(s_union_params,
                                r_ij,
                                depletant_type,
                                quat<float>(shape_test.orientation),
                                0.0,
                                0.0,
                                s_type_i,
                                quat<float>(shape_i.orientation),
                                s_diameter_i,
                                s_charge_i);

                            // sum up energy from logical warp
                            energy = hoomd::detail::WarpReduce<float, eval_threads>().Sum(energy);
                            }
                        else
                            {
                            if (threadIdx.x == 0)
                                {
                                // no parallelism in user function
                                energy = eval(
                                    r_ij,
                                    depletant_type,
                                    quat<float>(shape_test.orientation),
                                    0.0,
                                    0.0,
                                    s_type_i,
                                    quat<float>(shape_i.orientation),
                                    s_diameter_i,
                                    s_charge_i);
                                }
                            }

                        energy = hoomd::detail::WarpScan<float, eval_threads>().Broadcast(energy, 0);
                        }

                    f_i = 1.0f-fast::exp(-energy);
                    }

                f_i = overlap_i + (1-overlap_i)*f_i;

                if (f_i != 0.0)
                    {
                    // add this particle to the queue
                    unsigned int insert_point;
                    if (threadIdx.x == 0)
                        {
                        insert_point = atomicAdd(&s_depletant_queue_size, 1);
                        }
                    insert_point = hoomd::detail::WarpScan<unsigned int, eval_threads>().Broadcast(insert_point, 0);

                    if (insert_point < max_depletant_queue_size)
                        {
                        if (threadIdx.x == 0)
                            {
                            s_queue_didx[insert_point] = i_dep;
                            s_queue_itrial[insert_point] = i_trial;
                            s_queue_f[insert_point] = f_i;
                            }
                        }
                    else
                        {
                        // we will recheck and insert this on the next time through
                        break;
                        }
                    } // end if add_to_queue
                }

            // advance local depletant idx
            local_work_idx += group_size*n_groups*blocks_per_depletant;

            // compute new global index
            global_work_idx = global_work_offset + local_work_idx;
            i_dep = global_work_idx % max_n_depletants;
            i_trial = global_work_idx/max_n_depletants;
            n_depletants_i = d_n_depletants[i*2*ntrial+new_config*ntrial+i_trial];
            } // end while (s_depletant_queue_size < max_depletant_queue_size && i_dep < n_depletants_i)

        __syncthreads();

        // process the queue, group by group
        if (master && group == 0)
            s_adding_depletants = 0;

        if (master && group == 0)
            {
            // reset the queue for neighbor checks
            s_queue_size = 0;
            s_still_searching = 1;
            }

        __syncthreads();

        // is this group processing work from the first queue?
        bool active = group < min(s_depletant_queue_size, max_depletant_queue_size);

        if (active)
            {
            // regenerate depletant using seed from queue, this costs a few flops but is probably
            // better than storing one Scalar4 and a Scalar3 per thread in shared mem
            unsigned int i_dep_queue = s_queue_didx[group];
            unsigned int i_trial_queue = s_queue_itrial[group];

            hoomd::RandomGenerator rng(hoomd::RNGIdentifier::HPMCDepletants,
                new_config ? seed_i_new : seed_i_old,
                i_dep_queue, i_trial_queue,
                depletant_type);

            // depletant position and orientation
            vec3<Scalar> pos_test = vec3<Scalar>(generatePositionInOBB(rng, obb_i, dim));
            Shape shape_test(quat<Scalar>(), s_params[depletant_type]);
            quat<Scalar> o;
            if (shape_test.hasOrientation())
                {
                o = generateRandomOrientation(rng,dim);
                }

            // store them per group
            if (master)
                {
                s_pos_group[group] = vec_to_scalar3(pos_test);
                s_orientation_group[group] = quat_to_scalar4(o);
                }
            }

        if (master)
            {
            // stores overlaps
            s_len_group[group] = 0;
            }

        __syncthreads();

        // counters to track progress through the loop over potential neighbors
        unsigned int excell_size;
        unsigned int k = offset;

        if (active)
            {
            excell_size = d_excell_size[my_cell];

            if (master)
                overlap_checks += 2*excell_size;
            }

        // loop while still searching
        while (s_still_searching)
            {
            // fill the neighbor queue
            // loop through particles in the excell list and add them to the queue if they pass the circumsphere check

            // active threads add to the queue
            if (threadIdx.x == 0 && active)
                {
                // prefetch j
                unsigned int j, next_j = 0;
                if ((k >> 1) < excell_size)
                    next_j = __ldg(&d_excell_idx[excli(k >> 1, my_cell)]);

                // add to the queue as long as the queue is not full, and we have not yet reached the end of our own list
                // and as long as no overlaps have been found
                while (s_queue_size < max_queue_size && (k >> 1) < excell_size)
                    {
                    // which configuration of particle j are we checking against?
                    bool old = k & 1;

                    // prefetch next j
                    k += group_size;
                    j = next_j;

                    if ((k >> 1) < excell_size)
                        next_j = __ldg(&d_excell_idx[excli(k >> 1, my_cell)]);

                    unsigned int tag_j = d_tag[j];

                    // if j was never updated before i, do not check new config
                    if (!old && (j >= N_local || d_update_order_by_ptl[j] > update_order_i))
                        continue;

                    // read in position of neighboring particle, do not need it's orientation for circumsphere check
                    // for ghosts always load particle data
                    Scalar4 postype_j = (old || j >= N_local) ? d_postype[j] : d_trial_postype[j];
                    unsigned int type_j = __scalar_as_int(postype_j.w);
                    Shape shape_j(quat<Scalar>(), s_params[type_j]);

                    // load test particle configuration from shared mem
                    vec3<Scalar> pos_test(s_pos_group[group]);
                    Shape shape_test(quat<Scalar>(s_orientation_group[group]), s_params[depletant_type]);

                    // put particle j into the coordinate system of the test particle
                    vec3<Scalar> r_jk = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test);
                    r_jk = vec3<Scalar>(box.minImage(vec_to_scalar3(r_jk)));

                    bool insert_in_queue = i != j && (old || j < N_local) && tag_i < tag_j;

                    bool circumsphere_overlap = s_check_overlaps[overlap_idx(depletant_type, type_j)] &&
                        check_circumsphere_overlap(r_jk, shape_test, shape_j);

                    if (patch)
                        {
                        OverlapReal delta = 0.5*s_additive_cutoff[type_j];
                        delta += 0.5*s_additive_cutoff[depletant_type];
                        OverlapReal rsq = dot(r_jk,r_jk);
                        circumsphere_overlap |= rsq <= (r_cut_patch+delta)*(r_cut_patch+delta);
                        }

                    insert_in_queue &= circumsphere_overlap;

                    if (insert_in_queue)
                        {
                        // add this particle to the queue
                        unsigned int insert_point = atomicAdd(&s_queue_size, 1);

                        if (insert_point < max_queue_size)
                            {
                            s_queue_gid[insert_point] = group;
                            s_queue_j[insert_point] = (j << 1) | (old ? 1 : 0);
                            }
                        else
                            {
                            // or back up if the queue is already full
                            // we will recheck and insert this on the next time through
                            k -= group_size;
                            }
                        } // end if (k>>1) < excell_size
                    } // end while (s_queue_size < max_queue_size && (k>>1) < excell_size)
                } // end if active

            // sync to make sure all threads in the block are caught up
            __syncthreads();

            // when we get here, all threads have either finished their list, or encountered a full queue
            // either way, it is time to process overlaps
            // need to clear the still searching flag and sync first
            if (master && group == 0)
                s_still_searching = 0;

            unsigned int tidx_1d = offset + group_size*group;

            // max_queue_size is always <= block size, so we just need an if here
            if (tidx_1d < min(s_queue_size, max_queue_size))
                {
                // need to extract the overlap check to perform out of the shared mem queue
                unsigned int check_group = s_queue_gid[tidx_1d];
                unsigned int check_j_flag = s_queue_j[tidx_1d];
                bool check_old = check_j_flag & 1;
                unsigned int check_j  = check_j_flag >> 1;

                // build depletant shape from shared memory
                Scalar3 pos_test = s_pos_group[check_group];
                Shape shape_test(quat<Scalar>(s_orientation_group[check_group]), s_params[depletant_type]);

                // build shape j from global memory
                Scalar4 postype_j = check_old ? d_postype[check_j] : d_trial_postype[check_j];
                Scalar4 orientation_j = make_scalar4(1,0,0,0);
                unsigned int type_j = __scalar_as_int(postype_j.w);
                Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);
                if (shape_j.hasOrientation())
                    shape_j.orientation = quat<Scalar>(check_old ? d_orientation[check_j] : d_trial_orientation[check_j]);

                // put particle j into the coordinate system of the test particle
                vec3<Scalar> r_jk = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test);
                r_jk = vec3<Scalar>(box.minImage(vec_to_scalar3(r_jk)));

                bool circumsphere_overlap = s_check_overlaps[overlap_idx(depletant_type, type_j)] &&
                    check_circumsphere_overlap(r_jk, shape_test, shape_j);
                bool overlap_j = circumsphere_overlap &&
                    test_overlap(r_jk, shape_test, shape_j, err_count);

                overlap_j = hoomd::detail::WarpReduce<bool, eval_threads>().Sum(overlap_j);

                bool within_rcut = false;
                if (patch)
                    {
                    OverlapReal delta = 0.5*s_additive_cutoff[type_j];
                    delta += 0.5*s_additive_cutoff[depletant_type];
                    OverlapReal rsq = dot(r_jk,r_jk);
                    within_rcut = rsq <= (r_cut_patch+delta)*(r_cut_patch+delta);
                    }

                float energy = 0.0f;
                if (within_rcut)
                    {
                    // evaluate energy
                    if (patch_union)
                        {
                        energy = jit::eval_union(s_union_params,
                            r_jk,
                            depletant_type,
                            quat<float>(shape_test.orientation),
                            0.0,
                            0.0,
                            type_j,
                            quat<float>(shape_j.orientation),
                            d_diameter[check_j],
                            d_charge[check_j]);

                        // sum up energy from logical warp
                        energy = hoomd::detail::WarpReduce<float, eval_threads>().Sum(energy);
                        }
                    else
                        {
                        if (threadIdx.x == 0)
                            {
                            // no parallelism in user function
                            energy = eval(
                                r_jk,
                                depletant_type,
                                quat<float>(shape_test.orientation),
                                0.0,
                                0.0,
                                type_j,
                                quat<float>(shape_j.orientation),
                                d_diameter[check_j],
                                d_charge[check_j]);
                            }
                        }

                    energy = hoomd::detail::WarpScan<float, eval_threads>().Broadcast(energy, 0);
                    }

                if (threadIdx.x == 0 && (overlap_j || within_rcut))
                    {
                    // store the overlapping particle index in dynamically allocated shared memory
                    unsigned int position = atomicAdd(&s_len_group[check_group], 1);
                    if (position < max_len)
                        {
                        s_overlap_idx_list[check_group*max_len+position] = (check_j_flag << 1) | overlap_j;
                        s_energy_list[check_group*max_len+position] = energy;
                        }
                    }
                } // end if (processing neighbor)

            // threads that need to do more looking set the still_searching flag
            __syncthreads();
            if (master && group == 0)
                s_queue_size = 0;
            if (threadIdx.x == 0 && active && (k>>1) < excell_size)
                atomicAdd(&s_still_searching, 1);
            __syncthreads();

            } // end while (s_still_searching)

        if (active && master && s_len_group[group] && s_len_group[group] <= max_len)
            {
            #if (__CUDA_ARCH__ >= 600)
            unsigned int n = atomicAdd_system(&d_deltaF_or_nneigh[i], s_len_group[group]);
            #else
            unsigned int n = atomicAdd(&d_deltaF_or_nneigh[i], s_len_group[group]);
            #endif
            if (n < max_len_deltaF)
                {
                d_deltaF_or_len[i*max_len_deltaF+n] = s_len_group[group];

                float f_i;
                if ((new_config && !repulsive) || (!new_config && repulsive))
                    f_i = s_queue_f[group];
                else
                    f_i = -s_queue_f[group];

                d_deltaF_or[i*max_len_deltaF+n] = f_i/(Scalar)ntrial;
                }
            for (unsigned int k = 0; k < s_len_group[group]; k++)
                {
                if (n + k < max_len_deltaF)
                    {
                    d_deltaF_or_nlist[i*max_len_deltaF + n + k] = s_overlap_idx_list[group*max_len + k];
                    d_deltaF_or_energy[i*max_len_deltaF + n + k] = s_energy_list[group*max_len + k];
                    }
                }
            }

        // shared mem overflowed?
        if (active && master && s_len_group[group] > max_len)
            {
            #if (__CUDA_ARCH__ >= 600)
            atomicMax_system(d_req_len, s_len_group[group]);
            #else
            atomicMax(d_req_len, s_len_group[group]);
            #endif
            }

        // do we still need to process depletants?
        __syncthreads();
        if (master && group == 0)
            s_depletant_queue_size = 0;
        if (threadIdx.x == 0 && local_work_idx < n_work_local)
            atomicAdd(&s_adding_depletants, 1);
        __syncthreads();
        } // end loop over depletants

    if (master && group == 0)
        {
        // overflowed?
        unsigned int nneigh = d_deltaF_or_nneigh[i];
        if (nneigh > max_len_deltaF)
            {
            #if (__CUDA_ARCH__ >= 600)
            atomicMax_system(d_overflow, nneigh);
            #else
            atomicMax(d_overflow, nneigh);
            #endif
            }
        }

    if (err_count > 0)
        atomicAdd(&s_overlap_err_count, err_count);

    // count the overlap checks
    atomicAdd(&s_overlap_checks, overlap_checks);

    __syncthreads();

    // final tally into global mem
    if (master && group == 0)
        {
        #if (__CUDA_ARCH__ >= 600)
        atomicAdd_system(&d_counters->overlap_err_count, s_overlap_err_count);
        atomicAdd_system(&d_counters->overlap_checks, s_overlap_checks);
        #else
        atomicAdd(&d_counters->overlap_err_count, s_overlap_err_count);
        atomicAdd(&d_counters->overlap_checks, s_overlap_checks);
        #endif
        }

    if (n_depletants)
        {
        Shape shape_i(quat<Scalar>(quat<Scalar>()), s_params[s_type_i]);
        bool ignore_stats = shape_i.ignoreStatistics();
        if (!ignore_stats)
            {
            // increment number of inserted depletants
            #if (__CUDA_ARCH__ >= 600)
            atomicAdd_system(&d_implicit_counters[depletant_idx(depletant_type,
                depletant_type)].insert_count, n_depletants);
            #else
            atomicAdd(&d_implicit_counters[depletant_idx(depletant_type,
                depletant_type)].insert_count, n_depletants);
            #endif
            }
        }
    }

} // end namespace kernel

} // end namespace gpu

} // end namespace hpmc
#endif // __HIPCC__
