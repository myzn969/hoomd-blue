// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#include <hip/hip_runtime.h>
#include "hoomd/HOOMDMath.h"

#include "IntegratorHPMCMonoGPUDepletantsTypes.cuh"

namespace hpmc {

namespace gpu {

//! Wraps arguments to kernel::hpmc_insert_depletants_phase(n)
/*! \ingroup hpmc_data_structs */
struct hpmc_auxiliary_args_t
    {
    //! Construct a hpmc_auxiliary_args_t
    hpmc_auxiliary_args_t(const unsigned int *_d_tag,
                           const Scalar4 *_d_vel,
                           const Scalar4 *_d_trial_vel,
                           const float _gamma,
                           const unsigned int _nwork_local[],
                           const unsigned int _work_offset[],
                           const unsigned int *_d_n_depletants_ntrial,
                           const hipStream_t *_streams_phase1,
                           const hipStream_t *_streams_phase2,
                           const unsigned int _max_len,
                           unsigned int *_d_req_len,
                           const bool _add_ghosts,
                           const unsigned int _n_ghosts,
                           const GPUPartition& _gpu_partition_rank,
                           unsigned int *_d_deltaF_or_nneigh,
                           unsigned int *_d_deltaF_or_nlist,
                           unsigned int *_d_deltaF_or_len,
                           float *_d_deltaF_or_energy,
                           Scalar *_d_deltaF_or,
                           const unsigned int _deltaF_or_maxlen,
                           unsigned int *_d_overflow_or,
                           unsigned int *_d_deltaF_nor_nneigh,
                           unsigned int *_d_deltaF_nor_nlist,
                           unsigned int *_d_deltaF_nor_len,
                           unsigned int *_d_deltaF_nor_k,
                           float *_d_deltaF_nor_energy,
                           Scalar *_d_deltaF_nor,
                           const unsigned int _deltaF_nor_maxlen,
                           unsigned int *_d_overflow_nor,
                           const Scalar _r_cut_patch,
                           const Scalar *_d_additive_cutoff,
                           const unsigned int _eval_threads,
                           const Scalar *_d_charge,
                           const Scalar *_d_diameter)
                : d_tag(_d_tag),
                  d_vel(_d_vel),
                  d_trial_vel(_d_trial_vel),
                  gamma(_gamma),
                  nwork_local(_nwork_local),
                  work_offset(_work_offset),
                  d_n_depletants_ntrial(_d_n_depletants_ntrial),
                  streams_phase1(_streams_phase1),
                  streams_phase2(_streams_phase2),
                  max_len(_max_len),
                  d_req_len(_d_req_len),
                  add_ghosts(_add_ghosts),
                  n_ghosts(_n_ghosts),
                  gpu_partition_rank(_gpu_partition_rank),
                  d_deltaF_or_nneigh(_d_deltaF_or_nneigh),
                  d_deltaF_or_nlist(_d_deltaF_or_nlist),
                  d_deltaF_or_len(_d_deltaF_or_len),
                  d_deltaF_or_energy(_d_deltaF_or_energy),
                  d_deltaF_or(_d_deltaF_or),
                  deltaF_or_maxlen(_deltaF_or_maxlen),
                  d_overflow_or(_d_overflow_or),
                  d_deltaF_nor_nneigh(_d_deltaF_nor_nneigh),
                  d_deltaF_nor_nlist(_d_deltaF_nor_nlist),
                  d_deltaF_nor_len(_d_deltaF_nor_len),
                  d_deltaF_nor_k(_d_deltaF_nor_k),
                  d_deltaF_nor_energy(_d_deltaF_nor_energy),
                  d_deltaF_nor(_d_deltaF_nor),
                  deltaF_nor_maxlen(_deltaF_nor_maxlen),
                  d_overflow_nor(_d_overflow_nor),
                  r_cut_patch(_r_cut_patch),
                  d_additive_cutoff(_d_additive_cutoff),
                  eval_threads(_eval_threads),
                  d_charge(_d_charge),
                  d_diameter(_d_diameter)
        { };

    const unsigned int *d_tag;          //!< Particle tags
    const Scalar4 *d_vel;               //!< Particle velocities (.x component is the auxiliary variable)
    const Scalar4 *d_trial_vel;         //!< Particle velocities after trial move (.x component is the auxiliary variable)
    const float gamma;                  //!< Number of trial insertions per depletant
    const unsigned int *nwork_local;    //!< Number of insertions this rank handles, per GPU
    const unsigned int *work_offset;    //!< Offset into insertions for this rank
    const unsigned int *d_n_depletants_ntrial;     //!< Number of depletants per particle, depletant type pair and trial insertion
    const hipStream_t *streams_phase1;             //!< Stream for this depletant type, phase1 kernel
    const hipStream_t *streams_phase2;             //!< Stream for this depletant type, phase2 kernel
    const unsigned int max_len;         //!< Max length of dynamically allocated shared memory list
    unsigned int *d_req_len;            //!< Requested length of shared mem list per group
    const bool add_ghosts;              //!< True if we should add the ghosts from the domain decomposition
    const unsigned int n_ghosts;        //!< Number of ghost particles
    const GPUPartition& gpu_partition_rank; //!< Split of particles for this rank
    unsigned int *d_deltaF_or_nneigh; //!< Number of neighbors for logical or
    unsigned int *d_deltaF_or_nlist;  //!< Neighbor ids for logical or
    unsigned int *d_deltaF_or_len;    //!< Length of every logical or term
    float *d_deltaF_or_energy;        //!< Energy contribution to Mayer f function
    Scalar *d_deltaF_or;              //!< Free energy associated with logical or term
    const unsigned int deltaF_or_maxlen;    //!< Maximum number of neighbors for logical or
    unsigned int *d_overflow_or;            //!< Overflow flag for logical or
    unsigned int *d_deltaF_nor_nneigh;//!< Number of neighbors for logical nor
    unsigned int *d_deltaF_nor_nlist; //!< Neighbor ids for logical nor
    unsigned int *d_deltaF_nor_len;   //!< Length of every logical nor term
    unsigned int *d_deltaF_nor_k;     //!< Origin particle in phase2 kernel
    float *d_deltaF_nor_energy;       //!< Energy contributions to Mayer f function
    Scalar *d_deltaF_nor;             //!< Free energy associated with logical nor term
    const unsigned int deltaF_nor_maxlen;   //!< Maximum number of neighbors for logical nor
    unsigned int *d_overflow_nor;           //!< Overflow flag for logical nor
    const Scalar r_cut_patch;          //!< Cutoff radius
    const Scalar *d_additive_cutoff;   //!< Additive cutoff radii per type
    unsigned int eval_threads;         //!< Number of threads for energy evaluation
    const Scalar *d_charge;            //!< Particle charges
    const Scalar *d_diameter;          //!< Particle diameters
    };

//! Driver for kernel::hpmc_insert_depletants_auxiliary_phase2()
template< class Shape >
void hpmc_depletants_auxiliary_phase2(const hpmc_args_t& args,
                                       const hpmc_implicit_args_t& implicit_args,
                                       const hpmc_auxiliary_args_t& auxiliary_args,
                                       const typename Shape::param_type *params);

//! Driver for kernel::hpmc_insert_depletants_auxiliary_phase1()
template< class Shape >
void hpmc_depletants_auxiliary_phase1(const hpmc_args_t& args,
                                       const hpmc_implicit_args_t& implicit_args,
                                       const hpmc_auxiliary_args_t& auxiliary_args,
                                       const typename Shape::param_type *params);

void generate_num_depletants_ntrial(const Scalar4 *d_vel,
                                    const Scalar4 *d_trial_vel,
                                    const float gamma,
                                    const unsigned int depletant_type_a,
                                    const unsigned int depletant_type_b,
                                    const Index2D depletant_idx,
                                    const Scalar *d_lambda,
                                    const Scalar4 *d_postype,
                                    unsigned int *d_n_depletants,
                                    const unsigned int N_local,
                                    const bool add_ghosts,
                                    const unsigned int n_ghosts,
                                    const GPUPartition& gpu_partition,
                                    const unsigned int block_size,
                                    const hipStream_t *streams);

void get_max_num_depletants_ntrial(const float gamma,
                            unsigned int *d_n_depletants,
                            unsigned int *max_n_depletants,
                            const bool add_ghosts,
                            const unsigned int n_ghosts,
                            const hipStream_t *streams,
                            const GPUPartition& gpu_partition,
                            CachedAllocator& alloc);
} // end namespace gpu

} // end namespace hpmc
