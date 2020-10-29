// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#include <hip/hip_runtime.h>
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/BoxDim.h"
#include "hoomd/hpmc/HPMCCounters.h"
#include "hoomd/GPUPartition.cuh"

namespace hpmc {

namespace gpu {

//! Wraps arguments to hpmc_* template functions
/*! \ingroup hpmc_data_structs */
struct hpmc_args_t
    {
    //! Construct a hpmc_args_t
    hpmc_args_t(Scalar4 *_d_postype,
                Scalar4 *_d_orientation,
                Scalar4 *_d_vel,
                hpmc_counters_t *_d_counters,
                const unsigned int _counters_pitch,
                const Index3D& _ci,
                const uint3& _cell_dim,
                const Scalar3& _ghost_width,
                const unsigned int _N,
                const unsigned int _N_ghost,
                const unsigned int _num_types,
                const unsigned int _seed,
                const Scalar* _d,
                const Scalar* _a,
                const unsigned int *_check_overlaps,
                const Index2D& _overlap_idx,
                const unsigned int _move_ratio,
                const unsigned int _timestep,
                const unsigned int _dim,
                const BoxDim& _box,
                const unsigned int _select,
                const Scalar3 _ghost_fraction,
                const bool _domain_decomposition,
                const unsigned int _block_size,
                const unsigned int _tpp,
                const unsigned int _overlap_threads,
                const bool _have_auxiliary_variables,
                unsigned int *_d_reject_out_of_cell,
                Scalar4 *_d_trial_postype,
                Scalar4 *_d_trial_orientation,
                Scalar4 *_d_trial_vel,
                unsigned int *_d_trial_move_type,
                const unsigned int *_d_update_order_by_ptl,
                unsigned int *_d_excell_idx,
                const unsigned int *_d_excell_size,
                const Index2D& _excli,
                unsigned int *_d_nlist,
                unsigned int *_d_nneigh,
                const unsigned int _maxn,
                unsigned int *_d_overflow,
                const hipDeviceProp_t &_devprop,
                const GPUPartition& _gpu_partition,
                const hipStream_t *_streams,
                unsigned int *_d_type_params)
                : d_postype(_d_postype),
                  d_orientation(_d_orientation),
                  d_vel(_d_vel),
                  d_counters(_d_counters),
                  counters_pitch(_counters_pitch),
                  ci(_ci),
                  cell_dim(_cell_dim),
                  ghost_width(_ghost_width),
                  N(_N),
                  N_ghost(_N_ghost),
                  num_types(_num_types),
                  seed(_seed),
                  d_d(_d),
                  d_a(_a),
                  d_check_overlaps(_check_overlaps),
                  overlap_idx(_overlap_idx),
                  move_ratio(_move_ratio),
                  timestep(_timestep),
                  dim(_dim),
                  box(_box),
                  select(_select),
                  ghost_fraction(_ghost_fraction),
                  domain_decomposition(_domain_decomposition),
                  block_size(_block_size),
                  tpp(_tpp),
                  overlap_threads(_overlap_threads),
                  have_auxiliary_variables(_have_auxiliary_variables),
                  d_reject_out_of_cell(_d_reject_out_of_cell),
                  d_trial_postype(_d_trial_postype),
                  d_trial_orientation(_d_trial_orientation),
                  d_trial_vel(_d_trial_vel),
                  d_trial_move_type(_d_trial_move_type),
                  d_update_order_by_ptl(_d_update_order_by_ptl),
                  d_excell_idx(_d_excell_idx),
                  d_excell_size(_d_excell_size),
                  excli(_excli),
                  d_nlist(_d_nlist),
                  d_nneigh(_d_nneigh),
                  maxn(_maxn),
                  d_overflow(_d_overflow),
                  devprop(_devprop),
                  gpu_partition(_gpu_partition),
                  streams(_streams),
                  d_type_params(_d_type_params)
        {
        };

    Scalar4 *d_postype;               //!< postype array
    Scalar4 *d_orientation;           //!< orientation array
    Scalar4 *d_vel;                   //!< Velocities (.w component is auxillary variable)
    hpmc_counters_t *d_counters;      //!< Move accept/reject counters
    const unsigned int counters_pitch;         //!< Pitch of 2D array counters per GPU
    const Index3D& ci;                //!< Cell indexer
    const uint3& cell_dim;            //!< Cell dimensions
    const Scalar3& ghost_width;       //!< Width of the ghost layer
    const unsigned int N;             //!< Number of particles
    const unsigned int N_ghost;       //!< Number of ghost particles
    const unsigned int num_types;     //!< Number of particle types
    const unsigned int seed;          //!< RNG seed
    const Scalar* d_d;                //!< Maximum move displacement
    const Scalar* d_a;                //!< Maximum move angular displacement
    const unsigned int *d_check_overlaps; //!< Interaction matrix
    const Index2D& overlap_idx;       //!< Indexer into interaction matrix
    const unsigned int move_ratio;    //!< Ratio of translation to rotation moves
    const unsigned int timestep;      //!< Current time step
    const unsigned int dim;           //!< Number of dimensions
    const BoxDim& box;                //!< Current simulation box
    unsigned int select;              //!< Current selection
    const Scalar3 ghost_fraction;     //!< Width of the inactive layer
    const bool domain_decomposition;  //!< Is domain decomposition mode enabled?
    unsigned int block_size;          //!< Block size to execute
    unsigned int tpp;                 //!< Threads per particle
    unsigned int overlap_threads;     //!< Number of parallel threads per overlap check
    const bool have_auxiliary_variables; //!< True if we have auxiliary variables for depletants
    unsigned int *d_reject_out_of_cell;//!< Set to one to reject particle move
    Scalar4 *d_trial_postype;         //!< New positions (and type) of particles
    Scalar4 *d_trial_orientation;     //!< New orientations of particles
    Scalar4 *d_trial_vel;             //!< New auxiliary variables (velocites, w component)
    unsigned int *d_trial_move_type;  //!< per particle flag, whether it is a translation (1) or rotation (2), or inactive (0)
    const unsigned int *d_update_order_by_ptl;  //!< Lookup of update order by particle index
    unsigned int *d_excell_idx;       //!< Expanded cell list
    const unsigned int *d_excell_size;//!< Size of expanded cells
    const Index2D& excli;             //!< Excell indexer
    unsigned int *d_nlist;        //!< Neighbor list of overlapping particles after trial move
    unsigned int *d_nneigh;       //!< Number of overlapping particles after trial move
    unsigned int maxn;                //!< Width of neighbor list
    unsigned int *d_overflow;         //!< Overflow condition for neighbor list
    const hipDeviceProp_t& devprop;     //!< CUDA device properties
    const GPUPartition& gpu_partition; //!< Multi-GPU partition
    const hipStream_t *streams;        //!< kernel streams
    unsigned int *d_type_params;       //!< Per-type tuning parameters
    };

//! Wraps arguments for hpmc_update_pdata
struct hpmc_update_args_t
    {
    //! Construct an hpmc_update_args_t
    hpmc_update_args_t(Scalar4 *_d_postype,
        Scalar4 *_d_orientation,
        Scalar4 *_d_vel,
        hpmc_counters_t *_d_counters,
        unsigned int _counters_pitch,
        const GPUPartition& _gpu_partition,
        const bool _have_auxiliary_variable,
        const Scalar4 *_d_trial_postype,
        const Scalar4 *_d_trial_orientation,
        const Scalar4 *_d_trial_vel,
        const unsigned int *_d_trial_move_type,
        const unsigned int *_d_reject,
        const unsigned int _block_size)
        : d_postype(_d_postype),
          d_orientation(_d_orientation),
          d_vel(_d_vel),
          d_counters(_d_counters),
          counters_pitch(_counters_pitch),
          gpu_partition(_gpu_partition),
          have_auxiliary_variable(_have_auxiliary_variable),
          d_trial_postype(_d_trial_postype),
          d_trial_orientation(_d_trial_orientation),
          d_trial_vel(_d_trial_vel),
          d_trial_move_type(_d_trial_move_type),
          d_reject(_d_reject),
          block_size(_block_size)
     {}

    //! See hpmc_args_t for documentation on the meaning of these parameters
    Scalar4 *d_postype;
    Scalar4 *d_orientation;
    Scalar4 *d_vel;
    hpmc_counters_t *d_counters;
    unsigned int counters_pitch;
    const GPUPartition& gpu_partition;
    const bool have_auxiliary_variable;
    const Scalar4 *d_trial_postype;
    const Scalar4 *d_trial_orientation;
    const Scalar4 *d_trial_vel;
    const unsigned int *d_trial_move_type;
    const unsigned int *d_reject;
    const unsigned int block_size;
    };

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
                const hipStream_t *_streams,
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
                  streams(_streams),
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
    const hipStream_t *streams;      //!< Kernel streams
    const unsigned int *d_tuner_params; //!< Tuner parameters on device
    };

//! Driver for kernel::hpmc_narrow_phase()
template< class Shape >
void hpmc_narrow_phase(const hpmc_args_t& args, const typename Shape::param_type *params);

//! Driver for kernel::hpmc_gen_moves()
template< class Shape >
void hpmc_gen_moves(const hpmc_args_t& args, const typename Shape::param_type *params);

//! Driver for kernel::hpmc_update_pdata()
template< class Shape >
void hpmc_update_pdata(const hpmc_update_args_t& args, const typename Shape::param_type *params);

//! Driver for kernel::hpmc_excell()
void hpmc_excell(unsigned int *d_excell_idx,
                 unsigned int *d_excell_size,
                 const Index2D& excli,
                 const unsigned int *d_cell_idx,
                 const unsigned int *d_cell_size,
                 const unsigned int *d_cell_adj,
                 const Index3D& ci,
                 const Index2D& cli,
                 const Index2D& cadji,
                 const unsigned int ngpu,
                 const unsigned int block_size);

//! Kernel driver for kernel::hpmc_shift()
void hpmc_shift(Scalar4 *d_postype,
                int3 *d_image,
                const unsigned int N,
                const BoxDim& box,
                const Scalar3 shift,
                const unsigned int block_size);

void hpmc_sum_energies(const unsigned int *d_update_order_by_ptl,
                 const unsigned int *d_trial_move_type,
                 const unsigned int *d_reject_out_of_cell,
                 unsigned int *d_reject,
                 unsigned int *d_reject_out,
                 const unsigned int *d_nneigh,
                 const unsigned int *d_nlist,
                 const unsigned int N_old,
                 const unsigned int N,
                 const GPUPartition& gpu_partition,
                 const unsigned int maxn,
                 bool patch,
                 const unsigned int *d_nlist_patch_old,
                 const unsigned int *d_nlist_patch_new,
                 const unsigned int *d_nneigh_patch_old,
                 const unsigned int *d_nneigh_patch_new,
                 const float *d_energy_old,
                 const float *d_energy_new,
                 const unsigned int maxn_patch,
                 const unsigned int *d_deltaF_or_nneigh,
                 const unsigned int *d_deltaF_or_len,
                 const unsigned int *d_deltaF_or_nlist,
                 const float *d_deltaF_or_energy,
                 const Scalar *d_deltaF_or,
                 const unsigned maxn_deltaF_or,
                 const unsigned int *d_deltaF_nor_nneigh,
                 const unsigned int *d_deltaF_nor_len,
                 const unsigned int *d_deltaF_nor_k,
                 const unsigned int *d_deltaF_nor_nlist,
                 const float *d_deltaF_nor_energy,
                 const Scalar *d_deltaF_nor,
                 Scalar *d_F,
                 const unsigned int maxn_deltaF_nor,
                 const bool have_auxiliary_variables,
                 const unsigned int block_size,
                 const unsigned int tpp);

void hpmc_accept(const unsigned int *d_trial_move_type,
     unsigned int *d_reject_in,
     unsigned int *d_reject_out,
     const Scalar *d_F,
     unsigned int *d_condition,
     const unsigned int seed,
     const unsigned int select,
     const unsigned int timestep,
     const bool patch,
     const bool have_auxiliary_variables,
     const GPUPartition& gpu_partition,
     const unsigned int block_size);

} // end namespace gpu

} // end namespace hpmc
