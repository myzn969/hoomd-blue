// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "IntegratorHPMCMonoGPUDepletants.cuh"
#include "hoomd/GPUPartition.cuh"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/CachedAllocator.h"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

namespace hpmc
{
namespace gpu
{
namespace kernel
{

//! Generate number of depletants per particle
__global__ void generate_num_depletants(const unsigned int seed,
                                        const unsigned int timestep,
                                        const unsigned int select,
                                        const unsigned int depletant_type_a,
                                        const unsigned int depletant_type_b,
                                        const Index2D depletant_idx,
                                        const unsigned int work_offset,
                                        const unsigned int nwork,
                                        const Scalar *d_lambda,
                                        const Scalar4 *d_postype,
                                        unsigned int *d_n_depletants)
    {
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= nwork)
        return;

    idx += work_offset;

    hoomd::RandomGenerator rng_poisson(hoomd::RNGIdentifier::HPMCDepletantNum, idx, seed, timestep,
        select*depletant_idx.getNumElements() + depletant_idx(depletant_type_a,depletant_type_b));
    unsigned int type_i = __scalar_as_int(d_postype[idx].w);
    d_n_depletants[idx] = hoomd::PoissonDistribution<Scalar>(
        d_lambda[type_i*depletant_idx.getNumElements()+depletant_idx(depletant_type_a,depletant_type_b)])(rng_poisson);
    }

//! Generate number of depletants per particle (ntrial version)
__global__ void generate_num_depletants_ntrial(const Scalar4 *d_vel,
                                        const Scalar4 *d_trial_vel,
                                        const float gamma,
                                        const unsigned int depletant_type_a,
                                        const unsigned int depletant_type_b,
                                        const Index2D depletant_idx,
                                        const Scalar *d_lambda,
                                        const Scalar4 *d_postype,
                                        unsigned int *d_n_depletants,
                                        const unsigned int N_local,
                                        const unsigned int work_offset,
                                        const unsigned int nwork)
    {
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= nwork)
        return;

    unsigned int i = idx + work_offset;

    unsigned int i_trial_config = blockIdx.y;
    unsigned int new_config = i_trial_config & 1;

    if (i >= N_local && new_config)
        return; // ghosts only exist in the old config

    // draw a Poisson variate according to the seed stored in the auxillary variable (vel.x)
    unsigned int seed_i = new_config ? __scalar_as_int(d_trial_vel[i].x) : __scalar_as_int(d_vel[i].x);
    hoomd::RandomGenerator rng_num(hoomd::RNGIdentifier::HPMCDepletantNum,
        depletant_idx(depletant_type_a, depletant_type_b), seed_i);

    unsigned int type_i = __scalar_as_int(d_postype[i].w);
    Scalar lambda = d_lambda[type_i*depletant_idx.getNumElements()+depletant_idx(depletant_type_a,depletant_type_b)];
    unsigned int n = hoomd::PoissonDistribution<Scalar>(gamma*lambda)(rng_num);

    // store result
    d_n_depletants[2*i+new_config] = n;
    }

__global__ void hpmc_reduce_counters(const unsigned int ngpu,
                     const unsigned int pitch,
                     const hpmc_counters_t *d_per_device_counters,
                     hpmc_counters_t *d_counters,
                     const unsigned int implicit_pitch,
                     const Index2D depletant_idx,
                     const hpmc_implicit_counters_t *d_per_device_implicit_counters,
                     hpmc_implicit_counters_t *d_implicit_counters)
    {
    for (unsigned int igpu = 0; igpu < ngpu; ++igpu)
        {
        *d_counters = *d_counters + d_per_device_counters[igpu*pitch];

        for (unsigned int itype = 0; itype < depletant_idx.getNumElements(); ++itype)
            d_implicit_counters[itype] = d_implicit_counters[itype] + d_per_device_implicit_counters[itype+igpu*implicit_pitch];
        }
    }

} // end namespace kernel

void generate_num_depletants(const unsigned int seed,
                             const unsigned int timestep,
                             const unsigned int select,
                             const unsigned int depletant_type_a,
                             const unsigned int depletant_type_b,
                             const Index2D depletant_idx,
                             const Scalar *d_lambda,
                             const Scalar4 *d_postype,
                             unsigned int *d_n_depletants,
                             const unsigned int block_size,
                             const hipStream_t *streams,
                             const GPUPartition& gpu_partition)
    {
    // determine the maximum block size and clamp the input block size down
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        hipFuncAttributes attr;
        hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::generate_num_depletants));
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);
        unsigned int nwork = range.second - range.first;

        hipLaunchKernelGGL(kernel::generate_num_depletants, nwork/run_block_size+1, run_block_size, 0, streams[idev],
            seed,
            timestep,
            select,
            depletant_type_a,
            depletant_type_b,
            depletant_idx,
            range.first,
            nwork,
            d_lambda,
            d_postype,
            d_n_depletants);
        }
    }

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
                                    const hipStream_t *streams)
    {
    // determine the maximum block size and clamp the input block size down
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        hipFuncAttributes attr;
        hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::generate_num_depletants_ntrial));
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // add ghosts to final range
        if (idev == (int)gpu_partition.getNumActiveGPUs()-1 && add_ghosts)
            nwork += n_ghosts;

        if (!nwork) continue;

        dim3 grid(nwork/run_block_size + 1, 2, 1);
        dim3 threads(run_block_size, 1, 1);

        hipLaunchKernelGGL((kernel::generate_num_depletants_ntrial), grid, threads, 0, streams[idev],
            d_vel,
            d_trial_vel,
            gamma,
            depletant_type_a,
            depletant_type_b,
            depletant_idx,
            d_lambda,
            d_postype,
            d_n_depletants,
            N_local,
            range.first,
            nwork);
        }
    }

void get_max_num_depletants(unsigned int *d_n_depletants,
                            unsigned int *max_n_depletants,
                            const hipStream_t *streams,
                            const GPUPartition& gpu_partition,
                            CachedAllocator& alloc)
    {
    assert(d_n_depletants);
    thrust::device_ptr<unsigned int> n_depletants(d_n_depletants);
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        #ifdef __HIP_PLATFORM_HCC__
        max_n_depletants[idev] = thrust::reduce(thrust::hip::par(alloc).on(streams[idev]),
        #else
        max_n_depletants[idev] = thrust::reduce(thrust::cuda::par(alloc).on(streams[idev]),
        #endif
            n_depletants + range.first,
            n_depletants + range.second,
            0,
            thrust::maximum<unsigned int>());
        }
    }

//! Compute the max # of depletants per particle, trial insertion, and configuration
void get_max_num_depletants_ntrial(const float gamma,
                            unsigned int *d_n_depletants,
                            unsigned int *max_n_depletants,
                            const bool add_ghosts,
                            const unsigned int n_ghosts,
                            const hipStream_t *streams,
                            const GPUPartition& gpu_partition,
                            CachedAllocator& alloc)
    {
    assert(d_n_depletants);
    thrust::device_ptr<unsigned int> n_depletants(d_n_depletants);
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // add ghosts to final range
        if (idev == (int)gpu_partition.getNumActiveGPUs()-1 && add_ghosts)
            nwork += n_ghosts;

        #ifdef __HIP_PLATFORM_HCC__
        max_n_depletants[idev] = thrust::reduce(thrust::hip::par(alloc).on(streams[idev]),
        #else
        max_n_depletants[idev] = thrust::reduce(thrust::cuda::par(alloc).on(streams[idev]),
        #endif
            n_depletants + range.first*2,
            n_depletants + (range.first+nwork)*2,
            0,
            thrust::maximum<unsigned int>());
        }
    }

void reduce_counters(const unsigned int ngpu,
                     const unsigned int pitch,
                     const hpmc_counters_t *d_per_device_counters,
                     hpmc_counters_t *d_counters,
                     const unsigned int implicit_pitch,
                     const Index2D depletant_idx,
                     const hpmc_implicit_counters_t *d_per_device_implicit_counters,
                     hpmc_implicit_counters_t *d_implicit_counters)
    {
    hipLaunchKernelGGL(kernel::hpmc_reduce_counters, 1, 1, 0, 0,
                     ngpu,
                     pitch,
                     d_per_device_counters,
                     d_counters,
                     implicit_pitch,
                     depletant_idx,
                     d_per_device_implicit_counters,
                     d_implicit_counters);
    }

} // end namespace gpu
} // end namespace hpmc

