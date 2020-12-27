// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "IntegratorHPMCMonoGPUTypes.cuh"
#include "hoomd/GPUPartition.cuh"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

namespace hpmc
{
namespace gpu
{
namespace kernel
{

//! Kernel to generate expanded cells
/*! \param d_excell_idx Output array to list the particle indices in the expanded cells
    \param d_excell_size Output array to list the number of particles in each expanded cell
    \param excli Indexer for the expanded cells
    \param d_cell_idx Particle indices in the normal cells
    \param d_cell_size Number of particles in each cell
    \param d_cell_adj Cell adjacency list
    \param ci Cell indexer
    \param cli Cell list indexer
    \param cadji Cell adjacency indexer
    \param ngpu Number of active devices

    gpu_hpmc_excell_kernel executes one thread per cell. It gathers the particle indices from all neighboring cells
    into the output expanded cell.
*/
__global__ void hpmc_excell(unsigned int *d_excell_idx,
                            unsigned int *d_excell_size,
                            const Index2D excli,
                            const unsigned int *d_cell_idx,
                            const unsigned int *d_cell_size,
                            const unsigned int *d_cell_adj,
                            const Index3D ci,
                            const Index2D cli,
                            const Index2D cadji,
                            const unsigned int ngpu)
    {
    // compute the output cell
    unsigned int my_cell = 0;
    my_cell = blockDim.x * blockIdx.x + threadIdx.x;

    if (my_cell >= ci.getNumElements())
        return;

    unsigned int my_cell_size = 0;

    // loop over neighboring cells and build up the expanded cell list
    for (unsigned int offset = 0; offset < cadji.getW(); offset++)
        {
        unsigned int neigh_cell = d_cell_adj[cadji(offset, my_cell)];

        // iterate over per-device cell lists
        for (unsigned int igpu = 0; igpu < ngpu; ++igpu)
            {
            unsigned int neigh_cell_size = d_cell_size[neigh_cell+igpu*ci.getNumElements()];

            for (unsigned int k = 0; k < neigh_cell_size; k++)
                {
                // read in the index of the new particle to add to our cell
                unsigned int new_idx = d_cell_idx[cli(k, neigh_cell)+igpu*cli.getNumElements()];
                d_excell_idx[excli(my_cell_size, my_cell)] = new_idx;
                my_cell_size++;
                }
            }
        }

    // write out the final size
    d_excell_size[my_cell] = my_cell_size;
    }

//! Kernel for grid shift
/*! \param d_postype postype of each particle
    \param d_image Image flags for each particle
    \param N number of particles
    \param box Simulation box
    \param shift Vector by which to translate the particles

    Shift all the particles by a given vector.

    \ingroup hpmc_kernels
*/
__global__ void hpmc_shift(Scalar4 *d_postype,
                          int3 *d_image,
                          const unsigned int N,
                          const BoxDim box,
                          const Scalar3 shift)
    {
    // identify the active cell that this thread handles
    unsigned int my_pidx = blockIdx.x * blockDim.x + threadIdx.x;

    // this thread is inactive if it indexes past the end of the particle list
    if (my_pidx >= N)
        return;

    // pull in the current position
    Scalar4 postype = d_postype[my_pidx];

    // shift the position
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    pos += shift;

    // wrap the particle back into the box
    int3 image = d_image[my_pidx];
    box.wrap(pos, image);

    // write out the new position and orientation
    d_postype[my_pidx] = make_scalar4(pos.x, pos.y, pos.z, postype.w);
    d_image[my_pidx] = image;
    }

//!< Kernel to accept/reject
__global__ void hpmc_sum_energies(const unsigned int *d_update_order_by_ptl,
                 const unsigned int *d_trial_move_type,
                 const Scalar4 *d_trial_vel,
                 const unsigned int *d_reject_out_of_cell,
                 unsigned int *d_reject,
                 unsigned int *d_reject_out,
                 const unsigned int *d_nneigh,
                 const unsigned int *d_nlist,
                 const unsigned int N_old,
                 const unsigned int N,
                 const unsigned int nwork,
                 const unsigned work_offset,
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
                 const unsigned int *d_deltaF_or_config,
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
                 const unsigned int max_neighbors,
                 unsigned int *d_req_neighbors
                 )
    {
    unsigned offset = threadIdx.x;
    unsigned int group_size = blockDim.x;
    unsigned int group = threadIdx.y;
    unsigned int n_groups = blockDim.y;
    bool master = offset == 0;

    // the particle we are handling
    unsigned int i = blockIdx.x*n_groups + group;
    bool active = true;
    if (i >= nwork)
        active = false;
    i += work_offset;

    extern __shared__ char sdata[];

    float *s_energy_old = (float *) sdata;
    float *s_energy_new = (float *) (s_energy_old + n_groups);
    float *s_deltaF = (float *) (s_energy_new + n_groups);
    unsigned int *s_reject = (unsigned int *) (s_deltaF + n_groups);

    unsigned int *s_lookup_old = (unsigned int *) (s_reject + n_groups);
    unsigned int *s_sign_old = (unsigned int *) (s_lookup_old + n_groups*max_neighbors);
    unsigned int *s_lookup_new = (unsigned int *) (s_sign_old + n_groups*max_neighbors);
    unsigned int *s_sign_new = (unsigned int *) (s_lookup_new + n_groups*max_neighbors);
    unsigned int *s_len_old = (unsigned int *) (s_sign_new + n_groups*max_neighbors);
    unsigned int *s_len_new = (unsigned int *) (s_len_old + n_groups);

    bool move_active = false;
    if (active && master)
        {
        s_reject[group] = d_reject_out_of_cell[i];
        s_energy_old[group] = 0.0f;
        s_energy_new[group] = 0.0f;
        s_deltaF[group] = 0.0f;

        s_len_old[group] = 0;
        s_len_new[group] = 0;
        }

    if (active)
        {
        move_active = d_trial_move_type[i] > 0;
        }

    __syncthreads();

    if (active && move_active)
        {
        unsigned int update_order_i = d_update_order_by_ptl[i];

        // iterate over overlapping neighbors in old configuration
        unsigned int nneigh = d_nneigh[i];
        bool accept = true;
        for (unsigned int cur_neigh = offset; cur_neigh < nneigh; cur_neigh += group_size)
            {
            unsigned int primitive = d_nlist[cur_neigh+maxn*i];

            unsigned int j = primitive;
            bool old = true;
            if (j >= N_old)
                {
                j -= N_old;
                old = false;
                }

            // has j been updated? ghost particles are not updated
            bool j_has_been_updated = j < N && d_trial_move_type[j]
                && d_update_order_by_ptl[j] < update_order_i && !d_reject[j];

            // acceptance, reject if current configuration of particle overlaps
            if ((old && !j_has_been_updated) || (!old && j_has_been_updated))
                {
                accept = false;
                break;
                }

            } // end loop over neighbors

        if (!accept)
            {
            atomicMax(&s_reject[group], 1);
            }

        if (patch)
            {
            // iterate over overlapping neighbors in old configuration
            float energy_old = 0.0f;
            unsigned int nneigh = d_nneigh_patch_old[i];
            bool evaluated = false;
            for (unsigned int cur_neigh = offset; cur_neigh < nneigh; cur_neigh += group_size)
                {
                unsigned int primitive = d_nlist_patch_old[cur_neigh+maxn_patch*i];

                unsigned int j = primitive;
                bool old = true;
                if (j >= N_old)
                    {
                    j -= N_old;
                    old = false;
                    }

                // has j been updated? ghost particles are not updated
                bool j_has_been_updated = j < N && d_trial_move_type[j]
                    && d_update_order_by_ptl[j] < update_order_i && !d_reject[j];

                if ((old && !j_has_been_updated) || (!old && j_has_been_updated))
                    {
                    energy_old += d_energy_old[cur_neigh+maxn_patch*i];
                    evaluated = true;
                    }

                } // end loop over neighbors

            if (evaluated)
                atomicAdd(&s_energy_old[group], energy_old);

            // iterate over overlapping neighbors in new configuration
            float energy_new = 0.0f;
            nneigh = d_nneigh_patch_new[i];
            evaluated = false;
            for (unsigned int cur_neigh = offset; cur_neigh < nneigh; cur_neigh += group_size)
                {
                unsigned int primitive = d_nlist_patch_new[cur_neigh+maxn_patch*i];

                unsigned int j = primitive;
                bool old = true;
                if (j >= N_old)
                    {
                    j -= N_old;
                    old = false;
                    }

                // has j been updated? ghost particles are not updated
                bool j_has_been_updated = j < N && d_trial_move_type[j]
                    && d_update_order_by_ptl[j] < update_order_i && !d_reject[j];

                if ((old && !j_has_been_updated) || (!old && j_has_been_updated))
                    {
                    energy_new += d_energy_new[cur_neigh+maxn_patch*i];
                    evaluated = true;
                    }

                } // end loop over neighbors

            if (evaluated)
                atomicAdd(&s_energy_new[group], energy_new);
            }

        if (have_auxiliary_variables && master)
            {
            // depletants with auxiliary ntrial != 0
            unsigned int nneigh = d_deltaF_or_nneigh[i];
            unsigned int nterms = 0;
            unsigned int sign_i = 0;
            for (unsigned int cur_neigh = 0; cur_neigh < nneigh; cur_neigh += nterms)
                {
                nterms = d_deltaF_or_len[maxn_deltaF_or*i + cur_neigh];
                bool new_config = d_deltaF_or_config[maxn_deltaF_or*i + cur_neigh];

                bool has_overlap = false;
                float U_j = 0.0;
                for (unsigned int cur_term = cur_neigh; cur_term < cur_neigh + nterms; ++cur_term)
                    {
                    unsigned int j_flag = d_deltaF_or_nlist[maxn_deltaF_or*i + cur_term];

                    unsigned int j = j_flag >> 2;
                    bool old = j_flag & 2;

                    // has j been updated? ghost particles are not updated
                    bool j_has_been_updated = j < N && d_trial_move_type[j]
                        && d_update_order_by_ptl[j] < update_order_i && !d_reject[j];

                    if ((old && !j_has_been_updated) || (!old && j_has_been_updated))
                        {
                        if (j_flag & 1)
                            {
                            // shortcut
                            has_overlap = true;
                            break;
                            }
                        U_j += d_deltaF_or_energy[maxn_deltaF_or*i + cur_term];
                        }
                    }

                float f_j = has_overlap + (1-has_overlap)*(1.0f-fast::exp(-U_j));
                if (f_j != 0.0)
                    {
                    float f_i = d_deltaF_or[maxn_deltaF_or*i + cur_neigh];
                    if (new_config)
                        atomicAdd(&s_deltaF[group], logf(fabsf(1+f_i*f_j)));
                    else
                        atomicAdd(&s_deltaF[group], -logf(fabsf(1+f_i*f_j)));

                    if (new_config && f_i*f_j < -1)
                        sign_i ^= 1;
                    }
                } // end loop over terms

            if (sign_i != __scalar_as_int(d_trial_vel[i].y))
                {
                atomicMax(&s_reject[group], 1);
                }

            nneigh = d_deltaF_nor_nneigh[i];
            nterms = 0;
            for (unsigned int cur_neigh = 0; cur_neigh < nneigh; cur_neigh += nterms)
                {
                nterms = d_deltaF_nor_len[maxn_deltaF_nor*i + cur_neigh];
                unsigned int k_flag = d_deltaF_nor_k[maxn_deltaF_nor*i + cur_neigh];

                unsigned int k = k_flag >> 2;
                bool k_old = k_flag & 2;
                bool i_old = k_flag & 1;

                // has the inserting particle k been updated? ghost particles are not updated
                bool k_has_been_updated = k < N && d_trial_move_type[k]
                    && d_update_order_by_ptl[k] < update_order_i && !d_reject[k];

                if ((k_old && k_has_been_updated) || (!k_old && !k_has_been_updated))
                    continue;

                bool has_overlap = false;
                bool has_overlap_other = false;
                float U_j = 0.0;
                float U_j_other = 0.0;
                for (unsigned int cur_term = cur_neigh; cur_term < cur_neigh + nterms; ++cur_term)
                    {
                    unsigned int j_flag = d_deltaF_nor_nlist[maxn_deltaF_nor*i + cur_term];

                    unsigned int j = j_flag >> 2;
                    bool j_old = j_flag & 2;

                    if (j == i && i_old != j_old)
                        {
                        continue;
                        }

                    // has j been updated? ghost particles are not updated
                    bool j_has_been_updated = j < N && d_trial_move_type[j]
                        && d_update_order_by_ptl[j] < update_order_i && !d_reject[j];

                    if (i == j || (j_old && !j_has_been_updated) || (!j_old && j_has_been_updated))
                        {
                        if (j_flag & 1)
                            {
                            has_overlap = true;

                            if (j != i)
                                has_overlap_other = true;
                            }

                        float t = d_deltaF_nor_energy[maxn_deltaF_nor*i + cur_term];
                        U_j += t;
                        if (j != i)
                            U_j_other += t;
                        }
                    }

                float f_j = has_overlap + (1-has_overlap)*(1.0f-fast::exp(-U_j));
                float f_k = d_deltaF_nor[maxn_deltaF_nor*i + cur_neigh];
                unsigned int sign_k = 0;
                if (f_j != 0.0)
                    {
                    if (!i_old)
                        atomicAdd(&s_deltaF[group], logf(fabs(1+f_k*f_j)));
                    else
                        atomicAdd(&s_deltaF[group], -logf(fabs(1+f_k*f_j)));

                    if (f_k*f_j < -1.0f)
                        sign_k ^= 1;
                    }

                // add back neighbor term (excluding i) on the other side of the fraction
                float f_j_other = has_overlap_other + (1-has_overlap_other)*(1.0f-fast::exp(-U_j_other));
                if (f_j_other != 0.0)
                    {
                    if (!i_old)
                        atomicAdd(&s_deltaF[group], -logf(fabs(1+f_k*f_j_other)));
                    else
                        atomicAdd(&s_deltaF[group], logf(fabs(1+f_k*f_j_other)));

                    if (f_k*f_j_other < -1.0f)
                        sign_k ^= 1;
                    }

                if (sign_k)
                    {
                    unsigned int *s_lookup;
                    unsigned int *s_sign;
                    unsigned int *s_len;
                    if (i_old)
                        {
                        s_lookup = &s_lookup_old[group*max_neighbors];
                        s_sign = &s_sign_old[group*max_neighbors];
                        s_len = &s_len_old[group];
                        }
                    else
                        {
                        s_lookup = &s_lookup_new[group*max_neighbors];
                        s_sign = &s_sign_new[group*max_neighbors];
                        s_len = &s_len_new[group];
                        }

                    // store sign change in table for neighbor k
                    unsigned int l = 0;
                    for (; l < *s_len; ++l)
                        {
                        if (s_lookup[l] == k)
                            break;
                        }

                    if (l == *s_len)
                        {
                        // add at the end
                        unsigned int insert_pos = *s_len;
                        if (insert_pos >= max_neighbors)
                            {
                            #if (__CUDA_ARCH__ >= 600)
                            atomicMax_system(&d_req_neighbors[i], insert_pos);
                            #else
                            atomicMax(&d_req_neighbors[i], insert_pos);
                            #endif
                            }

                        if (insert_pos < max_neighbors)
                            {
                            s_lookup[insert_pos] = k;
                            s_sign[insert_pos] = 1;
                            }

                        *s_len++;
                        }
                    else
                        {
                        // update sign for neighbor k
                        s_sign[l] ^= 1;
                        }
                    }
                } // end loop over terms

            // have we changed any neighbor's sign?
            unsigned int len = s_len_old[group];
            for (unsigned int l = 0; l < len; ++l)
                {
                if (s_sign_old[l])
                    atomicMax(&s_reject[group], 1);
                }
            // have we changed any neighbor's sign?
            len = s_len_new[group];
            for (unsigned int l = 0; l < len; ++l)
                {
                if (s_sign_new[l])
                    atomicMax(&s_reject[group], 1);
                }
            } // end depletants

        } // end if (active && move_active)

    __syncthreads();

    if (master && active)
        {
        d_reject_out[i] = s_reject[group];
        if (move_active)
            {
            // write out final free energy change
            d_F[i] = s_energy_old[group] - s_energy_new[group] + s_deltaF[group];
            }
        }
    }

//!< Kernel to evaluate convergence
__global__ void hpmc_accept(
                 const unsigned int *d_trial_move_type,
                 unsigned int *d_reject_in,
                 unsigned int *d_reject_out,
                 const Scalar *d_F,
                 unsigned int *d_condition,
                 const unsigned int seed,
                 const unsigned int select,
                 const unsigned int timestep,
                 const bool patch,
                 const bool have_auxiliary_variables,
                 const unsigned int nwork,
                 const unsigned work_offset)
    {
    // the particle we are handling
    unsigned int work_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (work_idx >= nwork)
        return;
    unsigned int i = work_idx + work_offset;

    // is this particle considered?
    bool move_active = d_trial_move_type[i] > 0;

    // combine with reject flag from gen_moves for particles which are always rejected
    bool reject = d_reject_out[i];

    if (move_active)
        {
        float deltaF = d_F[i]; // deltaF = F_old - F_new

        // Metropolis-Hastings
        hoomd::RandomGenerator rng_i(hoomd::RNGIdentifier::HPMCMonoAccept, seed, i, select, timestep);
        bool accept = !reject && ((!patch && !have_auxiliary_variables)
            || (hoomd::detail::generate_canonical<double>(rng_i) < slow::exp(deltaF)));

        if ((accept && d_reject_in[i]) || (!accept && !d_reject_in[i]))
            {
            // flag that we're not done yet (a trivial race condition upon write)
            *d_condition = 1;
            }

        // write out to device memory
        d_reject_out[i] = accept ? 0 : 1;
        }
    }

//! Generate number of depletants per particle
__global__ void generate_num_depletants(const unsigned int seed,
                                        const unsigned int timestep,
                                        const unsigned int select,
                                        const unsigned int num_types,
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
    Index2D typpair_idx(num_types);
    unsigned int type_i = __scalar_as_int(d_postype[idx].w);
    d_n_depletants[idx] = hoomd::PoissonDistribution<Scalar>(
        d_lambda[type_i*depletant_idx.getNumElements()+depletant_idx(depletant_type_a,depletant_type_b)])(rng_poisson);
    }
} // end namespace kernel

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
                 const unsigned int block_size)
    {
    assert(d_excell_idx);
    assert(d_excell_size);
    assert(d_cell_idx);
    assert(d_cell_size);
    assert(d_cell_adj);

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    if (max_block_size == -1)
        {
        hipFuncAttributes attr;
        hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_excell));
        max_block_size = attr.maxThreadsPerBlock;
        }

    // setup the grid to run the kernel
    unsigned int run_block_size = min(block_size, (unsigned int)max_block_size);
    dim3 threads(run_block_size, 1, 1);
    dim3 grid(ci.getNumElements() / run_block_size + 1, 1, 1);

    hipLaunchKernelGGL(kernel::hpmc_excell, dim3(grid), dim3(threads), 0, 0, d_excell_idx,
                                           d_excell_size,
                                           excli,
                                           d_cell_idx,
                                           d_cell_size,
                                           d_cell_adj,
                                           ci,
                                           cli,
                                           cadji,
                                           ngpu);

    }

//! Kernel driver for kernel::hpmc_shift()
void hpmc_shift(Scalar4 *d_postype,
                int3 *d_image,
                const unsigned int N,
                const BoxDim& box,
                const Scalar3 shift,
                const unsigned int block_size)
    {
    assert(d_postype);
    assert(d_image);

    // setup the grid to run the kernel
    dim3 threads_shift(block_size, 1, 1);
    dim3 grid_shift(N / block_size + 1, 1, 1);

    hipLaunchKernelGGL(kernel::hpmc_shift, dim3(grid_shift), dim3(threads_shift), 0, 0, d_postype,
                                                      d_image,
                                                      N,
                                                      box,
                                                      shift);

    // after this kernel we return control of cuda managed memory to the host
    hipDeviceSynchronize();
    }


void hpmc_sum_energies(const unsigned int *d_update_order_by_ptl,
                 const unsigned int *d_trial_move_type,
                 const Scalar4 *d_trial_vel,
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
                 const unsigned int *d_deltaF_or_config,
                 const unsigned int maxn_deltaF_or,
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
                 const unsigned int tpp,
                 const unsigned int max_neighbors,
                 unsigned int *d_req_neighbors)
    {
    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    if (max_block_size == -1)
        {
        hipFuncAttributes attr;
        hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_sum_energies));
        max_block_size = attr.maxThreadsPerBlock;
        }

    // setup the grid to run the kernel
    unsigned int run_block_size = min(block_size, (unsigned int)max_block_size);

    // threads per particle
    unsigned int cur_tpp = min(run_block_size,tpp);
    while (run_block_size % cur_tpp != 0)
        cur_tpp--;

    unsigned int n_groups = run_block_size/cur_tpp;
    dim3 threads(cur_tpp, n_groups, 1);

    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;
        const unsigned int num_blocks = nwork/n_groups + 1;
        dim3 grid(num_blocks, 1, 1);

        unsigned int shared_bytes = n_groups * (3*sizeof(float) + 3*sizeof(unsigned int))
                                    + n_groups*max_neighbors*4*sizeof(unsigned int);

        hipLaunchKernelGGL(kernel::hpmc_sum_energies, grid, threads, shared_bytes, 0,
            d_update_order_by_ptl,
            d_trial_move_type,
            d_trial_vel,
            d_reject_out_of_cell,
            d_reject,
            d_reject_out,
            d_nneigh,
            d_nlist,
            N_old,
            N,
            nwork,
            range.first,
            maxn,
            patch,
            d_nlist_patch_old,
            d_nlist_patch_new,
            d_nneigh_patch_old,
            d_nneigh_patch_new,
            d_energy_old,
            d_energy_new,
            maxn_patch,
            d_deltaF_or_nneigh,
            d_deltaF_or_len,
            d_deltaF_or_nlist,
            d_deltaF_or_energy,
            d_deltaF_or,
            d_deltaF_or_config,
            maxn_deltaF_or,
            d_deltaF_nor_nneigh,
            d_deltaF_nor_len,
            d_deltaF_nor_k,
            d_deltaF_nor_nlist,
            d_deltaF_nor_energy,
            d_deltaF_nor,
            d_F,
            maxn_deltaF_nor,
            have_auxiliary_variables,
            max_neighbors,
            d_req_neighbors
            );
        }

    }

//!< Kernel to evaluate convergence
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
     const unsigned int block_size)
    {
    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    if (max_block_size == -1)
        {
        hipFuncAttributes attr;
        hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_accept));
        max_block_size = attr.maxThreadsPerBlock;
        }

    // setup the grid to run the kernel
    unsigned int run_block_size = min(block_size, (unsigned int)max_block_size);

    dim3 threads(run_block_size, 1, 1);

    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;
        const unsigned int num_blocks = nwork/run_block_size + 1;
        dim3 grid(num_blocks, 1, 1);

        hipLaunchKernelGGL(kernel::hpmc_accept, grid, threads, 0, 0,
            d_trial_move_type,
            d_reject_in,
            d_reject_out,
            d_F,
            d_condition,
            seed,
            select,
            timestep,
            patch,
            have_auxiliary_variables,
            nwork,
            range.first);
        }
    }
} // end namespace gpu
} // end namespace hpmc

