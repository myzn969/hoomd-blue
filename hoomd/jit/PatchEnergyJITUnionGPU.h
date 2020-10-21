#pragma once

#ifdef ENABLE_HIP

#include "PatchEnergyJITUnion.h"
#include "GPUEvalFactory.h"
#include "hoomd/managed_allocator.h"
#include "EvaluatorUnionGPU.cuh"

#include <vector>

//! Evaluate patch energies via runtime generated code, GPU version
template<class Shape>
class PYBIND11_EXPORT PatchEnergyJITUnionGPU : public PatchEnergyJITUnion<Shape>
    {
    public:
        //! Constructor
        PatchEnergyJITUnionGPU(std::shared_ptr<SystemDefinition> sysdef,
            std::shared_ptr<ExecutionConfiguration> exec_conf,
            const std::string& llvm_ir_iso, Scalar r_cut_iso,
            const unsigned int array_size_iso,
            const std::string& llvm_ir_union, Scalar r_cut_union,
            const unsigned int array_size_union,
            const std::string& code,
            const std::string& kernel_name,
            const std::vector<std::string>& options,
            const std::string& cuda_devrt_library_path,
            unsigned int compute_arch)
            : PatchEnergyJITUnion<Shape>(sysdef, exec_conf, llvm_ir_iso, r_cut_iso, array_size_iso, llvm_ir_union, r_cut_union, array_size_union),
              m_gpu_factory(exec_conf, code, kernel_name, options, cuda_devrt_library_path, compute_arch),
              m_d_union_params(this->m_sysdef->getParticleData()->getNTypes(),
                jit::union_params_t(), managed_allocator<jit::union_params_t>(this->m_exec_conf->isCUDAEnabled()))
            {
            m_gpu_factory.setAlphaPtr<Shape>(&this->m_alpha.front());
            m_gpu_factory.setAlphaUnionPtr<Shape>(&this->m_alpha_union.front());
            m_gpu_factory.setUnionParamsPtr<Shape>(&this->m_d_union_params.front());
            m_gpu_factory.setRCutUnion<Shape>(this->m_rcut_union);

            // tuning params for patch narrow phase
            std::vector<std::vector< unsigned int > > valid_params_patch(this->m_sysdef->getParticleData()->getNTypes()+1);
            const unsigned int narrow_phase_max_threads_per_eval = this->m_exec_conf->dev_prop.warpSize;
            auto& launch_bounds = m_gpu_factory.getLaunchBounds();
            for (auto cur_launch_bounds: launch_bounds)
                {
                for (unsigned int group_size=1; group_size <= cur_launch_bounds; group_size*=2)
                    {
                    for (unsigned int eval_threads=1; eval_threads <= narrow_phase_max_threads_per_eval; eval_threads *= 2)
                        {
                        if ((cur_launch_bounds % (group_size*eval_threads)) == 0)
                            valid_params_patch[0].push_back(cur_launch_bounds*1000000 + group_size*100 + eval_threads);
                        }
                    }
                }

            unsigned int tuning_bits = jit::union_params_t::getTuningBits();
            for (unsigned int itype = 0; itype < this->m_sysdef->getParticleData()->getNTypes(); ++itype)
                {
                for (int param = 0; param < (1<<tuning_bits); ++param)
                    valid_params_patch[1+itype].push_back(param);
                }
            m_tuner_narrow_patch.reset(new Autotuner(valid_params_patch, 5, 100000, "hpmc_narrow_patch", this->m_exec_conf));
            }

        virtual ~PatchEnergyJITUnionGPU() {}

        //! Set the per-type constituent particles
        /*! \param type The particle type to set the constituent particles for
            \param rcut The maximum cutoff over all constituent particles for this type
            \param types The type IDs for every constituent particle
            \param positions The positions
            \param orientations The orientations
            \param leaf_capacity Number of particles in OBB tree leaf
         */
        virtual void setParam(unsigned int type,
            pybind11::list types,
            pybind11::list positions,
            pybind11::list orientations,
            pybind11::list diameters,
            pybind11::list charges,
            unsigned int leaf_capacity=4);

        //! Asynchronously launch the JIT kernel
        /*! \param args Kernel arguments
            \param hStream stream to execute on
            */
        virtual void computePatchEnergyGPU(const typename hpmc::PatchEnergy<Shape>::gpu_args_t& args, hipStream_t hStream);

        //! Method to be called when number of types changes
        virtual void slotNumTypesChange()
            {
            PatchEnergyJITUnion<Shape>::slotNumTypesChange();
            unsigned int ntypes = this->m_sysdef->getParticleData()->getNTypes();
            m_d_union_params.resize(ntypes);

            // update device side pointer
            m_gpu_factory.setUnionParamsPtr<Shape>(&m_d_union_params.front());
            }

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

        std::vector<jit::union_params_t, managed_allocator<jit::union_params_t> > m_d_union_params;   //!< Parameters for each particle type on GPU
    };


//! Set the per-type constituent particles
template<class Shape>
void PatchEnergyJITUnionGPU<Shape>::setParam(unsigned int type,
    pybind11::list types,
    pybind11::list positions,
    pybind11::list orientations,
    pybind11::list diameters,
    pybind11::list charges,
    unsigned int leaf_capacity)
    {
    // set parameters in base class
    PatchEnergyJITUnion<Shape>::setParam(type, types, positions, orientations, diameters, charges, leaf_capacity);

    unsigned int N = len(positions);

    hpmc::detail::OBB *obbs = new hpmc::detail::OBB[N];

    jit::union_params_t params(N, true);

    // set shape parameters
    for (unsigned int i = 0; i < N; i++)
        {
        pybind11::list positions_i = pybind11::cast<pybind11::list>(positions[i]);
        vec3<float> pos = vec3<float>(pybind11::cast<float>(positions_i[0]), pybind11::cast<float>(positions_i[1]), pybind11::cast<float>(positions_i[2]));
        pybind11::list orientations_i = pybind11::cast<pybind11::list>(orientations[i]);
        float s = pybind11::cast<float>(orientations_i[0]);
        float x = pybind11::cast<float>(orientations_i[1]);
        float y = pybind11::cast<float>(orientations_i[2]);
        float z = pybind11::cast<float>(orientations_i[3]);
        quat<float> orientation(s, vec3<float>(x,y,z));

        float diameter = pybind11::cast<float>(diameters[i]);
        float charge = pybind11::cast<float>(charges[i]);
        params.mtype[i] = pybind11::cast<unsigned int>(types[i]);
        params.mpos[i] = pos;
        params.morientation[i] = orientation;
        params.mdiameter[i] = diameter;
        params.mcharge[i] = charge;

        // use a spherical OBB of radius 0.5*d
        obbs[i] = hpmc::detail::OBB(pos,0.5f*diameter);

        // we do not support exclusions
        obbs[i].mask = 1;
        }

    // build tree and store proxy structure
    hpmc::detail::OBBTree tree;
    bool internal_nodes_spheres = false;
    tree.buildTree(obbs, N, leaf_capacity, internal_nodes_spheres);
    delete [] obbs;
    bool managed = true;
    params.tree = hpmc::detail::GPUTree(tree, managed);

    // store result
    m_d_union_params[type] = params;

    // cudaMemadviseReadMostly
    m_d_union_params[type].set_memory_hint();
    }

//! Kernel driver for kernel::hpmc_narrow_phase_patch
template<class Shape>
void PatchEnergyJITUnionGPU<Shape>::computePatchEnergyGPU(const typename hpmc::PatchEnergy<Shape>::gpu_args_t& args,
    hipStream_t hStream)
    {
    #ifdef __HIP_PLATFORM_NVCC__
    assert(args.d_postype);
    assert(args.d_orientation);

    unsigned int param = m_tuner_narrow_patch->getParam(0);
    unsigned int block_size = param/1000000;
    unsigned int req_tpp = (param%1000000)/100;
    unsigned int eval_threads = param % 100;

    this->m_exec_conf->beginMultiGPU();
    m_tuner_narrow_patch->begin();

    const unsigned int *d_type_params = m_tuner_narrow_patch->getDeviceParams()+1;

    // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
    unsigned int run_block_size = std::min(block_size,
        m_gpu_factory.getKernelMaxThreads<Shape>(0, eval_threads, block_size)); // fixme GPU 0

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
                                          m_d_union_params.size()*sizeof(jit::union_params_t);

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

    // allocate some extra shared mem to store union shape parameters
    unsigned int max_extra_bytes = this->m_exec_conf->dev_prop.sharedMemPerBlock - shared_bytes - kernel_shared_bytes;

    // determine dynamically requested shared memory
    char *ptr = (char *)nullptr;
    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int i = 0; i < m_d_union_params.size(); ++i)
        {
        m_d_union_params[i].load_shared(ptr, available_bytes, d_type_params[i]);
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
            max_extra_bytes,
            d_type_params);

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
void export_PatchEnergyJITUnionGPU(pybind11::module &m, const std::string& name);

#ifdef __EXPORT_IMPL__
template<class Shape>
void export_PatchEnergyJITUnionGPU(pybind11::module &m, const std::string& name)
    {
    pybind11::class_<PatchEnergyJITUnionGPU<Shape>, PatchEnergyJITUnion<Shape>,
            std::shared_ptr<PatchEnergyJITUnionGPU<Shape> > >(m, name.c_str())
            .def(pybind11::init< std::shared_ptr<SystemDefinition>,
                                 std::shared_ptr<ExecutionConfiguration>,
                                 const std::string&, Scalar, const unsigned int,
                                 const std::string&, Scalar, const unsigned int,
                                 const std::string&, const std::string&,
                                 const std::vector<std::string>&,
                                 const std::string&,
                                 unsigned int>())
            .def("setParam",&PatchEnergyJITUnionGPU<Shape>::setParam)
            ;
    }
#endif
#endif
