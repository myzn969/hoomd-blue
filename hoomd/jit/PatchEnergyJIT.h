#ifndef _PATCH_ENERGY_JIT_H_
#define _PATCH_ENERGY_JIT_H_

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/hpmc/IntegratorHPMCMono.h"
#include "hoomd/managed_allocator.h"

#include "EvalFactory.h"


//! Evaluate patch energies via runtime generated code
/*! This class enables the widest possible use-cases of patch energies in HPMC with low energy barriers for users to add
    custom interactions that execute with high performance. It provides a generic interface for returning the energy of
    interaction between a pair of particles. The actual computation is performed by code that is loaded and compiled at
    run time using LLVM.

    The user provides LLVM IR code containing a function 'eval' with the defined function signature. On construction,
    this class uses the LLVM library to compile that IR down to machine code and obtain a function pointer to call.

    This is the first use of LLVM in HOOMD and it is experimental. As additional areas are identified as
    useful applications of LLVM, we will want to factor out some of the comment elements of this code
    into a generic LLVM module class. (i.e. handle broadcasting the string and compiling it in one place,
    with specific implementations requesting the function pointers they need).

    LLVM execution is managed with the KaleidoscopeJIT class in m_JIT. On construction, the LLVM module is loaded and
    compiled. KaleidoscopeJIT handles construction of C++ static members, etc.... When m_JIT is deleted, all of the compiled
    code and memory used in the module is deleted. KaleidoscopeJIT takes care of destructing C++ static members inside the
    module.

    LLVM JIT is capable of calling any function in the hosts address space. PatchEnergyJIT does not take advantage of
    that, limiting the user to a very specific API for computing the energy between a pair of particles.
*/
template<class Shape>
class PYBIND11_EXPORT PatchEnergyJIT : public hpmc::PatchEnergy<Shape>
    {
    public:
        //! Constructor
        PatchEnergyJIT(std::shared_ptr<ExecutionConfiguration> exec_conf, const std::string& llvm_ir, Scalar r_cut,
                       const unsigned int array_size);

        //! Get the maximum r_ij radius beyond which energies are always 0
        virtual Scalar getRCut()
            {
            return m_r_cut;
            }

        //! Get the maximum r_ij radius beyond which energies are always 0
        virtual inline Scalar getAdditiveCutoff(unsigned int type)
            {
            // this potential corresponds to a point particle
            return 0.0;
            }

        //! evaluate the energy of the patch interaction
        /*! \param r_ij Vector pointing from particle i to j
            \param type_i Integer type index of particle i
            \param d_i Diameter of particle i
            \param charge_i Charge of particle i
            \param q_i Orientation quaternion of particle i
            \param type_j Integer type index of particle j
            \param q_j Orientation quaternion of particle j
            \param d_j Diameter of particle j
            \param charge_j Charge of particle j
            \returns Energy of the patch interaction.
        */
        virtual float energy(const vec3<float>& r_ij,
            unsigned int type_i,
            const quat<float>& q_i,
            float d_i,
            float charge_i,
            unsigned int type_j,
            const quat<float>& q_j,
            float d_j,
            float charge_j)
            {
            return m_eval(r_ij, type_i, q_i, d_i, charge_i, type_j, q_j, d_j, charge_j);
            }

        static pybind11::object getAlphaNP(pybind11::object self)
            {
            auto self_cpp = self.cast<PatchEnergyJIT<Shape> *>();
            return pybind11::array(self_cpp->m_alpha_size, self_cpp->m_factory->getAlphaArray(), self);
            }

    protected:
        std::shared_ptr<ExecutionConfiguration> m_exec_conf; //!< The exceuction configuration
        //! function pointer signature
        typedef float (*EvalFnPtr)(const vec3<float>& r_ij, unsigned int type_i, const quat<float>& q_i, float, float, unsigned int type_j, const quat<float>& q_j, float, float);
        Scalar m_r_cut;                             //!< Cutoff radius
        std::shared_ptr<EvalFactory> m_factory;       //!< The factory for the evaluator function
        EvalFactory::EvalFnPtr m_eval;                //!< Pointer to evaluator function inside the JIT module
        unsigned int m_alpha_size;                  //!< Size of array
        std::vector<float, managed_allocator<float> > m_alpha; //!< Array containing adjustable parameters
    };

/*! \param exec_conf The execution configuration (used for messages and MPI communication)
    \param llvm_ir Contents of the LLVM IR to load
    \param r_cut Center to center distance beyond which the patch energy is 0

    After construction, the LLVM IR is loaded, compiled, and the energy() method is ready to be called.
*/
template<class Shape>
PatchEnergyJIT<Shape>::PatchEnergyJIT(std::shared_ptr<ExecutionConfiguration> exec_conf,
                const std::string& llvm_ir, Scalar r_cut, const unsigned int array_size)
    : m_exec_conf(exec_conf), m_r_cut(r_cut), m_alpha_size(array_size),
      m_alpha(array_size, 0.0, managed_allocator<float>(m_exec_conf->isCUDAEnabled()))
    {
    // build the JIT.
    m_factory = std::shared_ptr<EvalFactory>(new EvalFactory(llvm_ir));

    // get the evaluator
    m_eval = m_factory->getEval();

    if (!m_eval)
        {
        exec_conf->msg->error() << m_factory->getError() << std::endl;
        throw std::runtime_error("Error compiling JIT code.");
        }

    m_factory->setAlphaArray(&m_alpha.front());
    }

template<class Shape>
void export_PatchEnergyJIT(pybind11::module &m, const std::string& name);

#ifdef __EXPORT_IMPL__
template<class Shape>
void export_PatchEnergyJIT(pybind11::module &m, const std::string& name)
    {
    pybind11::class_<typename hpmc::PatchEnergy<Shape>,
               std::shared_ptr<typename hpmc::PatchEnergy<Shape> > >(m, name.c_str())
              .def(pybind11::init< >());
    pybind11::class_<PatchEnergyJIT<Shape>, typename hpmc::PatchEnergy<Shape>,
                     std::shared_ptr<PatchEnergyJIT<Shape> > >(m, name.c_str())
            .def(pybind11::init< std::shared_ptr<ExecutionConfiguration>,
                                 const std::string&,
                                 Scalar,
                                 const unsigned int >())
            .def("getRCut", &PatchEnergyJIT<Shape>::getRCut)
            .def("energy", &PatchEnergyJIT<Shape>::energy)
            .def_property_readonly("alpha_iso",&PatchEnergyJIT<Shape>::getAlphaNP)
            ;
    }
#endif

#endif // _PATCH_ENERGY_JIT_H_
