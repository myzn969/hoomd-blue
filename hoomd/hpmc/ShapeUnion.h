// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "hoomd/VectorMath.h"
#include "ShapeSphere.h"    //< For the base template of test_overlap
#include "ShapeSpheropolyhedron.h"
#include "ShapeConvexPolyhedron.h"
#include "GPUTree.h"

#include "hoomd/AABB.h"
#include "hoomd/ManagedArray.h"

#ifndef __SHAPE_UNION_H__
#define __SHAPE_UNION_H__

/*! \file ShapeUnion.h
    \brief Defines the ShapeUnion templated aggregate shape
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#include <iostream>
#endif

namespace hpmc
{

namespace detail
{

//! Stores the overlapping node pairs from a prior traversal
/* This data structure is used to accelerate the random choice of overlapping
   node pairs when depletants are reinserted, eliminating the need to traverse
   the same tree for all reinsertion attempts.
 */
struct union_depletion_storage
    {
    //! The inclusive prefix sum over previous weights of overlapping node pairs
    OverlapReal accumulated_weight;

    //! The node in tree a
    unsigned int cur_node_a;

    //! The node in tree b
    unsigned int cur_node_b;
    };

//! Data structure for shape composed of a union of multiple shapes
template<class Shape>
struct union_params : param_base
    {
    typedef GPUTree gpu_tree_type; //!< Handy typedef for GPUTree template
    typedef typename Shape::param_type mparam_type;

    //! Default constructor
    DEVICE inline union_params()
        : diameter(0.0), N(0), ignore(0)
        { }

    //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param available_bytes Size of remaining shared memory allocation
        \param mask bitmask indicating which arrays we should attempt to load
     */
    HOSTDEVICE inline void load_shared(char *& ptr, unsigned int &available_bytes,
                                       unsigned int mask) const
        {
        const unsigned int tree_bits = tree.getTuningBits();
        tree.load_shared(ptr, available_bytes, mask);
        mask >>= tree_bits;

        if (mask & 1)
            mpos.load_shared(ptr, available_bytes);
        bool params_in_shared_mem = false;
        if (mask & 2)
            params_in_shared_mem = mparams.load_shared(ptr, available_bytes);

        if (mask & 4)
            moverlap.load_shared(ptr, available_bytes);

        if (mask & 8)
            morientation.load_shared(ptr, available_bytes);

        // load all member parameters
        #if defined (__HIP_DEVICE_COMPILE__)
        __syncthreads();
        #endif

        mask >>= 4; // the remaining MSBs apply to all member shapes
        if (params_in_shared_mem)
            {
            // load only if we are sure that we are not touching any unified memory
            for (unsigned int i = 0; i < mparams.size(); ++i)
                {
                mparams[i].load_shared(ptr, available_bytes, mask);
                }
            }
        }

    //!< Returns the number of bits available for tuning
    HOSTDEVICE static inline unsigned int getTuningBits()
        {
        return GPUTree::getTuningBits() + 4 + mparam_type::getTuningBits();
        }

    #ifdef ENABLE_HIP
    //! Set CUDA memory hints
    void set_memory_hint() const
        {
        tree.set_memory_hint();

        mpos.set_memory_hint();
        morientation.set_memory_hint();
        mparams.set_memory_hint();
        moverlap.set_memory_hint();

        // attach member parameters
        for (unsigned int i = 0; i < mparams.size(); ++i)
            mparams[i].set_memory_hint();
        }
    #endif

    #ifndef __HIPCC__
    //! Shape constructor
    union_params(unsigned int _N, bool _managed)
        : N(_N)
        {
        mpos = ManagedArray<Scalar3>(N,_managed);
        morientation = ManagedArray<Scalar4>(N,_managed);
        mparams = ManagedArray<mparam_type>(N,_managed);
        moverlap = ManagedArray<unsigned int>(N,_managed);
        }
    #endif

    gpu_tree_type tree;                      //!< OBB tree for constituent shapes
    ManagedArray<Scalar3> mpos;         //!< Position vectors of member shapes
    ManagedArray<Scalar4> morientation; //!< Orientation of member shapes
    ManagedArray<mparam_type> mparams;        //!< Parameters of member shapes
    ManagedArray<unsigned int> moverlap;      //!< only check overlaps for which moverlap[i] & moverlap[j]
    OverlapReal diameter;                    //!< Precalculated overall circumsphere diameter
    unsigned int N;                           //!< Number of member shapes
    unsigned int ignore;                     //!<  Bitwise ignore flag for stats. 1 will ignore, 0 will not ignore
    } __attribute__((aligned(32)));

} // end namespace detail

//! Shape consisting of union of shapes of a single type but individual parameters
/*!
    The parameter defining a ShapeUnion is a structure implementing the HPMC shape interface and containing
    parameter objects for its member particles in its own parameters structure

    The purpose of ShapeUnion is to allow an overlap check to iterate through pairs of member shapes between
    two composite particles. The two particles overlap if any of their member shapes overlap.

    ShapeUnion stores an internal OBB tree for fast overlap checks.
*/
template<class Shape>
struct ShapeUnion
    {
    //! Define the parameter type
    typedef typename detail::union_params<Shape> param_type;

    //! Temporary storage for depletant insertion
    typedef struct detail::union_depletion_storage depletion_storage_type;

    //! Initialize a sphere_union
    DEVICE inline ShapeUnion(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), members(_params)
        {
        }

    //! Does this shape have an orientation
    DEVICE inline bool hasOrientation() const
        {
        if (members.N == 1)
            {
            // if we have only one member in the center, return that shape's anisotropy flag
            auto p = members.mpos[0];
            if (p.x == Scalar(0.0) && p.y == p.x && p.z == p.x)
                {
                Shape s(quat<Scalar>(), members.mparams[0]);
                return s.hasOrientation();
                }
            }

        return true;
        }

    //!Ignore flag for acceptance statistics
    DEVICE inline bool ignoreStatistics() const { return members.ignore; }

    //! Get the circumsphere diameter
    DEVICE inline OverlapReal getCircumsphereDiameter() const
        {
        // return the precomputed diameter
        return members.diameter;
        }

    //! Get the in-sphere radius
    DEVICE OverlapReal getInsphereRadius() const
        {
        // not implemented
        return OverlapReal(0.0);
        }

    //! Return the bounding box of the shape in world coordinates
    DEVICE inline detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        return getOBB(pos).getAABB();
        }

    //! Return a tight fitting OBB
    DEVICE inline detail::OBB getOBB(const vec3<Scalar>& pos) const
        {
        if (members.N > 0)
            {
            // get the root node OBB from the tree
            detail::OBB obb = members.tree.getOBB(0);

            // transform it into world-space
            obb.affineTransform(orientation, pos);

            return obb;
            }
        else
            {
            return detail::OBB(pos, OverlapReal(0.5)*members.diameter);
            }
        }

    //! Returns true if this shape splits the overlap check over several threads of a warp using threadIdx.x
    HOSTDEVICE static bool isParallel() {
        return true;
        }

    //! Returns the number of tuning bits for the GPU kernels
    HOSTDEVICE static inline unsigned int getTuningBits()
        {
        return param_type::getTuningBits();
        }

    quat<Scalar> orientation;    //!< Orientation of the particle

    const param_type& members;     //!< member data
    };

template<class Shape>
DEVICE inline bool test_narrow_phase_overlap(vec3<Scalar> dr,
                                             const ShapeUnion<Shape>& a,
                                             const ShapeUnion<Shape>& b,
                                             unsigned int cur_node_a,
                                             unsigned int cur_node_b,
                                             unsigned int &err)
    {
    //! Param type of the member shapes
    typedef typename Shape::param_type mparam_type;

    vec3<Scalar> r_ab = rotate(conj(b.orientation),dr);

    // loop through leaf particles of cur_node_a
    // parallel loop over N^2 interacting particle pairs
    unsigned int ptl_i = a.members.tree.getLeafNodePtrByNode(cur_node_a);
    unsigned int ptl_j = b.members.tree.getLeafNodePtrByNode(cur_node_b);

    unsigned int ptls_i_end = a.members.tree.getLeafNodePtrByNode(cur_node_a+1);
    unsigned int ptls_j_end = b.members.tree.getLeafNodePtrByNode(cur_node_b+1);

    // get starting offset for this thread
    unsigned int na = ptls_i_end - ptl_i;
    unsigned int nb = ptls_j_end - ptl_j;

    unsigned int len = na*nb;

    #if defined (__HIP_DEVICE_COMPILE__)
    unsigned int offset = threadIdx.x;
    unsigned int incr = blockDim.x;
    #else
    unsigned int offset = 0;
    unsigned int incr = 1;
    #endif

    // iterate over (a,b) pairs in row major
    for (unsigned int n = 0; n < len; n += incr)
        {
        if (n + offset < len)
            {
            unsigned int ishape = a.members.tree.getParticleByIndex(ptl_i+(n+offset)/nb);
            unsigned int jshape = b.members.tree.getParticleByIndex(ptl_j+(n+offset)%nb);

            const mparam_type& params_i = a.members.mparams[ishape];
            Shape shape_i(quat<Scalar>(), params_i);
            if (shape_i.hasOrientation())
                {
                auto q = a.members.morientation[ishape];
                shape_i.orientation = conj(b.orientation)*a.orientation * quat<Scalar>(q.w,vec3<Scalar>(q.x,q.y,q.z));
                }

            auto p = a.members.mpos[ishape];
            vec3<Scalar> pos_i(rotate(conj(b.orientation)*a.orientation,vec3<Scalar>(p.x,p.y,p.z))-r_ab);
            unsigned int overlap_i = a.members.moverlap[ishape];

            const mparam_type& params_j = b.members.mparams[jshape];
            Shape shape_j(quat<Scalar>(), params_j);
            if (shape_j.hasOrientation())
                {
                auto q  = b.members.morientation[jshape];
                shape_j.orientation = quat<Scalar>(q.w, vec3<Scalar>(q.x,q.y,q.z));
                }

            unsigned int overlap_j = b.members.moverlap[jshape];

            if (overlap_i & overlap_j)
                {
                auto p  = b.members.mpos[jshape];
                vec3<Scalar> r_ij = vec3<Scalar>(p.x,p.y,p.z) - pos_i;
                if (test_overlap(r_ij, shape_i, shape_j, err))
                    {
                    return true;
                    }
                }
            }
        }

    return false;
    }

template <class Shape >
DEVICE inline bool test_overlap(const vec3<Scalar>& r_ab,
                                const ShapeUnion<Shape>& a,
                                const ShapeUnion<Shape>& b,
                                unsigned int& err)
    {
    const detail::GPUTree& tree_a = a.members.tree;
    const detail::GPUTree& tree_b = b.members.tree;

    // perform a tandem tree traversal
    unsigned long int stack = 0;
    unsigned int cur_node_a = 0;
    unsigned int cur_node_b = 0;

    vec3<OverlapReal> dr_rot(rotate(conj(b.orientation),-r_ab));
    quat<OverlapReal> q(conj(b.orientation)*a.orientation);

    detail::OBB obb_a = tree_a.getOBB(cur_node_a);
    obb_a.affineTransform(q, dr_rot);

    detail::OBB obb_b = tree_b.getOBB(cur_node_b);

    unsigned int query_node_a = UINT_MAX;
    unsigned int query_node_b = UINT_MAX;

    while (cur_node_a != tree_a.getNumNodes() && cur_node_b != tree_b.getNumNodes())
        {
        query_node_a = cur_node_a;
        query_node_b = cur_node_b;

        if (detail::traverseBinaryStack(tree_a, tree_b, cur_node_a, cur_node_b, stack, obb_a, obb_b, q, dr_rot)
            && test_narrow_phase_overlap(r_ab, a, b, query_node_a, query_node_b, err))
            return true;
        }

    return false;
    }

#ifndef __HIPCC__
template<>
inline std::string getShapeSpec(const ShapeUnion<ShapeSphere>& sphere_union)
    {
    auto& members = sphere_union.members;

    unsigned int n_centers = members.N;
    std::ostringstream shapedef;
    shapedef << "{\"type\": \"SphereUnion\", \"centers\": [";
    for (unsigned int i = 0; i < n_centers-1; i++)
        {
        shapedef << "[" << members.mpos[i].x << ", " << members.mpos[i].y << ", " << members.mpos[i].z << "], ";
        }
    shapedef << "[" << members.mpos[n_centers-1].x << ", " << members.mpos[n_centers-1].y << ", " << members.mpos[n_centers-1].z << "]], \"diameters\": [";
    for (unsigned int i = 0; i < n_centers-1; i++)
        {
        shapedef << 2.0*members.mparams[i].radius << ", ";
        }
    shapedef << 2.0*members.mparams[n_centers-1].radius;
    shapedef << "]}";

    return shapedef.str();
    }
#endif

} // end namespace hpmc

#undef DEVICE
#undef HOSTDEVICE
#endif // end __SHAPE_UNION_H__
