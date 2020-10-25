// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "ComputeFreeVolume.h"
#include "UpdaterClusters.h"
#include "UpdaterMuVT.h"
#include "ExternalCallback.h"
#include "ExternalFieldLattice.h"
#include "ExternalFieldComposite.h"
#include "ExternalFieldWall.h"
#include "UpdaterExternalFieldWall.h"
#include "UpdaterRemoveDrift.h"

#include "AnalyzerSDF.h"
#include "UpdaterBoxMC.h"
#include "UpdaterClusters.h"

#include "Shapes.h"

#include "ShapeProxy.h"

#include "GPUTree.h"

#ifdef ENABLE_HIP
#include "IntegratorHPMCMonoGPU.h"
#include "ComputeFreeVolumeGPU.h"
#include "UpdaterClustersGPU.h"
#endif

#include "modules.h"

/*! \file module.cc
    \brief Export classes to python
*/
using namespace hpmc;
using namespace std;
namespace py = pybind11;

namespace hpmc
{

//! HPMC implementation details
/*! The detail namespace contains classes and functions that are not part of the HPMC public interface. These are
    subject to change without notice and are designed solely for internal use within HPMC.
*/
namespace detail
{

// could move the make_param functions back??

}; // end namespace detail

}; // end namespace hpmc

using namespace hpmc::detail;

template<class Shape>
void export_modules(pybind11::module& m, const std::string& name)
    {

    export_IntegratorHPMCMono< Shape >(m, "IntegratorHPMCMono"+name);
    export_ComputeFreeVolume< Shape >(m, "ComputeFreeVolume"+name);
    export_AnalyzerSDF< Shape >(m, "AnalyzerSDF"+name);
    export_UpdaterMuVT< Shape >(m, "UpdaterMuVT"+name);
    export_UpdaterClusters< Shape >(m, "UpdaterClusters"+name);

    export_ExternalFieldInterface<Shape>(m, "ExternalField"+name);
    export_LatticeField<Shape>(m, "ExternalFieldLattice"+name);
    export_ExternalFieldComposite<Shape>(m, "ExternalFieldComposite"+name);
    export_RemoveDriftUpdater<Shape>(m, "RemoveDriftUpdater"+name);
    export_ExternalFieldWall<Shape>(m, "Wall"+name);
    export_UpdaterExternalFieldWall<Shape>(m, "UpdaterExternalFieldWall"+name);
    export_ExternalCallback<Shape>(m, "ExternalCallback"+name);
    }

#ifdef ENABLE_HIP
template<class Shape>
void export_gpu_modules(pybind11::module& m, const std::string& name)
    {
    export_IntegratorHPMCMonoGPU< Shape >(m, "IntegratorHPMCMonoGPU"+name);
    export_ComputeFreeVolumeGPU< Shape >(m, "ComputeFreeVolumeGPU"+name);
    export_UpdaterClustersGPU< Shape >(m, "UpdaterClustersGPU"+name);
    }
#endif

//! Define the _hpmc python module exports
PYBIND11_MODULE(_hpmc, m)
    {
    export_IntegratorHPMC(m);

    export_UpdaterBoxMC(m);
    export_external_fields(m);
    export_shape_params(m);

    export_modules<ShapeSphere>(m, "Sphere");
    export_modules<ShapeConvexPolygon>(m, "ConvexPolygon");
    export_modules<ShapeSimplePolygon>(m, "SimplePolygon");
    export_modules<ShapeSpheropolygon>(m, "Spheropolygon");
    export_modules<ShapePolyhedron>(m, "Polyhedron");
    export_modules<ShapeEllipsoid>(m, "Ellipsoid");
    export_modules<ShapeFacetedEllipsoid>(m, "FacetedEllipsoid");
    export_modules<ShapeSphinx>(m, "Sphinx");
    export_modules<ShapeUnion<ShapeConvexPolyhedron> >(m, "ConvexPolyhedronUnion");
    export_modules<ShapeUnion<ShapeFacetedEllipsoid> >(m, "FacetedEllipsoidUnion");
    export_modules<ShapeUnion<ShapeSphere> >(m, "SphereUnion");
    export_modules<ShapeConvexPolyhedron>(m, "ConvexPolyhedron");
    export_modules<ShapeSpheropolyhedron>(m, "Spheropolyhdron");

    #ifdef ENABLE_HIP
    export_gpu_modules<ShapeSphere>(m, "Sphere");
    export_gpu_modules<ShapeConvexPolygon>(m, "ConvexPolygon");
    export_gpu_modules<ShapeSimplePolygon>(m, "SimplePolygon");
    export_gpu_modules<ShapeSpheropolygon>(m, "Spheropolygon");
    export_gpu_modules<ShapePolyhedron>(m, "Polyhedron");
    export_gpu_modules<ShapeEllipsoid>(m, "Ellipsoid");
    export_gpu_modules<ShapeFacetedEllipsoid>(m, "FacetedEllipsoid");
    #ifdef ENABLE_HPMC_SPHINX_GPU
    export_gpu_modules<ShapeSphinx>(m, "Sphinx");
    #endif
    export_gpu_modules<ShapeUnion<ShapeConvexPolyhedron> >(m, "ConvexPolyhedronUnion");
    export_gpu_modules<ShapeUnion<ShapeFacetedEllipsoid> >(m, "FacetedEllipsoidUnion");
    export_gpu_modules<ShapeUnion<ShapeSphere> >(m, "SphereUnion");
    export_gpu_modules<ShapeConvexPolyhedron>(m, "ConvexPolyhedron");
    export_gpu_modules<ShapeSpheropolyhedron>(m, "Spheropolyhdron");
    #endif

    py::class_<sph_params, std::shared_ptr<sph_params> >(m, "sph_params");
    py::class_<ell_params, std::shared_ptr<ell_params> >(m, "ell_params");
    py::class_<poly2d_verts, std::shared_ptr<poly2d_verts> >(m, "poly2d_verts");
    py::class_<poly3d_data, std::shared_ptr<poly3d_data> >(m, "poly3d_data");
    py::class_< poly3d_verts, std::shared_ptr< poly3d_verts > >(m, "poly3d_verts");
    py::class_<faceted_ellipsoid_params, std::shared_ptr<faceted_ellipsoid_params> >(m, "faceted_ellipsoid_params");
    py::class_<sphinx3d_params, std::shared_ptr<sphinx3d_params> >(m, "sphinx3d_params")
        .def_readwrite("circumsphereDiameter",&sphinx3d_params::circumsphereDiameter);
    py::class_< ShapeUnion<ShapeSphere>::param_type, std::shared_ptr< ShapeUnion<ShapeSphere>::param_type> >(m, "msph_params");

    py::class_< ShapeUnion<ShapeSpheropolyhedron>::param_type, std::shared_ptr< ShapeUnion<ShapeSpheropolyhedron>::param_type> >(m, "mpoly3d_params");
    py::class_< ShapeUnion<ShapeFacetedEllipsoid>::param_type, std::shared_ptr< ShapeUnion<ShapeFacetedEllipsoid>::param_type> >(m, "mfellipsoid_params");

    m.def("make_poly2d_verts", &make_poly2d_verts);
    m.def("make_poly3d_data", &make_poly3d_data);
    m.def("make_poly3d_verts", &make_poly3d_verts);
    m.def("make_ell_params", &make_ell_params);
    m.def("make_sph_params", &make_sph_params);
    m.def("make_faceted_ellipsoid", &make_faceted_ellipsoid);
    m.def("make_sphinx3d_params", &make_sphinx3d_params);
    m.def("make_convex_polyhedron_union_params", &make_union_params<ShapeSpheropolyhedron>);
    m.def("make_faceted_ellipsoid_union_params", &make_union_params<ShapeFacetedEllipsoid>);
    m.def("make_sphere_union_params", &make_union_params<ShapeSphere>);
    m.def("make_overlapreal3", &make_overlapreal3);
    m.def("make_overlapreal4", &make_overlapreal4);

    // export counters
    export_hpmc_implicit_counters(m);

    export_hpmc_clusters_counters(m);
    }

/*! \defgroup hpmc_integrators HPMC integrators
*/

/*! \defgroup hpmc_analyzers HPMC analyzers
*/

/*! \defgroup shape Shapes
    Shape classes define the geometry of shapes and associated overlap checks
*/

/*! \defgroup vecmath Vector Math
    Vector, matrix, and quaternion math routines
*/

/*! \defgroup hpmc_detail Details
    HPMC implementation details
    @{
*/

/*! \defgroup hpmc_data_structs Data structures
    HPMC internal data structures
*/

/*! \defgroup hpmc_kernels HPMC kernels
    HPMC GPU kernels
*/

/*! \defgroup minkowski Minkowski methods
    Classes and functions related to Minkowski overlap detection methods
*/

/*! \defgroup overlap Other overlap methods
    Classes and functions related to other (brute force) overlap methods
*/

/*! @} */
