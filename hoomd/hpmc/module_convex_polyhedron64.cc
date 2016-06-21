// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"
#include "ComputeFreeVolume.h"

#include "ShapeSphere.h"
#include "ShapeConvexPolygon.h"
#include "ShapePolyhedron.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeSpheropolyhedron.h"
#include "ShapeSpheropolygon.h"
#include "ShapeSimplePolygon.h"
#include "ShapeEllipsoid.h"
#include "ShapeFacetedSphere.h"
#include "ShapeSphinx.h"
#include "AnalyzerSDF.h"
#include "ShapeUnion.h"

#include "ExternalField.h"
#include "ExternalFieldWall.h"
#include "ExternalFieldLattice.h"
#include "ExternalFieldComposite.h"

#include "UpdaterExternalFieldWall.h"
#include "UpdaterRemoveDrift.h"
#include "UpdaterMuVT.h"
#include "UpdaterMuVTImplicit.h"

#ifdef ENABLE_CUDA
#include "IntegratorHPMCMonoGPU.h"
#include "IntegratorHPMCMonoImplicitGPU.h"
#include "ComputeFreeVolumeGPU.h"
#endif

// Include boost.python to do the exporting
#include <boost/python.hpp>

using namespace boost::python;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
{

//! Export the base HPMCMono integrators
void export_convex_polyhedron64()
    {
    export_IntegratorHPMCMono< ShapeConvexPolyhedron<64> >("IntegratorHPMCMonoConvexPolyhedron64");
    export_IntegratorHPMCMonoImplicit< ShapeConvexPolyhedron<64> >("IntegratorHPMCMonoImplicitConvexPolyhedron64");
    export_ComputeFreeVolume< ShapeConvexPolyhedron<64> >("ComputeFreeVolumeConvexPolyhedron64");
    export_AnalyzerSDF< ShapeConvexPolyhedron<64> >("AnalyzerSDFConvexPolyhedron64");
    export_UpdaterMuVT< ShapeConvexPolyhedron<64> >("UpdaterMuVTConvexPolyhedron64");
    export_UpdaterMuVTImplicit< ShapeConvexPolyhedron<64> >("UpdaterMuVTImplicitConvexPolyhedron64");

    export_ExternalFieldInterface<ShapeConvexPolyhedron<64> >("ExternalFieldConvexPolyhedron64");
    export_LatticeField<ShapeConvexPolyhedron<64> >("ExternalFieldLatticeConvexPolyhedron64");
    export_ExternalFieldComposite<ShapeConvexPolyhedron<64> >("ExternalFieldCompositeConvexPolyhedron64");
    export_RemoveDriftUpdater<ShapeConvexPolyhedron<64> >("RemoveDriftUpdaterConvexPolyhedron64");
    export_ExternalFieldWall<ShapeConvexPolyhedron<64> >("WallConvexPolyhedron64");
    export_UpdaterExternalFieldWall<ShapeConvexPolyhedron<64> >("UpdaterExternalFieldWallConvexPolyhedron64");

    #ifdef ENABLE_CUDA
    #ifdef ENABLE_SPHINX_GPU

    export_IntegratorHPMCMonoGPU< ShapeConvexPolyhedron<64> >("IntegratorHPMCMonoGPUConvexPolyhedron64");
    export_IntegratorHPMCMonoImplicitGPU< ShapeConvexPolyhedron<64> >("IntegratorHPMCMonoImplicitGPUConvexPolyhedron64");
    export_ComputeFreeVolumeGPU< ShapeConvexPolyhedron<64> >("ComputeFreeVolumeGPUConvexPolyhedron64");

    #endif
    #endif
    }

}
