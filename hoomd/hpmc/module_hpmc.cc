// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"

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

#ifdef ENABLE_CUDA
#include "IntegratorHPMCMonoGPU.h"
#endif

/*! \file module.cc
    \brief Export classes to python
*/

// Include boost.python to do the exporting
#include <boost/python.hpp>

using namespace boost::python;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
{

//! Export the base HPMCMono integrators
void export_hpmc()
    {
    export_IntegratorHPMCMono< ShapeSpheropolyhedron<8> >("IntegratorHPMCMonoSpheropolyhedron8");
    export_IntegratorHPMCMono< ShapeSpheropolyhedron<16> >("IntegratorHPMCMonoSpheropolyhedron16");
    export_IntegratorHPMCMono< ShapeSpheropolyhedron<32> >("IntegratorHPMCMonoSpheropolyhedron32");
    export_IntegratorHPMCMono< ShapeSpheropolyhedron<64> >("IntegratorHPMCMonoSpheropolyhedron64");
    export_IntegratorHPMCMono< ShapeSpheropolyhedron<128> >("IntegratorHPMCMonoSpheropolyhedron128");

    // implicit depletants
    export_IntegratorHPMCMonoImplicit< ShapeSpheropolyhedron<8> >("IntegratorHPMCMonoImplicitSpheropolyhedron8");
    export_IntegratorHPMCMonoImplicit< ShapeSpheropolyhedron<16> >("IntegratorHPMCMonoImplicitSpheropolyhedron16");
    export_IntegratorHPMCMonoImplicit< ShapeSpheropolyhedron<32> >("IntegratorHPMCMonoImplicitSpheropolyhedron32");
    export_IntegratorHPMCMonoImplicit< ShapeSpheropolyhedron<64> >("IntegratorHPMCMonoImplicitSpheropolyhedron64");
    export_IntegratorHPMCMonoImplicit< ShapeSpheropolyhedron<128> >("IntegratorHPMCMonoImplicitSpheropolyhedron128");
    }

}
