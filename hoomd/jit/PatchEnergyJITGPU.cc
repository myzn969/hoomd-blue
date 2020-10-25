// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifdef ENABLE_HIP

#include "hoomd/hpmc/IntegratorHPMC.h"
#include "PatchEnergyJITGPU.h"

void export_PatchEnergyJITGPU(pybind11::module &m)
    {
    pybind11::class_<PatchEnergyJITGPU, PatchEnergyJIT,
                     std::shared_ptr<PatchEnergyJITGPU> >(m, "PatchEnergyJITGPU")
            .def(pybind11::init< std::shared_ptr<ExecutionConfiguration>,
                                 const std::string&, Scalar,
                                 const unsigned int,
                                 const std::string&,
                                 const std::vector<std::string>&
                                >())
            ;
    }
#endif
