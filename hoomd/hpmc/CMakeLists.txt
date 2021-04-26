# Maintainer: joaander

set(_hpmc_shapes ShapeSphere
                 ShapeConvexPolygon
                 ShapeSimplePolygon
                 ShapeSpheropolygon
                 ShapePolyhedron
                 ShapeEllipsoid
                 ShapeFacetedEllipsoid
                 ShapeConvexPolyhedron
                 ShapeSpheropolyhedron
                 ShapeSphinx
                 )

set(_hpmc_gpu_shapes ${_hpmc_shapes})
option(ENABLE_HPMC_SPHINX_GPU "Enable sphinx on the GPU" OFF)
if (NOT ENABLE_HPMC_SPHINX_GPU)
    LIST(REMOVE_ITEM _hpmc_gpu_shapes ShapeSphinx)
endif()

set(_hpmc_union_shapes ShapeSphere
                       ShapeFacetedEllipsoid
                       ShapeConvexPolyhedron
   )
set(_hpmc_gpu_union_shapes ${_hpmc_union_shapes})

set(_hpmc_sources   module.cc
                    module_external_field.cc
                    UpdaterBoxMC.cc
                    IntegratorHPMC.cc
                    )

set(_hpmc_headers
    AnalyzerSDF.h
    ComputeFreeVolumeGPU.cuh
    ComputeFreeVolumeGPU.h
    ComputeFreeVolume.h
    ExternalFieldComposite.h
    ExternalField.h
    ExternalFieldLattice.h
    ExternalFieldWall.h
    GSDHPMCSchema.h
    GPUHelpers.cuh
    GPUTree.h
    HPMCCounters.h
    HPMCMiscFunctions.h
    HPMCPrecisionSetup.h
    IntegratorHPMC.h
    IntegratorHPMCMonoGPU.cuh
    IntegratorHPMCMonoGPUMoves.cuh
    IntegratorHPMCMonoGPUTypes.cuh
    IntegratorHPMCMonoGPUDepletants.cuh
    IntegratorHPMCMonoGPUDepletantsTypes.cuh
    IntegratorHPMCMonoGPUDepletantsAuxiliaryPhase1.cuh
    IntegratorHPMCMonoGPUDepletantsAuxiliaryPhase2.cuh
    IntegratorHPMCMonoGPUDepletantsAuxiliaryTypes.cuh
    IntegratorHPMCMonoGPUJIT.inc
    IntegratorHPMCMonoGPUJIT.h
    IntegratorHPMCMonoGPU.h
    IntegratorHPMCMono.h
    MinkowskiMath.h
    modules.h
    Moves.h
    OBB.h
    OBBTree.h
    ShapeConvexPolygon.h
    ShapeConvexPolyhedron.h
    ShapeEllipsoid.h
    ShapeFacetedEllipsoid.h
    ShapePolyhedron.h
    ShapeProxy.h
    ShapeSimplePolygon.h
    ShapeSphere.h
    ShapeSpheropolygon.h
    ShapeSpheropolyhedron.h
    ShapeSphinx.h
    ShapeUnion.h
    SphinxOverlap.h
    UpdaterClusters.h
    UpdaterClustersGPU.cuh
    UpdaterClustersGPUDepletants.cuh
    UpdaterExternalFieldWall.h
    UpdaterMuVT.h
    UpdaterRemoveDrift.h
    XenoCollide2D.h
    XenoCollide3D.h
    )

set(_hpmc_cu_sources IntegratorHPMCMonoGPU.cu
                     IntegratorHPMCMonoGPUDepletants.cu
                     UpdaterClustersGPU.cu
                     )

set(_hpmc_module_templates
    analyzer_sdf
    compute_free_volume
    external_callback
    external_field_composite
    external_field_interface
    external_field_lattice
    external_field_wall
    integrator_hpmc_mono
    updater_clusters
    updater_external_field_wall
    updater_muvt
    updater_remove_drift
    )

set(_hpmc_gpu_module_templates
    compute_free_volume_gpu
    integrator_hpmc_mono_gpu
    updater_clusters_gpu
    )

set(_hpmc_kernel_templates kernel_free_volume
                           kernel_gen_moves
                           kernel_narrow_phase
                           kernel_insert_depletants
                           kernel_update_pdata
                           kernel_cluster_overlaps
                           kernel_cluster_depletants
                           kernel_cluster_transform
                           kernel_depletants_auxiliary_phase1
                           kernel_depletants_auxiliary_phase2)

# expand the shape x GPU kernel matrix of template instantiations
foreach(MODULE ${_hpmc_module_templates})
    foreach(SHAPE ${_hpmc_shapes})
        set(SHAPE_INCLUDE ${SHAPE}.h)
        set(IS_UNION_SHAPE FALSE)
        set(_module_cc ${MODULE}_${SHAPE}.cc)
        configure_file(${MODULE}.cc.in ${_module_cc} @ONLY)
        set(_hpmc_sources ${_hpmc_sources} ${_module_cc})
    endforeach()

    foreach(SHAPE ${_hpmc_union_shapes})
        set(SHAPE_INCLUDE ${SHAPE}.h)
        set(_module_cc ${MODULE}_${SHAPE}_union.cc)
        set(IS_UNION_SHAPE TRUE)
        configure_file(${MODULE}.cc.in ${_module_cc} @ONLY)
        set(_hpmc_sources ${_hpmc_sources} ${_module_cc})
    endforeach()
endforeach()

if(ENABLE_HIP)
    # GPU classes
    foreach(MODULE ${_hpmc_gpu_module_templates})
        foreach(SHAPE ${_hpmc_gpu_shapes})
            set(SHAPE_INCLUDE ${SHAPE}.h)
            set(IS_UNION_SHAPE FALSE)
            set(_module_cc ${MODULE}_${SHAPE}.cc)
            configure_file(${MODULE}.cc.in ${_module_cc} @ONLY)
            set(_hpmc_sources ${_hpmc_sources} ${_module_cc})
        endforeach()

        foreach(SHAPE ${_hpmc_gpu_union_shapes})
            set(SHAPE_INCLUDE ${SHAPE}.h)
            set(_module_cc ${MODULE}_${SHAPE}_union.cc)
            set(IS_UNION_SHAPE TRUE)
            configure_file(${MODULE}.cc.in ${_module_cc} @ONLY)
            set(_hpmc_sources ${_hpmc_sources} ${_module_cc})
        endforeach()
    endforeach()

    # GPU kernels
    foreach(KERNEL ${_hpmc_kernel_templates})
        foreach(SHAPE ${_hpmc_gpu_shapes})
            set(SHAPE_INCLUDE ${SHAPE}.h)
            set(IS_UNION_SHAPE FALSE)
            set(_kernel_cu ${KERNEL}_${SHAPE}.cu)
            configure_file(${KERNEL}.cu.in ${_kernel_cu} @ONLY)
            set(_hpmc_cu_sources ${_hpmc_cu_sources} ${_kernel_cu})
        endforeach()

        foreach(SHAPE ${_hpmc_gpu_union_shapes})
            set(SHAPE_INCLUDE ${SHAPE}.h)
            set(_kernel_cu ${KERNEL}_${SHAPE}_union.cu)
            set(IS_UNION_SHAPE TRUE)
            configure_file(${KERNEL}.cu.in ${_kernel_cu} @ONLY)
            set(_hpmc_cu_sources ${_hpmc_cu_sources} ${_kernel_cu})
        endforeach()
    endforeach()
endif(ENABLE_HIP)

if (ENABLE_HIP)
set(_cuda_sources ${_hpmc_cu_sources})
endif (ENABLE_HIP)

pybind11_add_module(_hpmc SHARED ${_hpmc_sources} ${_cuda_sources} ${_hpmc_headers} NO_EXTRAS)
# alias into the HOOMD namespace so that plugins and symlinked components both work
add_library(HOOMD::_hpmc ALIAS _hpmc)
if (APPLE)
set_target_properties(_hpmc PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(_hpmc PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN/../jit;\$ORIGIN")
endif()

# enable out-of-source build
target_include_directories(_hpmc PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")

# link the library to its dependencies
target_link_libraries(_hpmc PUBLIC _hoomd)

if (ENABLE_HIP AND HIP_PLATFORM STREQUAL "nvcc")
target_link_libraries(_hpmc PUBLIC CUDA::cusparse )
endif()

if (BUILD_JIT)
target_link_libraries(_hpmc PUBLIC HOOMD::_jit)
endif()

fix_cudart_rpath(_hpmc)

# install the library
install(TARGETS _hpmc EXPORT HOOMDTargets
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/hpmc
        )

################ Python only modules
# copy python modules to the build directory to make it a working python package
MACRO(copy_file file)
    add_custom_command (
        OUTPUT ${file}
        DEPENDS ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT    "Copy hoomd/hpmc/${file}"
    )
ENDMACRO(copy_file)

set(files   analyze.py
            compute.py
            data.py
            __init__.py
            integrate.py
            update.py
            util.py
            field.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/hpmc
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_hpmc ALL DEPENDS ${files})

# install headers in installation target
install(FILES ${_hpmc_headers}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/include/hoomd/hpmc
       )

if (BUILD_TESTING)
    add_subdirectory(test-py)
    add_subdirectory(test)
endif()

if (BUILD_VALIDATION)
    add_subdirectory(validation)
endif()
