################################################################################

include(MeshFEMDependencies)

add_library(MeshFEM SHARED
        Algebra.hh
        BaseCellType.hh
        BoundaryConditions.cc
        BoundaryConditions.hh
        BoundaryLaplacian.hh
        BoundaryMesh.hh
        CollisionGrid.hh
        ComponentMask.hh
        Concepts.hh
        DenseCollisionGrid.hh
        DynamicSimulator.hh
        EdgeFields.cc
        EdgeFields.hh
        ElasticityTensor.hh
        EmbeddedElement.hh
        ExpressionVector.hh
        Eigensolver.cc
        Eigensolver.hh
        EquilibriumSolver.hh
        FEMMesh.hh
        FEMMesh.inl
        Fields.hh
        Flattening.hh
        Functions.hh
        FieldSampler.hh
        GaussQuadrature.hh
        Geometry.hh
        GridFunction.hh
        InterpolantRestriction.hh
        JSFieldWriter.hh
        Laplacian.hh
        LinearElasticity.hh
        LinearIndexer.hh
        MassMatrix.hh
        MaterialField.hh
        MaterialOptimization.hh
        Materials.cc
        Materials.hh
        MeshDataTraits.hh
        Meshing.hh
        MeshIO.cc
        MeshIO.hh
        MSHFieldParser.cc
        MSHFieldParser.hh
        MSHFieldWriter.hh
        OneForm.hh
        OrthotropicHomogenization.hh
        Parametrization.cc
        Parametrization.hh
        PeriodicBoundaryMatcher.hh
        PeriodicHomogenization.hh
        PerturbMesh.hh
        Poisson.hh
        Simplex.hh
        SimplicialMesh.hh
        SimplicialMeshInterface.hh
        StringUtils.cc
        StringUtils.hh
        SymmetricMatrix.hh
        SymmetricMatrixInterpolant.hh
        TensorProjection.hh
        TetMesh.hh
        TetMesh.inl
        Triangulate.h
        TriMesh.hh
        TriMesh.inl
        UniformLaplacian.hh
        util.h
        utils.hh
        VonMises.hh

        algorithms/get_element_components.hh
        algorithms/remove_if_index.hh
        filters/CurveCleanup.hh
        filters/extract_hole_boundaries.hh
        filters/extract_polygons.hh
        filters/extrude.hh
        filters/gen_cursor.hh
        filters/gen_grid.hh
        filters/hex_tet_subdiv.hh
        filters/highlight_dangling_vertices.hh
        filters/merge_duplicate_vertices.hh
        filters/quad_subdiv.hh
        filters/quad_subdiv_high_aspect.hh
        filters/quad_tri_subdiv.hh
        filters/quad_tri_subdiv_asymmetric.hh
        filters/reflect.hh
        filters/remove_dangling_vertices.hh
        filters/remove_small_components.hh
        filters/reorient_negative_elements.hh
        filters/ResampleCurve.hh
        filters/subdivide.hh
        filters/voxels_to_simplices.hh
        Handles/FEMMeshHandles.hh
        Handles/Handle.hh
        Handles/TetMeshHandles.hh
        Handles/TriMeshHandles.hh
        Utilities/apply.hh
        Utilities/ci_string.hh
        Utilities/EdgeAccessAdaptor.hh
        Utilities/EdgeSoupAdaptor.hh
        Utilities/IteratorMap.hh
        Utilities/NDArray.hh
        Utilities/RandomAccessIndexSet.hh
        EnergyDensities/Tensor.hh
        Utilities/NameMangling.hh
        wrappers/meshfem_triangle.h

        newton_optimizer/newton_optimizer.cc
        newton_optimizer/newton_optimizer.hh
        newton_optimizer/ConvergenceReport.hh
        newton_optimizer/MultiobjectiveProblem.cc
        newton_optimizer/NewtonHessianFactorization.cc

        EnergyDensities/LinearElasticEnergy.hh
        EnergyDensities/NeoHookeanEnergy.hh

        ElasticSolid.hh
        ElasticSheet.hh

        IPCIntegration/IPCObjectiveTerm.hh
        IPCIntegration/IPCObjectiveTerm.cc
        IPCIntegration/IPCWrapper.cc

        ClosestPointProjection.cc
)

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC ..)

# Use C++17
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        TBB::tbb
        Threads::Threads
        Eigen3::Eigen
        MeshFEMCore
        MeshFEMSparse
        optional::optional
        triangle::triangle
        tinyexpr::tinyexpr
        meshfem::boost
        meshfem::spectra
        rotation_optimization
        nlohmann_json::nlohmann_json
#		CGAL::CGAL
    PRIVATE
        warnings::all
)

target_compile_definitions(${PROJECT_NAME} PUBLIC -DMESHFEM_WITH_TBB -DNOMINMAX -D_ENABLE_EXTENDED_ALIGNED_STORAGE -D_USE_MATH_DEFINES)
meshfem_apply_abi_flags(${PROJECT_NAME})
# Generate MeshFEM_export.h which provides the MESHFEM_EXPORT macro
# for making our public API symbols visible.
meshfem_generate_export_header(${PROJECT_NAME} EXPORT_FILE_NAME ${CMAKE_BINARY_DIR}/exports/${PROJECT_NAME}_export.h)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_BINARY_DIR}/exports)

if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)

    if(CMAKE_VERSION VERSION_LESS 3.21)
        message(FATAL_ERROR
            "Building MeshFEM on Windows requires CMake >= 3.21 (found ${CMAKE_VERSION}) "
            "for $<TARGET_RUNTIME_DLLS>. Without this deployment step the python modules "
            "cannot import. POSIX builds are unaffected at the project minimum of 3.14.")
    endif()
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>"
                "${PROJECT_SOURCE_DIR}/python"
        COMMAND_EXPAND_LISTS
        COMMENT "Copying runtime DLLs next to the MeshFEM python modules")
endif()

if(MSVC)
    # /we4273: inconsistent dll linkage means a symbol was silently dropped from the
    # export table, which is the defect this whole arrangement exists to prevent.
    target_compile_options(${PROJECT_NAME} PRIVATE /we4273)
    foreach (meshfem_absorbed_target MeshFEMCore MeshFEMSparse)
        if (TARGET ${meshfem_absorbed_target})
            target_compile_definitions(${meshfem_absorbed_target} PRIVATE ${PROJECT_NAME}_EXPORTS)
            target_compile_options(${meshfem_absorbed_target} PRIVATE /we4273)
        endif()
    endforeach()
endif()

################################################################################
# Configure manual compiler flags based on CMake options
################################################################################
if(MESHFEM_WITH_IPC_TOOLKIT)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DMESHFEM_WITH_IPC_TOOLKIT)
    target_link_libraries(${PROJECT_NAME} PUBLIC ipc::toolkit)
endif()

if (MESHFEM_WITH_TINYAD)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DMESHFEM_WITH_TINYAD)
    target_link_libraries(${PROJECT_NAME} PUBLIC TinyAD)
endif()
