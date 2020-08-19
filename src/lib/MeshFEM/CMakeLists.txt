################################################################################

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
        EdgeFields.cc
        EdgeFields.hh
        ElasticityTensor.hh
        EmbeddedElement.hh
        ExpressionVector.hh
        FEMMesh.hh
        FEMMesh.inl
        Fields.hh
        Flattening.hh
        function_traits.hh
        Functions.hh
        Future.hh
        GaussQuadrature.cc
        GaussQuadrature.hh
        Geometry.hh
        GlobalBenchmark.cc
        GlobalBenchmark.hh
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
        NTuple.hh
        OneForm.hh
        OrthotropicHomogenization.hh
        Parallelism.hh
        Parallelism.cc
        ParallelAssembly.hh
        PeriodicBoundaryMatcher.hh
        PeriodicHomogenization.hh
        PerturbMesh.hh
        Poisson.hh
        Simplex.hh
        SimplicialMesh.hh
        SimplicialMeshInterface.hh
        SparseMatrices.hh
        StringUtils.cc
        StringUtils.hh
        SymmetricMatrix.hh
        SymmetricMatrixInterpolant.hh
        TemplateHacks.hh
        TensorProjection.hh
        TetMesh.hh
        TetMesh.inl
        Timer.hh
        Triangulate.h
        TriMesh.hh
        TriMesh.inl
        Types.cc
        Types.hh
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
        Utilities/NameMangling.hh
        wrappers/meshfem_triangle.h
)

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC ..)

# Use C++11
target_compile_features(${PROJECT_NAME} PUBLIC ${CXX11_FEATURES})

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        Threads::Threads
        Eigen3::Eigen
        json::json
        tbb::tbb
        optional::optional
        triangle::triangle
        tinyexpr::tinyexpr
        umfpack::umfpack
        cholmod::cholmod
        meshfem::boost
#		CGAL::CGAL
    PRIVATE
        warnings::all
)

target_compile_definitions(${PROJECT_NAME} PUBLIC -DMESHFEM_WITH_TBB -DNOMINMAX -D_ENABLE_EXTENDED_ALIGNED_STORAGE -D_USE_MATH_DEFINES)
target_compile_options(${PROJECT_NAME} PUBLIC -fvisibility=hidden)

# Generate MeshFEM_export.h which provides the MESHFEM_EXPORT macro
# for making our public API symbols visible.
include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME} EXPORT_FILE_NAME ${CMAKE_BINARY_DIR}/exports/${PROJECT_NAME}_export.h)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_BINARY_DIR}/exports)

################################################################################
# Configure manual compiler flags based on CMake options
################################################################################

if(MESHFEM_DISABLE_CXX11_ABI_GCC)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -D_GLIBCXX_USE_CXX11_ABI=0)
endif()

if(MESHFEM_ENABLE_BENCHMARKING)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DBENCHMARK)
endif()
