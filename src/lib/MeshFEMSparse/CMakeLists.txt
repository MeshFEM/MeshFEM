if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(MESHFEMSPARSE_TOPLEVEL_PROJECT ON)
else()
    set(MESHFEMSPARSE_TOPLEVEL_PROJECT OFF)
endif()

option(MESHFEMSPARSE_BUILD_TESTS "Build MeshFEMSparse tests" ${MESHFEMSPARSE_TOPLEVEL_PROJECT})

include(MeshFEMSparseDependencies)

add_library(MeshFEMSparse STATIC
    BlockCSCHessian.cc
    BorderedSparseHessian.cc
    Solvers/make_cholesky_factorizer.cc
    Solvers/CatamariFactorizer.cc
    Solvers/MatrixRecorder.cc
    Solvers/PardisoFactorizer.cc
    Solvers/pardiso_ordering.cc
    Solvers/AccelerateFactorizer.cc
    Solvers/accelerate_ordering.cc
)

if (MESHFEM_USE_LEGACY_CATAMARI)
    get_target_property(modified_sources MeshFEMSparse SOURCES)
    list(REMOVE_ITEM modified_sources Solvers/CatamariFactorizer.cc)
    list(APPEND modified_sources Solvers/CatamariFactorizerLegacy.cc)
    set_target_properties(MeshFEMSparse PROPERTIES SOURCES "${modified_sources}")
endif()

target_include_directories(MeshFEMSparse PUBLIC .. ${CMAKE_BINARY_DIR}/exports)
target_compile_features(MeshFEMSparse PUBLIC cxx_std_17)

target_link_libraries(MeshFEMSparse
    PUBLIC
        MeshFEMCore
        Eigen3::Eigen
        TBB::tbb
        Threads::Threads
    PRIVATE
        warnings::all
)

target_compile_options(MeshFEMSparse PUBLIC -fvisibility=hidden)

if(MESHFEM_WITH_UMFPACK)
    target_compile_definitions(MeshFEMSparse PUBLIC -DMESHFEM_WITH_UMFPACK)
    target_link_libraries(MeshFEMSparse PUBLIC umfpack::umfpack)
endif()

if(MESHFEM_WITH_CHOLMOD)
    target_compile_definitions(MeshFEMSparse PUBLIC -DMESHFEM_WITH_CHOLMOD)
    target_link_libraries(MeshFEMSparse PUBLIC cholmod::cholmod)
endif()

if (MESHFEM_WITH_CATAMARI)
    target_compile_definitions(MeshFEMSparse PUBLIC -DMESHFEM_WITH_CATAMARI)
    target_link_libraries(MeshFEMSparse PUBLIC catamari)
    if (MESHFEM_USE_LEGACY_CATAMARI)
        target_compile_definitions(MeshFEMSparse PUBLIC -DMESHFEM_USE_LEGACY_CATAMARI)
    endif()
endif()

if (MESHFEM_WITH_SCOTCH)
    target_compile_definitions(MeshFEMSparse PUBLIC -DMESHFEM_WITH_SCOTCH)
    target_link_libraries(MeshFEMSparse PUBLIC SCOTCH::scotch SCOTCH::scotcherrexit)
endif()

if (MESHFEM_WITH_MKL_PARDISO)
    target_compile_definitions(MeshFEMSparse PUBLIC -DMESHFEM_WITH_MKL_PARDISO)
endif()

if (MESHFEM_WITH_PARDISO)
    set(MKLROOT /opt/intel/mkl)
    target_compile_definitions(MeshFEMSparse PUBLIC -DMESHFEM_WITH_PARDISO)
    target_link_libraries(MeshFEMSparse PUBLIC
        /home/jpanetta/software/pardiso/libpardiso600-GNU800-X86-64.so
         -L${MKLROOT}/lib/intel64 -lmkl_rt -Wl,--no-as-needed -lpthread -lm -ldl -lgomp
        -lgfortran)
endif()

if (MESHFEMSPARSE_BUILD_TESTS)
    include(CTest)
    if (BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()
