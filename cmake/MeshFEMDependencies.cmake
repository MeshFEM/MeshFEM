# Prepare dependencies
#
# For each third-party library, if the appropriate target doesn't exist yet,
# download it via external project, and add_subdirectory to build it alongside
# this project.

### Configuration
set(MESHFEM_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
set(MESHFEM_EXTERNAL "${MESHFEM_ROOT}/3rdparty")

# Make MESHFEM_EXTERNAL path available also to parent projects.
get_directory_property(hasParent PARENT_DIRECTORY)
if (hasParent)
    set(MESHFEM_EXTERNAL "${MESHFEM_EXTERNAL}" PARENT_SCOPE)
endif()

# Download and update 3rdparty libraries
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)
include(MeshFEMDownloadExternal)

################################################################################
# Required libraries
################################################################################

# C++11 threads
find_package(Threads REQUIRED) # provides Threads::Threads

# Boost library
if(NOT TARGET meshfem::boost)
    include(boost)
    add_library(meshfem_boost INTERFACE)
    add_library(meshfem::boost ALIAS meshfem_boost)
    target_link_libraries(meshfem_boost INTERFACE
        Boost::filesystem
        Boost::system
        Boost::program_options
    )
endif()

# Catch2
if(NOT TARGET Catch2::Catch2 AND (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR))
    meshfem_download_catch()
    add_subdirectory(${MESHFEM_EXTERNAL}/Catch2)
    list(APPEND CMAKE_MODULE_PATH ${MESHFEM_EXTERNAL}/Catch2/contrib)
endif()

# Eigen3 library
if(NOT TARGET Eigen3::Eigen)
    add_library(meshfem_eigen INTERFACE)
    meshfem_download_eigen()
    target_include_directories(meshfem_eigen SYSTEM INTERFACE ${MESHFEM_EXTERNAL}/eigen)
    add_library(Eigen3::Eigen ALIAS meshfem_eigen)
endif()

# json library
if(NOT TARGET json::json)
    add_library(meshfem_json INTERFACE)
    meshfem_download_json()
    target_include_directories(meshfem_json SYSTEM INTERFACE ${MESHFEM_EXTERNAL}/json)
    target_include_directories(meshfem_json SYSTEM INTERFACE ${MESHFEM_EXTERNAL}/json/nlohmann)
    add_library(json::json ALIAS meshfem_json)
endif()

# Optional library
if(NOT TARGET optional::optional)
    meshfem_download_optional()
    add_library(optional_lite INTERFACE)
    target_include_directories(optional_lite SYSTEM INTERFACE ${MESHFEM_EXTERNAL}/optional/include)
    add_library(optional::optional ALIAS optional_lite)
endif()

# TBB library
if(NOT TARGET tbb::tbb)
    set(TBB_BUILD_STATIC OFF CACHE BOOL " " FORCE)
    set(TBB_BUILD_SHARED ON CACHE BOOL " " FORCE)
    set(TBB_BUILD_TBBMALLOC ON CACHE BOOL " " FORCE) # needed for CGAL's parallel mesher
    set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL " " FORCE)
    set(TBB_BUILD_TESTS OFF CACHE BOOL " " FORCE)

    meshfem_download_tbb()
    add_subdirectory(${MESHFEM_EXTERNAL}/tbb tbb EXCLUDE_FROM_ALL)
    #set_property(TARGET tbb_static tbb_def_files PROPERTY FOLDER "dependencies")
    #set_target_properties(tbb_static PROPERTIES COMPILE_FLAGS "-Wno-implicit-fallthrough -Wno-missing-field-initializers -Wno-unused-parameter -Wno-keyword-macro")

    add_library(tbb_tbb INTERFACE)
    target_include_directories(tbb_tbb SYSTEM INTERFACE ${MESHFEM_EXTERNAL}/tbb/include)
    target_link_libraries(tbb_tbb INTERFACE tbbmalloc tbb)
    add_library(tbb::tbb ALIAS tbb_tbb)

    meshfem_target_hide_warnings(tbb_tbb)
endif()

# Triangle library
if(NOT TARGET triangle::triangle)
    meshfem_download_triangle()
    add_subdirectory(${MESHFEM_EXTERNAL}/triangle triangle)
    target_include_directories(triangle SYSTEM INTERFACE ${MESHFEM_EXTERNAL}/triangle)
    add_library(triangle::triangle ALIAS triangle)
endif()

# Spectra library
if(NOT TARGET spectra::spectra)
    meshfem_download_spectra()
    add_library(meshfem_spectra INTERFACE)
    target_include_directories(meshfem_spectra SYSTEM INTERFACE ${MESHFEM_EXTERNAL}/spectra/include)
    add_library(meshfem::spectra ALIAS meshfem_spectra)
endif()

# TinyExpr library
if(NOT TARGET tinyexpr::tinyexpr)
    meshfem_download_tinyexpr()
    add_library(meshfem_tinyexpr ${MESHFEM_EXTERNAL}/tinyexpr/tinyexpr.c)
    target_include_directories(meshfem_tinyexpr SYSTEM PUBLIC ${MESHFEM_EXTERNAL}/tinyexpr)
    add_library(tinyexpr::tinyexpr ALIAS meshfem_tinyexpr)
endif()

# Cholmod solver
find_package(CHOLMOD REQUIRED) # provides cholmod::cholmod

# UmfPack solver
find_package(UMFPACK REQUIRED) # provides umfpack::umfpack

################################################################################
# Optional libraries
################################################################################

# Ceres
if (MESHFEM_WITH_CERES AND NOT TARGET ceres::ceres)
    if (MESHFEM_PREFER_SYSTEM_CERES)
        find_package(Ceres QUIET)
         if(CERES_FOUND)
             add_library(ceres_lib INTERFACE)
             target_include_directories(ceres_lib SYSTEM INTERFACE  ${CERES_INCLUDE_DIRS})
             target_link_libraries(ceres_lib INTERFACE MeshFEM ${CERES_LIBRARIES})
             add_library(ceres::ceres ALIAS ceres_lib)
         endif()
    endif()
    if (NOT TARGET ceres::ceres)
        meshfem_download_ceres()
        option(MINIGLOG "" ON)
        set(BUILD_TESTING OFF CACHE BOOL " " FORCE)
        set(BUILD_DOCUMENTATION OFF CACHE BOOL " " FORCE)
        set(BUILD_EXAMPLES OFF CACHE BOOL " " FORCE)
        set(BUILD_BENCHMARKS OFF CACHE BOOL " " FORCE)
        get_target_property(EIGEN_INCLUDE_DIR_HINTS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
        set(EIGEN_PREFER_EXPORTED_EIGEN_CMAKE_CONFIGURATION FALSE)
        if("$ENV{CLUSTER}" STREQUAL "PRINCE")
            # Hints for SuiteSparse on Prince cluster
            set(SUITESPARSE_INCLUDE_DIR_HINTS "$ENV{SUITESPARSE_INC}")
            set(SUITESPARSE_LIBRARY_DIR_HINTS "$ENV{SUITESPARSE_LIB}")
        endif()
        add_subdirectory(${MESHFEM_EXTERNAL}/ceres)
        add_library(ceres::ceres ALIAS ceres)
        meshfem_target_hide_warnings(ceres)
    endif()
elseif(NOT TARGET ceres::ceres)
    message(STATUS "Google's ceres-solver not found; MaterialOptimization_cli won't be built")
endif()
