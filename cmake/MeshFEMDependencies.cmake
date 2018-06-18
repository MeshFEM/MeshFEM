# Prepare MeshFEM dependencies
#
# For each third-party library, if the appropriate target doesn't exist yet,
# download it via external project, and add_subdirectory to build it alongside
# MeshFEM

### Configuration
set(MESHFEM_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
set(MESHFEM_EXTERNAL "${MESHFEM_ROOT}/3rdparty")

# Download and update 3rdparty libraries
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include(MeshFEMDownloadExternal)

################################################################################
# Required libraries
################################################################################

# C++11 threads
find_package(Threads REQUIRED) # provides Threads::Threads

# Boost library
find_package(Boost 1.55 REQUIRED COMPONENTS filesystem system program_options)

# Eigen3 library
if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    meshfem_download_eigen()
    target_include_directories(Eigen3::Eigen SYSTEM INTERFACE ${MESHFEM_EXTERNAL}/eigen)
endif()

# json library
if(NOT TARGET json::json)
    add_library(json::json INTERFACE IMPORTED)
    meshfem_download_json()
    target_include_directories(json::json SYSTEM INTERFACE ${MESHFEM_EXTERNAL}/json)
    target_include_directories(json::json SYSTEM INTERFACE ${MESHFEM_EXTERNAL}/json/nlohmann)
endif()

# TBB library
if(NOT TARGET tbb::tbb)
    set(TBB_BUILD_STATIC ON CACHE BOOL " " FORCE)
    set(TBB_BUILD_SHARED OFF CACHE BOOL " " FORCE)
    set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL " " FORCE)
    set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL " " FORCE)
    set(TBB_BUILD_TESTS OFF CACHE BOOL " " FORCE)

    meshfem_download_tbb()
    add_subdirectory(${MESHFEM_EXTERNAL}/tbb tbb)
    set_property(TARGET tbb_static tbb_def_files PROPERTY FOLDER "dependencies")

    add_library(meshfem_tbb INTERFACE)
    target_include_directories(meshfem_tbb SYSTEM INTERFACE ${MESHFEM_EXTERNAL}/tbb/include)
    target_link_libraries(meshfem_tbb INTERFACE tbb_static)
    add_library(tbb::tbb ALIAS meshfem_tbb)
endif()

# Triangle library
if(NOT TARGET triangle::triangle)
    meshfem_download_triangle()
    add_subdirectory(${MESHFEM_EXTERNAL}/triangle triangle)
    target_include_directories(triangle INTERFACE ${MESHFEM_EXTERNAL}/triangle)
    add_library(triangle::triangle ALIAS triangle)
endif()

# TinyExpr library
if(NOT TARGET tinyexpr::tinyexpr)
    meshfem_download_tinyexpr()
    add_library(meshfem_tinyexpr ${MESHFEM_EXTERNAL}/tinyexpr/tinyexpr.c)
    target_include_directories(meshfem_tinyexpr PUBLIC ${MESHFEM_EXTERNAL}/tinyexpr)
    add_library(tinyexpr::tinyexpr ALIAS meshfem_tinyexpr)
endif()

# Cholmod solver
find_package(Cholmod REQUIRED) # provides cholmod::cholmod

# UmfPack solver
find_package(Umfpack REQUIRED) # provides umfpack::umfpack

################################################################################
# Optional libraries
################################################################################

# find_package(Ceres)
# if(CERES_FOUND)
#     add_library(ceres_lib INTERFACE)
#     target_include_directories(ceres_lib SYSTEM PUBLIC ${CERES_INCLUDE_DIRS})
#     target_link_libraries(ceres_lib MeshFEM ${CERES_LIBRARIES})
#     add_library(ceres::ceres ALIAS ceres_lib)
# else()
#     message(STATUS "Google's ceres-solver not found; MaterialOptimization_cli won't be built")
# endif()
