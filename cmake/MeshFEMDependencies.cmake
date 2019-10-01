# Prepare dependencies
#
# For each third-party library, if the appropriate target doesn't exist yet,
# download it via external project, and add_subdirectory to build it alongside
# this project.

### Configuration
set(MESHFEM_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
set(MESHFEM_EXTERNAL "${MESHFEM_ROOT}/3rdparty")

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
find_package(Boost 1.54 REQUIRED COMPONENTS filesystem system program_options QUIET)
if(NOT TARGET meshfem::boost)
    add_library(meshfem_boost INTERFACE)
    if(TARGET Boost::filesystem AND TARGET Boost::system AND TARGET Boost::program_options)
		#target_include_directories(meshfem_boost SYSTEM INTERFACE ${Boost_INCLUDE_DIRS})
        target_link_libraries(meshfem_boost INTERFACE
            Boost::filesystem
            Boost::system
            Boost::program_options)
    else()
        # When CMake and Boost versions are not in sync, imported targets may not be available... (sigh)
		target_include_directories(meshfem_boost SYSTEM INTERFACE ${Boost_INCLUDE_DIRS})
        target_link_libraries(meshfem_boost INTERFACE ${Boost_LIBRARIES})
    endif()
    add_library(meshfem::boost ALIAS meshfem_boost)
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
    set(TBB_BUILD_STATIC ON CACHE BOOL " " FORCE)
    set(TBB_BUILD_SHARED OFF CACHE BOOL " " FORCE)
    set(TBB_BUILD_TBBMALLOC ON CACHE BOOL " " FORCE) # needed for CGAL's parallel mesher
    set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL " " FORCE)
    set(TBB_BUILD_TESTS OFF CACHE BOOL " " FORCE)

    meshfem_download_tbb()
    add_subdirectory(${MESHFEM_EXTERNAL}/tbb tbb)
    set_property(TARGET tbb_static tbb_def_files PROPERTY FOLDER "dependencies")
    #set_target_properties(tbb_static PROPERTIES COMPILE_FLAGS "-Wno-implicit-fallthrough -Wno-missing-field-initializers -Wno-unused-parameter -Wno-keyword-macro")

    add_library(meshfem_tbb INTERFACE)
    target_include_directories(meshfem_tbb SYSTEM INTERFACE ${MESHFEM_EXTERNAL}/tbb/include)
    target_link_libraries(meshfem_tbb INTERFACE tbb_static tbbmalloc_static)
    add_library(tbb::tbb ALIAS meshfem_tbb)
endif()

# Triangle library
if(NOT TARGET triangle::triangle)
    meshfem_download_triangle()
    add_subdirectory(${MESHFEM_EXTERNAL}/triangle triangle)
    target_include_directories(triangle SYSTEM INTERFACE ${MESHFEM_EXTERNAL}/triangle)
    add_library(triangle::triangle ALIAS triangle)
endif()

# TinyExpr library
if(NOT TARGET tinyexpr::tinyexpr)
    meshfem_download_tinyexpr()
    add_library(meshfem_tinyexpr ${MESHFEM_EXTERNAL}/tinyexpr/tinyexpr.c)
    target_include_directories(meshfem_tinyexpr SYSTEM PUBLIC ${MESHFEM_EXTERNAL}/tinyexpr)
    add_library(tinyexpr::tinyexpr ALIAS meshfem_tinyexpr)
endif()

# Cholmod solver
find_package(Cholmod REQUIRED) # provides cholmod::cholmod

# UmfPack solver
find_package(Umfpack REQUIRED) # provides umfpack::umfpack

################################################################################
# Optional libraries
################################################################################

find_package(Ceres QUIET)
if(CERES_FOUND)
    add_library(ceres_lib INTERFACE)
    target_include_directories(ceres_lib SYSTEM INTERFACE  ${CERES_INCLUDE_DIRS})
    target_link_libraries(ceres_lib INTERFACE MeshFEM ${CERES_LIBRARIES})
    add_library(ceres::ceres ALIAS ceres_lib)
else()
    message(STATUS "Google's ceres-solver not found; MaterialOptimization_cli won't be built")
endif()
