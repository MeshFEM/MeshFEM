################################################################################
# General Informations
################################################################################

cmake_minimum_required(VERSION 3.1)
project(MeshFEM)

# CMP0063: Honor visibility properties for all target types.
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

################################################################################

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS_RELWITHASSERT        "-O3" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_C_FLAGS_RELWITHASSERT          "-O3" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELWITHASSERT    "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
mark_as_advanced(
    CMAKE_CXX_FLAGS_RELWITHASSERT
    CMAKE_C_FLAGS_RELWITHASSERT
    CMAKE_EXE_LINKER_FLAGS_RELWITHASSERT
    CMAKE_SHARED_LINKER_FLAGS_RELWITHASSERT
    CMAKE_MODULE_LINKER_FLAGS_RELWITHASSERT
    CMAKE_STATIC_LINKER_FLAGS_RELWITHASSERT
)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to RelWithAssert")
    set(CMAKE_BUILD_TYPE "RelWithAssert")
endif()

set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Choose the type of build. Options are: None Debug Release RelWithDebInfo MinSizeRel RelWithAssert" FORCE)

# Color output
include(UseColors)

# C++11/14 features
include(CXXFeatures)

# Export compile flags (used for autocompletion of the C++ code)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# CMake plugin for vscode
include(CMakeToolsHelpers OPTIONAL)

# Enable more warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-comment")

# Helper function
include(MeshFEMSingleApp)

################################################################################
# Options
################################################################################

cmake_host_system_information(RESULT MY_HOSTNAME QUERY HOSTNAME)
if("$ENV{CLUSTER}" STREQUAL "PRINCE")
    set(DISABLE_CXX11_ABI_DEFAULT ON)
else()
    set(DISABLE_CXX11_ABI_DEFAULT OFF)
endif()

option(DISABLE_CXX11_ABI_GCC "Don't use GCC's new C++11 ABI; needed to prevent linker errors with libraries compiled with the old ABI" ${DISABLE_CXX11_ABI_DEFAULT})
option(ENABLE_BENCHMARKING "Enable the benchmarking code" OFF)

################################################################################
# MeshFEM sources
################################################################################

add_library(${PROJECT_NAME}
        src/lib/MeshFEM/BoundaryConditions.cc
        src/lib/MeshFEM/EdgeFields.cc
        src/lib/MeshFEM/GlobalBenchmark.cc
        src/lib/MeshFEM/Materials.cc
        src/lib/MeshFEM/MeshIO.cc
        src/lib/MeshFEM/MSHFieldParser.cc
        src/lib/MeshFEM/Types.cc
)

target_include_directories(${PROJECT_NAME} PUBLIC src/lib)

# Use C++11
target_compile_features(${PROJECT_NAME} PUBLIC ${CXX11_FEATURES})

################################################################################
# Configure manual compiler flags based on options
################################################################################

if(DISABLE_CXX11_ABI_GCC)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -D_GLIBCXX_USE_CXX11_ABI=0)
endif()

if(ENABLE_BENCHMARKING)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DBENCHMARK)
endif()

################################################################################
# Dependencies
################################################################################

# Download and define targets
include(MeshFEMDependencies)

target_link_libraries(${PROJECT_NAME}
        Threads::Threads
        meshfem::boost
        Eigen3::Eigen
        json::json
        tbb::tbb
        triangle::triangle
        tinyexpr::tinyexpr
        cholmod::cholmod
        umfpack::umfpack
)

target_compile_definitions(${PROJECT_NAME} PUBLIC -DMESHFEM_WITH_TBB)

################################################################################
# Subdirectories
################################################################################

add_subdirectory(src/bin)
add_subdirectory(tests)
