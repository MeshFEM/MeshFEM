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
include(Warnings)

# Helper function
include(MeshFEMSingleApp)

################################################################################
# Options
################################################################################

# By default, disable C++11 ABI on the Prince cluster
cmake_host_system_information(RESULT MY_HOSTNAME QUERY HOSTNAME)
if("$ENV{CLUSTER}" STREQUAL "PRINCE")
    set(DISABLE_CXX11_ABI_DEFAULT ON)
else()
    set(DISABLE_CXX11_ABI_DEFAULT OFF)
endif()

# By default, compile binaries only for toplevel projects
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(MESHFEM_LIB_ONLY_DEFAULT OFF)
else()
    set(MESHFEM_LIB_ONLY_DEFAULT ON)
endif()

option(MESHFEM_DISABLE_CXX11_ABI_GCC "Don't use GCC's new C++11 ABI; needed to prevent linker errors with libraries compiled with the old ABI" ${DISABLE_CXX11_ABI_DEFAULT})
option(MESHFEM_ENABLE_BENCHMARKING   "Enable the benchmarking code" OFF)
option(MESHFEM_LIB_ONLY              "Compile only the MeshFEM library (not the binary applications)" ${MESHFEM_LIB_ONLY_DEFAULT})

################################################################################
# Dependencies
################################################################################

# Download and define targets for third-party dependencies
include(MeshFEMDependencies)

################################################################################
# Subdirectories
################################################################################

add_subdirectory(src/lib)
if(NOT MESHFEM_LIB_ONLY)
    add_subdirectory(src/bin)
endif()

################################################################################
# Configure manual compiler flags based on options
################################################################################

if(MESHFEM_DISABLE_CXX11_ABI_GCC)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -D_GLIBCXX_USE_CXX11_ABI=0)
endif()

if(MESHFEM_ENABLE_BENCHMARKING)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DBENCHMARK)
endif()

# Compile tests only if this is a top-level project
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # Unit tests
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()
