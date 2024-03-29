cmake_minimum_required(VERSION 3.16)
project(MeshFEM)

# CMP0063: Honor visibility properties for all target types.
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Detects whether this is a top-level project
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(MESHFEM_TOPLEVEL_PROJECT ON)
else()
    set(MESHFEM_TOPLEVEL_PROJECT OFF)
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

set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Choose the type of build. Options are: None Debug Release RelWithDebInfo MinSizeRel RelWithAssert")

################################################################################

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

# We need -fPIC when compiling our libraries and our dependencies for
# the python bindings (shared libraries) to link.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Helper functions
include(MeshFEMUtils)

################################################################################
# Options
################################################################################

option(MESHFEM_DISABLE_CXX11_ABI_GCC "Don't use GCC's new C++11 ABI; needed to prevent linker errors with libraries compiled with the old ABI" OFF)
option(MESHFEM_ENABLE_BENCHMARKING   "Enable the benchmarking code"                                                OFF)
option(MESHFEM_BUILD_BINARIES        "Build executable programs"                                                   ${MESHFEM_TOPLEVEL_PROJECT})
option(MESHFEM_BUILD_PYTHON_BINDINGS "Build python bindings"                                                       ON)
option(MESHFEM_BIND_LONG_DOUBLE      "Also bind the long-double FEMMesh instantiations"                            OFF)
option(MESHFEM_WITH_CERES            "Compile MeshFEM with Ceres"                                                  OFF)
option(MESHFEM_PREFER_SYSTEM_CERES   "Do not build a local version of Ceres if one is already found on the system" ON)
option(MESHFEM_WITH_UMFPACK          "Compile MeshFEM with UMFPACK for LU factorization (GPL licensed)"            OFF)
option(MESHFEM_WITH_OFFSCREEN_RENDER "Build the offscreen renderer (Requires GLEW/OpenGL or OSMesa)"               ON)
option(MESHFEM_FORCE_Z7              "Force /Z7 option on MSVC"                                                    OFF)
option(MESHFEM_WITH_TBB              "Compile MeshFEM with TBB support"                                            ON)
option(MESHFEM_VECTORIZE             "Add march=native and ffast-math flags to encourage generation of SIMD instructions. These options will propagate to code linking against MeshFEM"   OFF)

if(MESHFEM_FORCE_Z7)
    # Hard-coded string replace for now, until CMake provides a better way to do this:
    # https://gitlab.kitware.com/cmake/cmake/-/issues/19084
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELWITHASSERT "${CMAKE_CXX_FLAGS_RELWITHASSERT}")
endif()

################################################################################
# Dependencies
################################################################################

# Download and define targets for third-party dependencies
include(MeshFEMDependencies)

################################################################################
# Subdirectories
################################################################################

add_subdirectory(src/lib)

if(MESHFEM_BUILD_BINARIES)
    add_subdirectory(src/bin)
endif()

if(MESHFEM_BUILD_PYTHON_BINDINGS)
    # we compile for (without making them pass `-DPYTHON_EXECUTABLE=...` to `cmake`
    if (DEFINED ENV{PYTHON_EXECUTABLE})
        set(PYTHON_EXECUTABLE "$ENV{PYTHON_EXECUTABLE}" CACHE INTERNAL "Python executable for which to build pybind11 bindings")
        #message("Set python executable to ${PYTHON_EXECUTABLE}")
    endif()

    # Download pybind dependency when pybind11 is missing
    if(NOT TARGET pybind11::module)
        include(MeshFEMDownloadExternal)
        meshfem_download_project(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.6.1
        )
        add_subdirectory(${MESHFEM_EXTERNAL}/pybind11)
    endif()

    add_subdirectory(src/python_bindings)
endif()

if (MESHFEM_WITH_OFFSCREEN_RENDER)
add_subdirectory(3rdparty/OffscreenRenderer)
endif()

################################################################################
# Unit tests
################################################################################

# Compile tests only if this is a top-level project
if(MESHFEM_TOPLEVEL_PROJECT)
    # Unit tests
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()
