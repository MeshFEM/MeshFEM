################################################################################
# General Informations
################################################################################

cmake_minimum_required(VERSION 3.14...3.31)
project(MeshFEM)

# CMP0063: Honor visibility properties for all target types.
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

# Export compile flags (used for autocompletion of the C++ code)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Detects whether this is a top-level project
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(MESHFEM_TOPLEVEL_PROJECT ON)
else()
    set(MESHFEM_TOPLEVEL_PROJECT OFF)
endif()

################################################################################

list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
)
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)

set(MESHFEM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(MESHFEM_EXTERNAL "${MESHFEM_ROOT}/3rdparty")

include(FetchContent)
include(MeshFEMFetchEditableRepo)

################################################################################
# Options
################################################################################
option(MESHFEM_WITH_OFFSCREEN_RENDER "Build the offscreen renderer (Requires GLEW/OpenGL or OSMesa)"    ON)
option(MESHFEM_WITH_CERES            "Compile MeshFEM with Ceres"   OFF)
option(MESHFEM_PREFER_SYSTEM_CERES   "Do not build a local version of Ceres if one is already found on the system"   ON)
option(MESHFEM_WITH_IPC_TOOLKIT      "Compile MeshFEM with Ipc_toolkit" ON)
option(MESHFEM_WITH_TINYAD           "Whether to use the TinyAD `Scalar` type as the autodiff backend intead of (nested) Eigen" ON)
option(MESHFEM_BIND_LONG_DOUBLE      "Also bind the long-double FEMMesh instantiations" OFF)

option(MESHFEM_FORCE_TINYAD_DOWNLOAD "Whether to force the download of TinyAD (on behalf of the parent project) even when not using TinyAD for MeshFEM's autodiff" OFF)
mark_as_advanced(MESHFEM_FORCE_TINYAD_DOWNLOAD)

if (MESHFEM_WITH_PARDISO AND MESHFEM_WITH_MKL_PARDISO)
    message( FATAL_ERROR "MESHFEM_WITH_PARDISO and MESHFEM_WITH_MKL_PARDISO are mutually exclusive.")
endif()

if (MESHFEM_ADDRESS_SANITIZER)
    # Enable address sanitizer in debug mode.
    set (CMAKE_C_FLAGS_DEBUG      "${CMAKE_C_FLAGS_DEBUG}      -fno-omit-frame-pointer -fsanitize=address -fsanitize-recover=address")
    set (CMAKE_CXX_FLAGS_DEBUG    "${CMAKE_CXX_FLAGS_DEBUG}    -fno-omit-frame-pointer -fsanitize=address -fsanitize-recover=address")
    set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize-recover=address")
endif()

option(MESHFEM_BUILD_BINARIES        "Build executable programs"    ${MESHFEM_TOPLEVEL_PROJECT})
option(MESHFEM_BUILD_TESTS           "Build MeshFEM tests"           ${MESHFEM_BUILD_BINARIES})

# We need -fPIC when compiling our libraries and our dependencies for
# the python bindings (shared libraries) to link.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MESHFEM_BUILD_TESTS)
    include(CTest)
endif()

if(NOT DEFINED MESHFEMCORE_BUILD_TESTS)
    set(MESHFEMCORE_BUILD_TESTS ${MESHFEM_BUILD_TESTS} CACHE BOOL "Build MeshFEMCore tests")
endif()

# Bring in required MeshFEM components
set(MESHFEMCORE_GIT_REPOSITORY "https://github.com/MeshFEM/MeshFEMCore.git" CACHE STRING "MeshFEMCore git repository")
set(MESHFEMCORE_GIT_TAG "main" CACHE STRING "MeshFEMCore git revision")
set(MESHFEMSPARSE_GIT_REPOSITORY "https://github.com/MeshFEM/MeshFEMSparse.git" CACHE STRING "MeshFEMSparse git repository")
set(MESHFEMSPARSE_GIT_TAG "main" CACHE STRING "MeshFEMSparse git revision")
set(ROTATIONOPTIMIZATION_GIT_REPOSITORY "https://github.com/EPFL-LGG/RotationOptimization.git" CACHE STRING "RotationOptimization git repository")
set(ROTATIONOPTIMIZATION_GIT_TAG "b75f269e57cec86e40d8c9ab2fadecaad392774b" CACHE STRING "RotationOptimization git revision")
set(OFFSCREENRENDERER_GIT_REPOSITORY "https://github.com/MeshFEM/OffscreenRenderer.git" CACHE STRING "OffscreenRenderer git repository")
set(OFFSCREENRENDERER_GIT_TAG "e60c3e731fb41253bbe6291b9c233d907c8f0e9e" CACHE STRING "OffscreenRenderer git revision")

meshfem_prepare_editable_repo(MeshFEMCore
    "${MESHFEM_EXTERNAL}/MeshFEMCore" "${MESHFEMCORE_GIT_TAG}")
FetchContent_Declare(MeshFEMCore
    GIT_REPOSITORY ${MESHFEMCORE_GIT_REPOSITORY}
    GIT_TAG        ${MESHFEMCORE_GIT_TAG}
    SOURCE_DIR     ${MESHFEM_EXTERNAL}/MeshFEMCore
)
FetchContent_GetProperties(MeshFEMCore)
if(NOT TARGET MeshFEMCore)
    FetchContent_MakeAvailable(MeshFEMCore)
endif()
list(APPEND CMAKE_MODULE_PATH ${meshfemcore_SOURCE_DIR}/cmake)
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)
include(MeshFEMCoreDownloadExternal)

if(MSVC)
    set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHASSERT Release RelWithDebInfo NoConfig None "")
endif()

if (MSVC)
    set(MESHFEM_RELWITHASSERT_OPT "/O2")
else()
    set(MESHFEM_RELWITHASSERT_OPT "-O3")
endif()
set(CMAKE_CXX_FLAGS_RELWITHASSERT "${MESHFEM_RELWITHASSERT_OPT}" CACHE STRING "Flags used during RelWithAssert builds" FORCE)
set(CMAKE_C_FLAGS_RELWITHASSERT   "${MESHFEM_RELWITHASSERT_OPT}" CACHE STRING "Flags used during RelWithAssert builds" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELWITHASSERT    "" CACHE STRING "Flags used during RelWithAssert builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssert builds" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssert builds" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssert builds" FORCE)
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

# CMake plugin for vscode
include(CMakeToolsHelpers OPTIONAL)

# Enable more warnings
include(Warnings)

# Helper functions
include(MeshFEMUtils)

include(sanitizers)

################################################################################
# Subdirectories
################################################################################
# RotationOptimization consumes Eigen when it is available as a CMake target.
# Prepare MeshFEMCore's foundational dependencies before adding this third-party
# project so it does not configure its bundled Eigen test tree.
include(MeshFEMCoreDependencies)

if(NOT TARGET RotationOptimization)
    meshfem_prepare_editable_repo(RotationOptimization
        "${MESHFEM_EXTERNAL}/RotationOptimization" "${ROTATIONOPTIMIZATION_GIT_TAG}")
    FetchContent_Declare(RotationOptimization
        GIT_REPOSITORY ${ROTATIONOPTIMIZATION_GIT_REPOSITORY}
        GIT_TAG        ${ROTATIONOPTIMIZATION_GIT_TAG}
        SOURCE_DIR     ${MESHFEM_EXTERNAL}/RotationOptimization
    )
    FetchContent_MakeAvailable(RotationOptimization)
endif()

# RotationOptimization FORCEs these back to -O3, so re-assert them after it.
set(CMAKE_CXX_FLAGS_RELWITHASSERT "${MESHFEM_RELWITHASSERT_OPT}" CACHE STRING "Flags used during RelWithAssert builds" FORCE)
set(CMAKE_C_FLAGS_RELWITHASSERT   "${MESHFEM_RELWITHASSERT_OPT}" CACHE STRING "Flags used during RelWithAssert builds" FORCE)

if(NOT DEFINED MESHFEMSPARSE_BUILD_TESTS)
    set(MESHFEMSPARSE_BUILD_TESTS ${MESHFEM_BUILD_TESTS} CACHE BOOL "Build MeshFEMSparse tests")
endif()

if(NOT TARGET MeshFEMSparse)
    meshfem_prepare_editable_repo(MeshFEMSparse
        "${MESHFEM_EXTERNAL}/MeshFEMSparse" "${MESHFEMSPARSE_GIT_TAG}")
    FetchContent_Declare(MeshFEMSparse
        GIT_REPOSITORY ${MESHFEMSPARSE_GIT_REPOSITORY}
        GIT_TAG        ${MESHFEMSPARSE_GIT_TAG}
        SOURCE_DIR     ${MESHFEM_EXTERNAL}/MeshFEMSparse
    )
    FetchContent_MakeAvailable(MeshFEMSparse)
endif()

add_subdirectory(src/lib)

if(MESHFEM_BUILD_BINARIES)
    add_subdirectory(src/bin)
endif()

# Allow the user's PYTHON_EXECUTABLE environment variable influence the python
# we compile for (without making them pass `-DPYTHON_EXECUTABLE=...` to `cmake`
if (DEFINED ENV{PYTHON_EXECUTABLE})
    file(TO_CMAKE_PATH "$ENV{PYTHON_EXECUTABLE}" MESHFEM_ENV_PYTHON)
    set(PYTHON_EXECUTABLE "${MESHFEM_ENV_PYTHON}" CACHE INTERNAL "Python executable for which to build pybind11 bindings")
    if (NOT DEFINED Python_EXECUTABLE)
        set(Python_EXECUTABLE "${MESHFEM_ENV_PYTHON}" CACHE FILEPATH "Python executable for pybind11 bindings")
    endif()
    if (NOT DEFINED Python3_EXECUTABLE)
        set(Python3_EXECUTABLE "${MESHFEM_ENV_PYTHON}" CACHE FILEPATH "Python executable for pybind11 bindings")
    endif()
    unset(MESHFEM_ENV_PYTHON)
endif()

# Download pybind dependency when pybind11 is missing
if(NOT TARGET pybind11::module)
    include(MeshFEMDownloadExternal)
    meshfem_download_project(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG a2e59f0e7065404b44dfe92a28aca47ba1378dc4
    )
    add_subdirectory(${MESHFEM_EXTERNAL}/pybind11)
endif()

add_subdirectory(src/python_bindings)

if(MESHFEM_WITH_OFFSCREEN_RENDER)
    meshfem_prepare_editable_repo(OffscreenRenderer
        "${MESHFEM_EXTERNAL}/OffscreenRenderer" "${OFFSCREENRENDERER_GIT_TAG}")
    FetchContent_Declare(OffscreenRenderer
        GIT_REPOSITORY ${OFFSCREENRENDERER_GIT_REPOSITORY}
        GIT_TAG        ${OFFSCREENRENDERER_GIT_TAG}
        SOURCE_DIR     ${MESHFEM_EXTERNAL}/OffscreenRenderer
    )
    FetchContent_MakeAvailable(OffscreenRenderer)
endif()

################################################################################
# Unit tests
################################################################################

# Compile tests if requested.
if (MESHFEM_BUILD_TESTS AND BUILD_TESTING)
    list(APPEND CMAKE_MODULE_PATH ${MESHFEM_EXTERNAL}/Catch2/contrib)
    add_subdirectory(tests)
endif()
