cmake_minimum_required(VERSION 3.14...3.31)
project(MeshFEMDemos)

if (${CMAKE_VERSION} VERSION_GREATER "3.13")
    cmake_policy(SET CMP0076 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Export compile flags (used for autocompletion of the C++ code)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Explicitly define OpenMP flags if libomp is installed
find_package(OpenMP)

if (OpenMP_FOUND)
    message(STATUS "OpenMP found, enabling support.")
else()
    message(WARNING "OpenMP not found. Building without OpenMP support!")
endif()

option(INCLUDE_TINYAD "Whether to include TinyAD and build associated bindings" ON)

# IsotropicAutodiff requires TinyAD even when the comparison module is disabled.
set(MESHFEM_FORCE_TINYAD_DOWNLOAD ON CACHE BOOL "" FORCE)

if (NOT TARGET MeshFEM)
    option(MESHFEM_ENABLE_BENCHMARKING "" ON)
    add_subdirectory(${THIRD_PARTY_DIR}/MeshFEM)
endif()

# Color errors/warnings.
list(APPEND CMAKE_MODULE_PATH ${THIRD_PARTY_DIR}/MeshFEM/3rdparty/MeshFEMCore/cmake)
include(UseColors)

include("${THIRD_PARTY_DIR}/TaylorAutodiff/SME2.cmake")
add_subdirectory(${THIRD_PARTY_DIR}/IsotropicAutodiff)
add_subdirectory(python_bindings)
