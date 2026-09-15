cmake_minimum_required(VERSION 3.1...3.29)
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

if (INCLUDE_TINYAD) # If we want TinyAD, tell MeshFEM to download it
    set(MESHFEM_FORCE_TINYAD_DOWNLOAD ON CACHE BOOL "" FORCE)
endif()

if (NOT TARGET MeshFEM)
    option(MESHFEM_ENABLE_BENCHMARKING "" ON)
    add_subdirectory(${THIRD_PARTY_DIR}/MeshFEM)
endif()

# Color errors/warnings.
list(APPEND CMAKE_MODULE_PATH ${THIRD_PARTY_DIR}/MeshFEM/cmake)
include(UseColors)

# Fix the Unknown CMake command "taylorad_configure_sme2"
include("${THIRD_PARTY_DIR}/TaylorAutodiff/SME2.cmake")

add_subdirectory(python_bindings)
