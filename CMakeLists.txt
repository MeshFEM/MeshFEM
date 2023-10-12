cmake_minimum_required(VERSION 3.1)
project(MeshFEMDemos)

if (${CMAKE_VERSION} VERSION_GREATER "3.13")
    cmake_policy(SET CMP0076 NEW)
endif()

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Export compile flags (used for autocompletion of the C++ code)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

if (NOT TARGET MeshFEM)
    option(MESHFEM_ENABLE_BENCHMARKING "" ON)
    add_subdirectory(${THIRD_PARTY_DIR}/MeshFEM)
endif()

# Color errors/warnings.
list(APPEND CMAKE_MODULE_PATH ${THIRD_PARTY_DIR}/MeshFEM/cmake)
include(UseColors)

add_library(DiscreteShell INTERFACE)
target_sources(DiscreteShell INTERFACE DiscreteShell.hh)
target_link_libraries(DiscreteShell INTERFACE MeshFEM warnings::all)

add_subdirectory(python_bindings)
