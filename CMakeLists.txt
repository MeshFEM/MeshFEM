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

set(MESHFEM_THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/)
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

# Export compile flags (used for autocompletion of the C++ code)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# CMake plugin for vscode
include(CMakeToolsHelpers OPTIONAL)

# Enable more warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-comment")

################################################################################
# Platform-specific options
################################################################################

cmake_host_system_information(RESULT MY_HOSTNAME QUERY HOSTNAME)
if(MY_HOSTNAME MATCHES "log-*")
    set(DISABLE_CXX11_ABI_DEFAULT ON)
else()
    set(DISABLE_CXX11_ABI_DEFAULT OFF)
endif()
option(DISABLE_CXX11_ABI_GCC "Don't use GCC's new C++11 ABI; needed to prevent linker errors with libraries compiled with the old ABI" ${DISABLE_CXX11_ABI_DEFAULT})

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

target_include_directories(${PROJECT_NAME} PUBLIC src/lib/CSGFEM) # will be removed
target_include_directories(${PROJECT_NAME} PUBLIC src/lib/MeshFEM) # will be removed
target_include_directories(${PROJECT_NAME} PUBLIC src/lib)

################################################################################
# Configure manual compiler flags based on options
################################################################################

if(DISABLE_CXX11_ABI_GCC)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -D_GLIBCXX_USE_CXX11_ABI=0)
endif()

################################################################################
# Required libraries
################################################################################

# Use C++11
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)

# # Hide internal symbols
# set_target_properties(${PROJECT_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)

# C++11 Thread library
# find_package(Threads REQUIRED)
# target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_THREAD_LIBS_INIT})

# Boost library
find_package(Boost 1.55 REQUIRED COMPONENTS filesystem system program_options)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${Boost_LIBRARIES})

# Eigen3 library
find_package(Eigen3 REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

# json library
if(NOT TARGET json)
    add_library(json INTERFACE)
    target_include_directories(json SYSTEM INTERFACE ${MESHFEM_THIRD_PARTY_DIR}/json)
endif()
target_link_libraries(${PROJECT_NAME} PUBLIC json)

# TBB library
find_package(TBB)
if(TBB_FOUND)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DHAS_TBB)
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${TBB_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${TBB_LIBRARIES})
endif()

# Triangle library
# find_package(Triangle REQUIRED)
# target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${Triangle_INCLUDE_DIRS})
# target_link_libraries(${PROJECT_NAME} PUBLIC ${Triangle_LIBRARIES})
if(NOT TARGET triangle)
    add_subdirectory(${MESHFEM_THIRD_PARTY_DIR}/triangle triangle)
    target_link_libraries(${PROJECT_NAME} PUBLIC triangle)
endif()

# Matheval library
find_package(Matheval REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Matheval::matheval)

# Cholmod solver
find_package(Cholmod REQUIRED)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${CHOLMOD_INCLUDES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${CHOLMOD_LIBRARIES})

# UmfPack solver
find_package(Umfpack REQUIRED)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${UMFPACK_INCLUDES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${UMFPACK_LIBRARIES})

# Lapack library (do we need this?)
# find_package(LAPACK REQUIRED)
# target_link_libraries(${PROJECT_NAME} PUBLIC ${LAPACK_LIBRARIES})

# Ceres library
find_package(Ceres)
if(CERES_FOUND)
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${CERES_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${CERES_LIBRARIES})
else()
    message(STATUS "Google's ceres-solver not found; MaterialOptimization_cli won't be built")
endif()

################################################################################
# Subdirectories
################################################################################

add_subdirectory(src/bin)
add_subdirectory(tests)
