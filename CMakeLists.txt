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

# Export compile flags (used for autocompletion of the C++ code)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# CMake plugin for vscode
include(CMakeToolsHelpers OPTIONAL)

# Enable more warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-comment")

################################################################################
# Platform-specific options
################################################################################

option(DISABLE_CXX11_ABI_GCC "Don't use GCC's new C++11 ABI; needed to prevent linker errors with libraries compiled with the old ABI" OFF)

################################################################################
# MeshFEM sources
################################################################################

add_library(${PROJECT_NAME}
		BoundaryConditions.cc
		EdgeFields.cc
		GlobalBenchmark.cc
		Materials.cc
		MeshIO.cc
		MSHFieldParser.cc
		Types.cc

		BaseCellType.hh
		BoundaryConditions.hh
		BoundaryLaplacian.hh
		BoundaryMesh.hh
		CollisionGrid.hh
		ComponentMask.hh
		Concepts.hh
		DenseCollisionGrid.hh
		EdgeFields.hh
		EmbeddedElement.hh
		ExpressionVector.hh
		FEMMesh.hh
		function_traits.hh
		Functions.hh
		Future.hh
		GaussQuadrature.hh
		Geometry.hh
		GlobalBenchmark.hh
		GridFunction.hh
		HalfEdge.hh
		HalfedgeDictionary.hh
		InterpolantRestriction.hh
		JSFieldWriter.hh
		Laplacian.hh
		LinearElasticity.hh
		MassMatrix.hh
		MaterialField.hh
		MaterialOptimization.hh
		Materials.hh
		MeshDataTraits.hh
		MeshIO.hh
		MSHFieldParser.hh
		MSHFieldWriter.hh
		NTuple.hh
		OneForm.hh
		OrthotropicHomogenization.hh
		PeriodicBoundaryMatcher.hh
		PeriodicHomogenization.hh
		PerturbMesh.hh
		Poisson.hh
		Simplex.hh
		SimplicialMesh.hh
		SimplicialMeshInterface.hh
		SymmetricMatrixInterpolant.hh
		TemplateHacks.hh
		TetMesh.hh
		TriMesh.hh
		Types.hh
		UniformLaplacian.hh

		Triangulate.h
		util.h
)

target_include_directories(${PROJECT_NAME} PUBLIC .)

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
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_THREAD_LIBS_INIT})

# Boost library
find_package(Boost 1.55 REQUIRED COMPONENTS filesystem system program_options thread)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${Boost_LIBRARIES})

# Eigen3 library
find_package(Eigen3 REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

# CSGFEM library
find_package(CSGFEM REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC CSGFEM)

# TBB library
find_package(TBB)
if(TBB_FOUND)
	target_compile_definitions(${PROJECT_NAME} PUBLIC -DHAS_TBB)
	target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${TBB_INCLUDE_DIRS})
	target_link_libraries(${PROJECT_NAME} PUBLIC ${TBB_LIBRARIES})
endif()

# Triangle library
find_package(Triangle REQUIRED)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${Triangle_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${Triangle_LIBRARIES})

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
# Independent binaries
################################################################################

add_executable(mesh_convert mesh_convert.cc)
target_link_libraries(mesh_convert ${PROJECT_NAME})
set_target_properties(mesh_convert PROPERTIES CXX_STANDARD 11)

add_executable(PeriodicHomogenization_cli PeriodicHomogenization_cli.cc)
target_link_libraries(PeriodicHomogenization_cli ${PROJECT_NAME})
set_target_properties(PeriodicHomogenization_cli PROPERTIES CXX_STANDARD 11)

add_executable(ConstStrainDisplacement_cli ConstStrainDisplacement_cli.cc)
target_link_libraries(ConstStrainDisplacement_cli ${PROJECT_NAME})
set_target_properties(ConstStrainDisplacement_cli PROPERTIES CXX_STANDARD 11)

add_executable(DeformedCells_cli DeformedCells_cli.cc)
target_link_libraries(DeformedCells_cli ${PROJECT_NAME})
set_target_properties(DeformedCells_cli PROPERTIES CXX_STANDARD 11)

if(CERES_FOUND)
	add_executable(MaterialOptimization_cli MaterialOptimization_cli.cc MaterialOptimization.cc)
	target_link_libraries(MaterialOptimization_cli ${PROJECT_NAME})
	set_target_properties(MaterialOptimization_cli PROPERTIES CXX_STANDARD 11)
endif()

add_executable(Simulate_cli Simulate_cli.cc)
target_link_libraries(Simulate_cli ${PROJECT_NAME})
set_target_properties(Simulate_cli PROPERTIES CXX_STANDARD 11)

add_executable(Poisson_cli Poisson_cli.cc)
target_link_libraries(Poisson_cli ${PROJECT_NAME})
set_target_properties(Poisson_cli PROPERTIES CXX_STANDARD 11)

add_executable(ExtractBMatrix ExtractBMatrix.cc)
target_link_libraries(ExtractBMatrix ${PROJECT_NAME})
set_target_properties(ExtractBMatrix PROPERTIES CXX_STANDARD 11)

################################################################################
# Include binaries in tools/
################################################################################

add_subdirectory(tools)
