# Prepare dependencies owned by the main MeshFEM library/application layer.

if(NOT DEFINED MESHFEM_ROOT)
    get_filename_component(MESHFEM_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

if(NOT DEFINED MESHFEM_EXTERNAL)
    set(MESHFEM_EXTERNAL "${MESHFEM_ROOT}/3rdparty")
endif()

get_directory_property(hasParent PARENT_DIRECTORY)
if (hasParent)
    set(MESHFEM_ROOT "${MESHFEM_ROOT}" PARENT_SCOPE)
    set(MESHFEM_EXTERNAL "${MESHFEM_EXTERNAL}" PARENT_SCOPE)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)

include(MeshFEMDownloadExternal)
include(MeshFEMUtils)
include(Warnings)

################################################################################
# Main MeshFEM dependencies
################################################################################

# Boost library
# The unit tests now use Boost::iostreams to load lzma-compressed data...
find_package(Boost 1.54 REQUIRED COMPONENTS filesystem system iostreams program_options QUIET)
if(NOT TARGET meshfem::boost)
    add_library(meshfem_boost INTERFACE)
    if(TARGET Boost::filesystem AND TARGET Boost::system AND TARGET Boost::program_options AND TARGET Boost::iostreams)
        target_link_libraries(meshfem_boost INTERFACE
            Boost::filesystem
            Boost::iostreams
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
if(NOT TARGET Catch2::Catch2 AND MESHFEM_BUILD_TESTS)
    meshfem_download_catch()
    add_subdirectory(${MESHFEM_EXTERNAL}/Catch2 ${CMAKE_BINARY_DIR}/3rdparty/Catch2)
    list(APPEND CMAKE_MODULE_PATH ${MESHFEM_EXTERNAL}/Catch2/contrib)
endif()

# json library
if(NOT TARGET nlohmann_json::nlohmann_json)
    meshfem_download_json()
    add_library(meshfem_json INTERFACE)
    target_include_directories(meshfem_json SYSTEM INTERFACE ${MESHFEM_EXTERNAL}/json/include)
    target_include_directories(meshfem_json SYSTEM INTERFACE ${MESHFEM_EXTERNAL}/json/include/nlohmann)
    add_library(nlohmann_json::nlohmann_json ALIAS meshfem_json)
    add_library(json::json ALIAS meshfem_json)
endif()

# Optional library
if(NOT TARGET optional::optional)
    meshfem_download_optional()
    add_library(optional_lite INTERFACE)
    target_include_directories(optional_lite SYSTEM INTERFACE ${MESHFEM_EXTERNAL}/optional/include)
    add_library(optional::optional ALIAS optional_lite)
endif()

# Triangle library
if(NOT TARGET triangle::triangle)
    meshfem_download_triangle()
    add_subdirectory(${MESHFEM_EXTERNAL}/triangle triangle)
    target_include_directories(triangle SYSTEM INTERFACE ${MESHFEM_EXTERNAL}/triangle)
    add_library(triangle::triangle ALIAS triangle)
endif()

# Spectra library
if(NOT TARGET spectra::spectra)
    meshfem_download_spectra()
    add_library(meshfem_spectra INTERFACE)
    target_include_directories(meshfem_spectra SYSTEM INTERFACE ${MESHFEM_EXTERNAL}/spectra/include)
    add_library(meshfem::spectra ALIAS meshfem_spectra)
endif()

# TinyExpr library
if(NOT TARGET tinyexpr::tinyexpr)
    meshfem_download_tinyexpr()
    add_library(meshfem_tinyexpr STATIC ${MESHFEM_EXTERNAL}/tinyexpr/tinyexpr.c)
    target_include_directories(meshfem_tinyexpr SYSTEM PUBLIC ${MESHFEM_EXTERNAL}/tinyexpr)
    # UCRT's ceil/floor are dllimport intrinsics, so under /Oi their addresses are not link-time
    # constants and tinyexpr's builtin table is rejected (C2099). Do not patch tinyexpr instead.
    target_compile_options(meshfem_tinyexpr PRIVATE $<$<C_COMPILER_ID:MSVC>:/Oi->)
    add_library(tinyexpr::tinyexpr ALIAS meshfem_tinyexpr)
endif()

################################################################################
# Optional MeshFEM dependencies
################################################################################

# Ceres
if (MESHFEM_WITH_CERES AND NOT TARGET ceres::ceres)
    if (MESHFEM_PREFER_SYSTEM_CERES)
        find_package(Ceres QUIET)
         if(CERES_FOUND)
             add_library(ceres_lib INTERFACE)
             target_include_directories(ceres_lib SYSTEM INTERFACE  ${CERES_INCLUDE_DIRS})
             target_link_libraries(ceres_lib INTERFACE MeshFEM ${CERES_LIBRARIES})
             add_library(ceres::ceres ALIAS ceres_lib)
         endif()
    endif()
    if (NOT TARGET ceres::ceres)
        meshfem_download_ceres()
        option(MINIGLOG "" ON)
        set(BUILD_TESTING OFF CACHE BOOL " " FORCE)
        set(BUILD_DOCUMENTATION OFF CACHE BOOL " " FORCE)
        set(BUILD_EXAMPLES OFF CACHE BOOL " " FORCE)
        set(BUILD_BENCHMARKS OFF CACHE BOOL " " FORCE)
        get_target_property(EIGEN_INCLUDE_DIR_HINTS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
        set(EIGEN_PREFER_EXPORTED_EIGEN_CMAKE_CONFIGURATION FALSE)
        add_subdirectory(${MESHFEM_EXTERNAL}/ceres ${CMAKE_BINARY_DIR}/3rdparty/ceres)
        add_library(ceres::ceres ALIAS ceres)
        meshfem_target_hide_warnings(ceres)
    endif()
elseif(NOT TARGET ceres::ceres)
    message(STATUS "Google's ceres-solver not found; MaterialOptimization_cli won't be built")
endif()

if (MESHFEM_WITH_IPC_TOOLKIT AND NOT TARGET ipc::toolkit)
    meshfem_download_ipc_toolkit()

    # We hit ODR violations/alignment issues if ipc_toolkit is built with an incompatible `march` setting from MeshFEM.
    # Specifically, we get a SEGFAULT when attempting an 32-byte aligned read from an unaligned address
    # (copying from IPC's insufficiently aligned Eigen::MatrixXd into our aligned Eigen::MatrixXd).
    # We therefore force a configuration that is compatible.
    set(IPC_TOOLKIT_WITH_SIMD OFF) # Note that `ipc_toolkit`'s autodetect feature seems to apply different flags on AVX-512 systems...
    add_subdirectory(${MESHFEM_EXTERNAL}/ipc_toolkit ${CMAKE_BINARY_DIR}/3rdparty/ipc_toolkit)

    if(MESHFEM_NATIVE)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(-march=native COMPILER_SUPPORTS_MARCH_NATIVE)
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            message(STATUS "Enabling -march=native for ipc_toolkit to match MeshFEM's MESHFEM_NATIVE option")
            target_compile_options(ipc_toolkit PRIVATE -march=native)
        else()
            check_cxx_compiler_flag(-mcpu=native COMPILER_SUPPORTS_MCPU_NATIVE)
            if(COMPILER_SUPPORTS_MCPU_NATIVE)
                message(STATUS "Enabling -mcpu=native for ipc_toolkit to match MeshFEM's MESHFEM_NATIVE option")
                target_compile_options(ipc_toolkit PRIVATE -mcpu=native)
            endif()
        endif()
    endif()
endif()

if ((MESHFEM_WITH_TINYAD OR MESHFEM_FORCE_TINYAD_DOWNLOAD) AND (NOT TARGET TinyAD))
    meshfem_download_tinyad()
    add_subdirectory(${MESHFEM_EXTERNAL}/TinyAD ${CMAKE_BINARY_DIR}/3rdparty/TinyAD)
endif()
