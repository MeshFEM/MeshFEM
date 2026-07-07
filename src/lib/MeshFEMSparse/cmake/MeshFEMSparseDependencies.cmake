# Prepare dependencies owned by MeshFEMSparse.

if(NOT DEFINED MESHFEMSPARSE_ROOT)
    get_filename_component(MESHFEMSPARSE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

if(NOT DEFINED MESHFEM_EXTERNAL)
    set(MESHFEM_EXTERNAL "${MESHFEMSPARSE_ROOT}/3rdparty")
endif()

get_directory_property(hasParent PARENT_DIRECTORY)
if (hasParent)
    set(MESHFEMSPARSE_ROOT "${MESHFEMSPARSE_ROOT}" PARENT_SCOPE)
    set(MESHFEM_EXTERNAL "${MESHFEM_EXTERNAL}" PARENT_SCOPE)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)

include(MeshFEMCoreDependencies)
include(MeshFEMSparseDownloadExternal)

if(NOT TARGET Catch2::Catch2 AND MESHFEMSPARSE_BUILD_TESTS)
    meshfem_download_catch()
    add_subdirectory(${MESHFEM_EXTERNAL}/Catch2 ${CMAKE_BINARY_DIR}/3rdparty/Catch2)
    list(APPEND CMAKE_MODULE_PATH ${MESHFEM_EXTERNAL}/Catch2/contrib)
endif()

if (MESHFEM_WITH_CHOLMOD)
    find_package(CHOLMOD REQUIRED) # provides cholmod::cholmod
endif()

if (MESHFEM_WITH_UMFPACK)
    find_package(UMFPACK REQUIRED) # provides umfpack::umfpack
endif()

if (MESHFEM_WITH_CATAMARI AND NOT TARGET catamari)
    if (MESHFEM_USE_LEGACY_CATAMARI)
        meshfem_download_catamari_legacy()
        add_subdirectory(${MESHFEM_EXTERNAL}/catamari_legacy ${CMAKE_BINARY_DIR}/3rdparty/catamari_legacy)
    else()
        if (NOT EXISTS ${MESHFEM_EXTERNAL}/catamari/CMakeLists.txt)
            message(FATAL_ERROR "MESHFEM_WITH_CATAMARI is enabled, but ${MESHFEM_EXTERNAL}/catamari is missing.")
        endif()
        add_subdirectory(${MESHFEM_EXTERNAL}/catamari ${CMAKE_BINARY_DIR}/3rdparty/catamari)
    endif()
endif()

if (MESHFEM_WITH_SCOTCH)
    find_package(SCOTCH QUIET)
    if (NOT TARGET SCOTCH::scotch)
        message(STATUS "Scotch not found; support will be disabled")
        set(MESHFEM_WITH_SCOTCH OFF)
    endif()
endif()
