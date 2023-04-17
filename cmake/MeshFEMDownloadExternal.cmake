################################################################################
include(DownloadProject)

# With CMake 3.8 and above, we can hide warnings about git being in a
# detached head by passing an extra GIT_CONFIG option
if(NOT (${CMAKE_VERSION} VERSION_LESS "3.8.0"))
    set(MESHFEM_EXTRA_OPTIONS "GIT_CONFIG advice.detachedHead=false")
else()
    set(MESHFEM_EXTRA_OPTIONS "")
endif()

# Shortcut function
function(meshfem_download_project name)
    download_project(
        PROJ         ${name}
        SOURCE_DIR   ${MESHFEM_EXTERNAL}/${name}
        DOWNLOAD_DIR ${MESHFEM_EXTERNAL}/.cache/${name}
        QUIET
        ${MESHFEM_EXTRA_OPTIONS}
        ${ARGN}
    )
endfunction()

################################################################################

## Catch2
function(meshfem_download_catch)
    meshfem_download_project(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG 182c910b4b63ff587a3440e08f84f70497e49a81 # Version 2.13.10
    )
endfunction()

## Ceres
function(meshfem_download_ceres)
    meshfem_download_project(ceres
        GIT_REPOSITORY https://github.com/MeshFEM/ceres-solver.git
        GIT_TAG        114ace92eec013192f156d5452551f68695ac86e
    )
endfunction()

## Eigen
function(meshfem_download_eigen)
    meshfem_download_project(eigen
        URL     https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
        URL_MD5 9e30f67e8531477de4117506fe44669b
    )
endfunction()

## Json
function(meshfem_download_json)
    meshfem_download_project(json
        URL      https://github.com/nlohmann/json/releases/download/v3.1.2/include.zip
        URL_HASH SHA256=495362ee1b9d03d9526ba9ccf1b4a9c37691abe3a642ddbced13e5778c16660c
    )
endfunction()

## Optional
function(meshfem_download_optional)
    meshfem_download_project(optional
        URL     https://github.com/martinmoene/optional-lite/archive/v3.0.0.tar.gz
        URL_MD5 a66541380c51c0d0a1e593cc2ca9fe8a
    )
endfunction()

## TBB
function(meshfem_download_tbb)
    meshfem_download_project(tbb
        GIT_REPOSITORY https://github.com/wjakob/tbb.git
        GIT_TAG        141b0e310e1fb552bdca887542c9c1a8544d6503
    )
endfunction()

## Tinyexpr
function(meshfem_download_tinyexpr)
    meshfem_download_project(tinyexpr
        GIT_REPOSITORY https://github.com/codeplea/tinyexpr.git
        GIT_TAG        4e8cc0067a1e2378faae23eb2dfdd21e9e9907c2
    )
endfunction()

## Triangle
function(meshfem_download_triangle)
    meshfem_download_project(triangle
        GIT_REPOSITORY https://github.com/libigl/triangle.git
        GIT_TAG        d6761dd691e2e1318c83bf7773fea88d9437464a
    )
endfunction()

## Spectra
function(meshfem_download_spectra)
    meshfem_download_project(spectra
        GIT_REPOSITORY https://github.com/yixuan/spectra.git
        GIT_TAG        ec27cfd2210a9b2322825c4cb8e5d47f014e1ac3
    )
endfunction()
