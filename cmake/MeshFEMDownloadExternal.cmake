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
	URL     https://github.com/catchorg/Catch2/archive/v2.13.10.tar.gz
    URL_MD5 7a4dd2fd14fb9f46198eb670ac7834b7
    )
endfunction()

## Ceres
function(meshfem_download_ceres)
    meshfem_download_project(ceres
        GIT_REPOSITORY https://github.com/jdumas/ceres-solver.git
        GIT_TAG        2ba66a2c22959d9c455a8f2074dc7a605c4a92e8
    )
endfunction()

## Eigen
function(meshfem_download_eigen)
    meshfem_download_project(eigen
        URL     https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
        URL_MD5 9e30f67e8531477de4117506fe44669b
        # URL     https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        # URL_MD5 4c527a9171d71a72a9d4186e65bea559
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
        GIT_REPOSITORY https://github.com/oneapi-src/oneTBB
        GIT_TAG        c9497714821c3d443ee44c732609eb6850195ffb
    )
endfunction()

## Tinyexpr
function(meshfem_download_tinyexpr)
    meshfem_download_project(tinyexpr
        GIT_REPOSITORY https://github.com/codeplea/tinyexpr.git
        GIT_TAG        ffb0d41b13e5f8d318db95feb071c220c134fe70
    )
endfunction()

## Triangle
function(meshfem_download_triangle)
    meshfem_download_project(triangle
        GIT_REPOSITORY https://github.com/libigl/triangle.git
        GIT_TAG        3ee6cac2230f0fe1413879574f741c7b6da11221
    )
endfunction()

## Spectra
function(meshfem_download_spectra)
    meshfem_download_project(spectra
        GIT_REPOSITORY https://github.com/yixuan/spectra.git
        GIT_TAG        8c7242e08f0fb7f6a0022cfe232e3dc5b5bd4eb4
    )
endfunction()

## Catamari
function(meshfem_download_catamari)
    meshfem_download_project(catamari
        GIT_REPOSITORY https://github.com/jpanetta/catamari.git
        GIT_TAG        23b858e18f639261bae4a2fe15d00a286cda963d)
endfunction()
