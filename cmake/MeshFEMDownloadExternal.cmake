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
        URL     https://github.com/catchorg/Catch2/archive/v2.3.0.tar.gz
        URL_MD5 1fc90ff3b7b407b83057537f4136489e
    )
endfunction()

## Eigen
function(meshfem_download_eigen)
    meshfem_download_project(eigen
        URL     http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2
        URL_MD5 a7aab9f758249b86c93221ad417fbe18
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
        GIT_TAG        4c3ffe5a5f37addef0dd6283c74c4402a3b4ebc9
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
        GIT_TAG        d6761dd691e2e1318c83bf7773fea88d9437464a
    )
endfunction()
