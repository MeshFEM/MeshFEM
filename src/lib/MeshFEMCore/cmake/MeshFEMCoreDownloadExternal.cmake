################################################################################
include(FetchContent)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# Shortcut function
function(meshfem_download_project name)
    set(source_dir "${MESHFEM_EXTERNAL}/${name}")
    if(EXISTS "${source_dir}")
        file(GLOB source_dir_contents LIST_DIRECTORIES true "${source_dir}/*")
        if(source_dir_contents)
            return()
        endif()
    endif()

    FetchContent_Populate(${name}
        SOURCE_DIR ${source_dir}
        ${ARGN}
    )
endfunction()

################################################################################

## Eigen
function(meshfem_download_eigen)
    meshfem_download_project(eigen
        # URL     https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
        # URL_MD5 9e30f67e8531477de4117506fe44669b
        URL     https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        URL_MD5 4c527a9171d71a72a9d4186e65bea559
    )
endfunction()

## TBB
function(meshfem_download_tbb)
    meshfem_download_project(tbb
        GIT_REPOSITORY https://github.com/oneapi-src/oneTBB
        GIT_TAG        0c0ff192a2304e114bc9e6557582dfba101360ff # v2022.0.0 from Oct 31, 2024
    )
endfunction()
