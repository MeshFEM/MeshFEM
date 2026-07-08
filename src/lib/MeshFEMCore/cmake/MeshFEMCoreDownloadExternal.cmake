################################################################################
include(FetchContent)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# Shortcut function
function(meshfem_checkout_git_revision name source_dir revision)
    find_package(Git REQUIRED)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY "${source_dir}"
        OUTPUT_VARIABLE current_revision
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE rev_parse_result)
    if(rev_parse_result)
        message(FATAL_ERROR "Could not determine ${name} revision in ${source_dir}")
    endif()
    if(NOT current_revision STREQUAL revision)
        message(STATUS "Updating ${name} to ${revision}")
        execute_process(COMMAND ${GIT_EXECUTABLE} fetch origin ${revision}
            WORKING_DIRECTORY "${source_dir}"
            RESULT_VARIABLE fetch_result)
        if(fetch_result)
            message(FATAL_ERROR "Could not fetch ${name} revision ${revision}")
        endif()
        execute_process(COMMAND ${GIT_EXECUTABLE} checkout ${revision}
            WORKING_DIRECTORY "${source_dir}"
            RESULT_VARIABLE checkout_result)
        if(checkout_result)
            message(FATAL_ERROR "Could not check out ${name} revision ${revision}")
        endif()
    endif()
endfunction()

function(meshfem_download_project name)
    cmake_parse_arguments(ARG "" "GIT_TAG;SOURCE_DIR" "" ${ARGN})
    if(ARG_SOURCE_DIR)
        set(source_dir "${ARG_SOURCE_DIR}")
    else()
        set(source_dir "${MESHFEM_EXTERNAL}/${name}")
    endif()

    if(EXISTS "${source_dir}")
        file(GLOB source_dir_contents LIST_DIRECTORIES true "${source_dir}/*")
        if(source_dir_contents)
            if(ARG_GIT_TAG AND EXISTS "${source_dir}/.git")
                meshfem_checkout_git_revision(${name} "${source_dir}" ${ARG_GIT_TAG})
            endif()
            return()
        endif()
    endif()

    set(populate_args ${ARG_UNPARSED_ARGUMENTS})
    if(ARG_GIT_TAG)
        list(APPEND populate_args GIT_TAG ${ARG_GIT_TAG})
    endif()

    FetchContent_Populate(${name}
        SOURCE_DIR ${source_dir}
        ${populate_args}
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
