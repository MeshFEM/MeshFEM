# Safely reuse repositories expected to be edited locally. Existing checkouts
# are only fast-forwarded when clean and behind the requested revision; all
# other local state is preserved and hidden from FetchContent's update logic.

function(meshfem_prepare_editable_repo name source_dir revision)
    string(TOUPPER "${name}" name_upper)
    set(override_var "FETCHCONTENT_SOURCE_DIR_${name_upper}")

    # A cached override was explicitly supplied by the user.
    get_property(user_override CACHE "${override_var}" PROPERTY TYPE SET)
    if(user_override)
        return()
    endif()

    # Avoid processing the same checkout more than once per configuration.
    if(DEFINED ${override_var} AND NOT "${${override_var}}" STREQUAL "")
        return()
    endif()

    # Allow FetchContent to perform the initial clone.
    if(NOT EXISTS "${source_dir}/CMakeLists.txt")
        return()
    endif()

    # Existing non-Git sources are always externally managed.
    if(NOT EXISTS "${source_dir}/.git")
        message(STATUS "Using existing non-Git ${name} source: ${source_dir}")
        set(${override_var} "${source_dir}" PARENT_SCOPE)
        return()
    endif()

    find_package(Git REQUIRED)

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" status --porcelain
        WORKING_DIRECTORY "${source_dir}"
        OUTPUT_VARIABLE status
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE status_result
    )
    if(status_result OR status)
        message(STATUS "Using locally modified ${name} checkout: ${source_dir}")
        set(${override_var} "${source_dir}" PARENT_SCOPE)
        return()
    endif()

    # Fetching updates Git's object database but does not modify the worktree.
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" fetch origin "${revision}"
        WORKING_DIRECTORY "${source_dir}"
        RESULT_VARIABLE fetch_result
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(fetch_result)
        message(WARNING
            "Could not fetch ${name} revision ${revision}; using existing checkout")
        set(${override_var} "${source_dir}" PARENT_SCOPE)
        return()
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${source_dir}"
        OUTPUT_VARIABLE head
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE head_result
    )
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse FETCH_HEAD
        WORKING_DIRECTORY "${source_dir}"
        OUTPUT_VARIABLE requested
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE requested_result
    )

    if(head_result OR requested_result)
        message(WARNING "Could not inspect ${name}; using existing checkout")
    elseif(head STREQUAL requested)
        message(STATUS "${name} is already at ${revision}")
    else()
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" merge-base --is-ancestor "${head}" "${requested}"
            WORKING_DIRECTORY "${source_dir}"
            RESULT_VARIABLE can_fast_forward
        )
        if(can_fast_forward EQUAL 0)
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" merge --ff-only "${requested}"
                WORKING_DIRECTORY "${source_dir}"
                RESULT_VARIABLE update_result
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(update_result)
                message(WARNING
                    "Could not fast-forward ${name}; using existing checkout")
            else()
                message(STATUS "Fast-forwarded ${name} to ${revision}")
            endif()
        else()
            message(STATUS
                "Preserving ${name} at ${head}: it contains local or divergent "
                "commits relative to ${revision}")
        endif()
    endif()

    # Prevent FetchContent from modifying this worktree during this configure.
    set(${override_var} "${source_dir}" PARENT_SCOPE)
endfunction()
