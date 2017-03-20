# First, try to find Eigen3Config.cmake; this should work with Eigen 3.3 or
# newer when installed in a standard location.
find_package(Eigen3 QUIET NO_MODULE)

# For older versions of eigen, or when the above fails, we must do our own
# search guided by environment vars.
if(NOT Eigen3_FOUND)
    include (FindPackageHandleStandardArgs)
    # Finds the include files directory
    find_path(EIGEN_INCLUDE_DIRS
        NAMES Eigen/Core
        DOC "The directory where Eigen/Core resides"
        HINTS "${EIGEN_ROOT_DIR}"
        PATHS
        $ENV{EIGEN_INC}
        $ENV{EIGEN_PATH}
        /opt/local/include
        /usr/local/include
        /usr/include
        PATH_SUFFIXES eigen3
        NO_DEFAULT_PATH
    )

    if(EIGEN_INCLUDE_DIRS)
        mark_as_advanced(EIGEN_INCLUDE_DIRS)
    endif()

    find_package_handle_standard_args(Eigen DEFAULT_MSG EIGEN_INCLUDE_DIRS)
endif()

