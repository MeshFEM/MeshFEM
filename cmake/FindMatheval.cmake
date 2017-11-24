################################################################################
# - Find matheval library
# Find the native matheval includes and library
# This module defines
#  MATHEVAL::MATHEVAL. imported target to the matheval library
#  MATHEVAL_INCLUDE_DIRS, where to find matheval.h, Set when
#                      MATHEVAL_INCLUDE_DIR is found.
#  MATHEVAL_LIBRARIES, libraries to link against to use matheval.
#  MATHEVAL_ROOT_DIR, the base directory to search for matheval.
#                  This can also be an environment variable.
#  MATHEVAL_FOUND, If false, do not try to use matheval.
#
# also defined, but not for general use are
#  MATHEVAL_LIBRARY, where to find the matheval library.
################################################################################

# If MATHEVAL_ROOT_DIR was defined in the environment, use it.
if(NOT MATHEVAL_ROOT_DIR AND NOT $ENV{MATHEVAL_ROOT_DIR} STREQUAL "")
	set(MATHEVAL_ROOT_DIR $ENV{MATHEVAL_ROOT_DIR})
endif()

# Hard-coded guesses
set(_MATHEVAL_SEARCH_DIRS
	${THIRD_PARTY_DIR}/matheval
	${MESHFEM_THIRD_PARTY_DIR}/matheval
	${MATHEVAL_ROOT_DIR}
	/usr/local
	/sw # Fink
	/opt/local # DarwinPorts
	/opt/csw # Blastwave
	/opt/lib/MATHEVAL
)

find_path(MATHEVAL_INCLUDE_DIR
	NAMES
		matheval.h
	HINTS
		${_MATHEVAL_SEARCH_DIRS}
	PATHS
		$ENV{LIBMATHEVAL_INC}
	PATH_SUFFIXES
		include
)

find_library(MATHEVAL_LIBRARY
	NAMES
		matheval
	HINTS
		${_MATHEVAL_SEARCH_DIRS}
	PATHS
		$ENV{LIBMATHEVAL_LIB}
	PATH_SUFFIXES
		lib64 lib
)

# handle the QUIETLY and REQUIRED arguments and set MATHEVAL_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MATHEVAL DEFAULT_MSG
	MATHEVAL_LIBRARY MATHEVAL_INCLUDE_DIR)

if(MATHEVAL_FOUND)
	set(MATHEVAL_LIBRARIES ${MATHEVAL_LIBRARY})
	set(MATHEVAL_INCLUDE_DIRS ${MATHEVAL_INCLUDE_DIR})
endif()

mark_as_advanced(
	MATHEVAL_INCLUDE_DIR
	MATHEVAL_LIBRARY
)

# Imported target for Matheval
if(NOT TARGET Matheval::matheval)
	add_library(Matheval::matheval UNKNOWN IMPORTED)

	# Interface include directory
	set_target_properties(Matheval::matheval PROPERTIES
			INTERFACE_INCLUDE_DIRECTORIES "${MATHEVAL_INCLUDE_DIRS}")

	# Link to library file
	set_target_properties(Matheval::matheval PROPERTIES
			IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
			IMPORTED_LOCATION "${MATHEVAL_LIBRARIES}")
endif()
