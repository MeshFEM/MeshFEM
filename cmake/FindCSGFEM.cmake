################################################################################
# Find CSGFEM
# The following variables are set
#
# CSGFEM_FOUND
# CSGFEM_INCLUDE_DIRS
#
# It searches the environment variable $CSGFEM_INC
################################################################################

find_path(CSGFEM_INCLUDE
		CSGFile.hh
		HINTS
			${PROJECT_SOURCE_DIR}/../CSGFEM
		PATHS
			ENV CSGFEM_INC
			${THIRD_PARTY_DIR}/CSGFEM
			${PROJECT_SOURCE_DIR}/../CSGFEM
			"C:/Program Files/CSGFEM/"
		PATH_SUFFIXES include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CSGFEM DEFAULT_MSG CSGFEM_INCLUDE)

if(CSGFEM_FOUND)
	set(CSGFEM_INCLUDE_DIRS ${CSGFEM_INCLUDE})
endif()

mark_as_advanced(CSGFEM_INCLUDE_DIRS)

# Interface target for CSGFEM
if(NOT TARGET CSGFEM)
	add_library(CSGFEM INTERFACE)
	target_include_directories(CSGFEM SYSTEM INTERFACE ${CSGFEM_INCLUDE_DIRS})
endif()
