# Umfpack lib usually requires linking to a blas library.
# It is up to the user of this module to find a BLAS and link to it.

if(UMFPACK_INCLUDES AND UMFPACK_LIBRARIES)
  set(UMFPACK_FIND_QUIETLY TRUE)
endif(UMFPACK_INCLUDES AND UMFPACK_LIBRARIES)

find_path(UMFPACK_INCLUDES
  NAMES
  umfpack.h
  HINTS
  $ENV{UMFPACKDIR}
  $ENV{SUITESPARSE_ROOT}/include
  $ENV{SUITESPARSE_INC}
  ${INCLUDE_INSTALL_DIR}
  PATH_SUFFIXES
  suitesparse
  ufsparse
)

find_library(UMFPACK_LIBRARIES umfpack
    HINTS
    $ENV{UMFPACKDIR}
    $ENV{SUITESPARSE_ROOT}/lib
    $ENV{SUITESPARSE_LIB}
    ${LIB_INSTALL_DIR}
)

if(UMFPACK_LIBRARIES)

  if (NOT UMFPACK_LIBDIR)
    get_filename_component(UMFPACK_LIBDIR ${UMFPACK_LIBRARIES} PATH)
  endif(NOT UMFPACK_LIBDIR)

  find_library(COLAMD_LIBRARY colamd PATHS ${UMFPACK_LIBDIR} $ENV{UMFPACKDIR} ${LIB_INSTALL_DIR})
  if (COLAMD_LIBRARY)
    set(UMFPACK_LIBRARIES ${UMFPACK_LIBRARIES} ${COLAMD_LIBRARY})
  endif (COLAMD_LIBRARY)

  find_library(AMD_LIBRARY amd PATHS ${UMFPACK_LIBDIR} $ENV{UMFPACKDIR} ${LIB_INSTALL_DIR})
  if (AMD_LIBRARY)
    set(UMFPACK_LIBRARIES ${UMFPACK_LIBRARIES} ${AMD_LIBRARY})
  endif (AMD_LIBRARY)

  find_library(SUITESPARSE_LIBRARY SuiteSparse PATHS ${UMFPACK_LIBDIR} $ENV{UMFPACKDIR} ${LIB_INSTALL_DIR})
  if (SUITESPARSE_LIBRARY)
    set(UMFPACK_LIBRARIES ${UMFPACK_LIBRARIES} ${SUITESPARSE_LIBRARY})
  endif (SUITESPARSE_LIBRARY)

endif(UMFPACK_LIBRARIES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(UMFPACK DEFAULT_MSG
                                  UMFPACK_INCLUDES UMFPACK_LIBRARIES)

mark_as_advanced(UMFPACK_INCLUDES UMFPACK_LIBRARIES AMD_LIBRARY COLAMD_LIBRARY SUITESPARSE_LIBRARY)

if(UMFPACK_FOUND AND NOT TARGET umfpack::umfpack)
  if(${CMAKE_VERSION} VERSION_LESS "3.11.0")
    add_library(umfpack_umfpack INTERFACE)
    target_include_directories(umfpack_umfpack SYSTEM INTERFACE ${UMFPACK_INCLUDES})
    target_link_libraries(umfpack_umfpack INTERFACE "${UMFPACK_LIBRARIES}")
    add_library(umfpack::umfpack ALIAS umfpack_umfpack)
  else()
    add_library(umfpack::umfpack INTERFACE IMPORTED)
    target_include_directories(umfpack::umfpack SYSTEM INTERFACE ${UMFPACK_INCLUDES})
    target_link_libraries(umfpack::umfpack INTERFACE "${UMFPACK_LIBRARIES}")
  endif()
endif()
