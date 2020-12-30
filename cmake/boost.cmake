if(TARGET boost)
    return()
endif()

message(STATUS "Third-party: creating target 'boost'")

include(FetchContent)
FetchContent_Declare(
    boost-cmake
    GIT_REPOSITORY https://github.com/Orphis/boost-cmake.git
    GIT_TAG 70b12f62da331dd402b78102ec8f6a15d59a7af9
)

# This guy will download boost using FetchContent
FetchContent_MakeAvailable(boost-cmake)
