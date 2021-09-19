if(TARGET boost)
    return()
endif()

message(STATUS "Third-party: creating target 'boost'")

include(FetchContent)
FetchContent_Declare(
    boost-cmake
    GIT_REPOSITORY https://github.com/Orphis/boost-cmake.git
    GIT_TAG 40cb41d86eab0d7fdc18af4b04b733f8cc852d2a
)

# This guy will download boost using FetchContent
FetchContent_MakeAvailable(boost-cmake)
