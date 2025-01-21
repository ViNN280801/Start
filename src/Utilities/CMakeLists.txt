# src/Utilities/CMakeLists.txt

set(UTILITIES_SOURCES
    ConfigParser.cpp
    GmshUtils.cpp
    Timer.cpp
    Utilities.cpp
)

set(LIB_UTILITIES Utilities)
add_library(${LIB_UTILITIES} ${UTILITIES_SOURCES})

target_link_libraries(${LIB_UTILITIES} PRIVATE
    ${Boost_LIBRARIES}
    ${TBB_LIBRARIES}
)

target_include_directories(${LIB_UTILITIES} PRIVATE
    ${Boost_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS})
