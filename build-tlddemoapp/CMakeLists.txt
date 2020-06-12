cmake_minimum_required(VERSION 3.10)

project(TldDemoApp)
find_package(OpenCV REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(VARIABLE APP_SOURCES)
set(APP_SOURCES 
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/cmdline_parser.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/unit_tests.cpp)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker" ${OpenCV_INCLUDE_DIRS})
add_executable(tld_demo_app ${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/main.cpp ${APP_SOURCES})
target_link_libraries(tld_demo_app ${OpenCV_LIBS})
target_link_libraries(tld_demo_app tldtracker)

