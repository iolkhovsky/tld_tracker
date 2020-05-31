cmake_minimum_required(VERSION 3.10)

project(TldDemoApp)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker" ${OpenCV_INCLUDE_DIRS})
add_executable(tld_demo_app ${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/main.cpp ${TLD_SOURCES})
target_link_libraries(tld_demo_app ${OpenCV_LIBS})
target_link_libraries(tld_demo_app tldtracker)

