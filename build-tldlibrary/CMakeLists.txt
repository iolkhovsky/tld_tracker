cmake_minimum_required(VERSION 3.10)

project(TldTrackerLib)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(VARIABLE TLD_SOURCES)
set(TLD_SOURCES 
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/augmentator.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/fern.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/fern_fext.cpp
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/integrator.cpp
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/object_detector.cpp
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/object_model.cpp
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/opt_flow_tracker.cpp
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/scanning_grid.cpp
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/tld_tracker.cpp
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/tld_utils.cpp)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker" ${OpenCV_INCLUDE_DIRS})
add_library(tldtracker SHARED ${TLD_SOURCES})
target_link_libraries(tldtracker ${OpenCV_LIBS})
target_include_directories(tldtracker PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker")

