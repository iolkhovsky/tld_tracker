cmake_minimum_required(VERSION 3.10)

project(TldTrackerLib)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(VARIABLE TLD_SOURCES)
set(VARIABLE TLD_HEADERS)
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
set(TLD_HEADERS 
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/augmentator.h
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/common.h
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/fern.h
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/fern_fext.h
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/integrator.h
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/object_detector.h
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/object_classifier.h
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/object_model.h
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/opt_flow_tracker.h
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/scanning_grid.h
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/tld_tracker.h
${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker/tracker/tld_utils.h)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker" ${OpenCV_INCLUDE_DIRS})
add_library(tldtracker SHARED ${TLD_SOURCES})
target_link_libraries(tldtracker ${OpenCV_LIBS})
target_include_directories(tldtracker PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../tld_tracker")

