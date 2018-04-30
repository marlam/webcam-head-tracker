# Copyright (C) 2017 Martin Lambers <marlam@marlam.de>
#
# Copying and distribution of this file, with or without modification, are
# permitted in any medium without royalty provided the copyright notice and this
# notice are preserved. This file is offered as-is, without any warranty.

cmake_minimum_required(VERSION 3.4)

project(libwebcamheadtracker)
set(WEBCAMHEADTRACKER_VERSION 1.0.0)
set(WEBCAMHEADTRACKER_LIBVERSION 0.0.0)
set(WEBCAMHEADTRACKER_SOVERSION 0)

option(WEBCAMHEADTRACKER_BUILD_DOCUMENTATION "Build API reference documentation (requires Doxygen)" OFF)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -Wall -Wextra")
endif()

find_package(OpenCV REQUIRED highgui imgproc objdetect calib3d video)
include_directories(${OPENCV_INCLUDE_DIRS} )
link_directories(${OPENCV_LIBRARY_DIRS})
add_definitions(${OPENCV_DEFINITIONS})
string(TOLOWER ${OpenCV_DIR} OpenCV_DIR_LOWER)
find_file(HAARCASCADE_FRONTALFACE_ALT_XML
    haarcascade_frontalface_alt.xml
    ${OpenCV_DIR}/haarcascades 
    ${OpenCV_DIR_LOWER}/haarcascades)
if(${HAARCASCADE_FRONTALFACE_ALT_XML} STREQUAL "HAARCASCADE_FRONTALFACE_ALT_XML-NOTFOUND")
    message(STATUS "haarcascade_frontalface_alt.xml not found -- will be required at run time")
else()
    add_definitions(-DHAARCASCADE_FRONTALFACE_ALT_XML=${HAARCASCADE_FRONTALFACE_ALT_XML})
endif()

find_package(dlib REQUIRED)
add_definitions(${dlib_DEFINITIONS})
find_file(SHAPE_PREDICTOR_68_FACE_LANDMARKS_DAT
    shape_predictor_68_face_landmarks.dat
    ${dlib_DIR}/../../../../share/dlib)
if(SHAPE_PREDICTOR_68_FACE_LANDMARKS_DAT STREQUAL "SHAPE_PREDICTOR_68_FACE_LANDMARKS_DAT-NOTFOUND")
    message(WARNING "shape_predictor_68_face_landmarks.dat not found -- will be required at run time")
else()
    add_definitions(-DSHAPE_PREDICTOR_68_FACE_LANDMARKS_DAT=${SHAPE_PREDICTOR_68_FACE_LANDMARKS_DAT})
endif()

add_library(libwebcamheadtracker SHARED
	webcam-head-tracker.cpp webcam-head-tracker.hpp)
set_target_properties(libwebcamheadtracker PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
set_target_properties(libwebcamheadtracker PROPERTIES OUTPUT_NAME webcamheadtracker)
set_target_properties(libwebcamheadtracker PROPERTIES VERSION ${WEBCAMHEADTRACKER_LIBVERSION})
set_target_properties(libwebcamheadtracker PROPERTIES SOVERSION ${WEBCAMHEADTRACKER_SOVERSION})
target_link_libraries(libwebcamheadtracker ${OpenCV_LIBS} dlib::dlib)
install(TARGETS libwebcamheadtracker
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib${LIB_SUFFIX}
    ARCHIVE DESTINATION lib${LIB_SUFFIX}
)
install(FILES
    webcam-head-tracker.hpp
    DESTINATION include/webcam-head-tracker)
include(CMakePackageConfigHelpers)
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/include)
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})
configure_package_config_file(
    WebcamHeadTrackerConfig.cmake.in ${CMAKE_BINARY_DIR}/WebcamHeadTrackerConfig.cmake
    INSTALL_DESTINATION lib${LIB_SUFFIX}/cmake/WebcamHeadTracker-{WEBCAMHEADTRACKER_VERSION}
    PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/WebcamHeadTrackerConfigVersion.cmake
    VERSION ${WEBCAMHEADTRACKER_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_BINARY_DIR}/WebcamHeadTrackerConfig.cmake
    ${CMAKE_BINARY_DIR}/WebcamHeadTrackerConfigVersion.cmake
    DESTINATION ${LIB_INSTALL_DIR}/cmake/WebcamHeadTracker-${WEBCAMHEADTRACKER_VERSION}
)

if(WEBCAMHEADTRACKER_BUILD_DOCUMENTATION)
  find_package(Doxygen REQUIRED)
  configure_file("${CMAKE_SOURCE_DIR}/Doxyfile.in" "${CMAKE_BINARY_DIR}/Doxyfile" @ONLY)
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/html")
  add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/html/index.html"
    COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_BINARY_DIR}/Doxyfile"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    DEPENDS "${CMAKE_SOURCE_DIR}/Doxyfile.in"
            "${CMAKE_SOURCE_DIR}/webcam-head-tracker.hpp"
    COMMENT "Generating API documentation with Doxygen" VERBATIM
  )
  add_custom_target(doc ALL DEPENDS "${CMAKE_BINARY_DIR}/html/index.html")
  install(DIRECTORY "${CMAKE_BINARY_DIR}/html" DESTINATION share/doc/libwebcamheadtracker)
endif()

add_executable(webcam-head-tracker-demo webcam-head-tracker-demo.cpp)
target_link_libraries(webcam-head-tracker-demo libwebcamheadtracker)
install(TARGETS webcam-head-tracker-demo RUNTIME DESTINATION bin)

add_executable(webcam-snapshots webcam-snapshots.cpp)
target_link_libraries(webcam-snapshots ${OpenCV_LIBS})
install(TARGETS webcam-snapshots RUNTIME DESTINATION bin)
