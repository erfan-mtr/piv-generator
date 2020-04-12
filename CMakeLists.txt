cmake_minimum_required(VERSION 2.8) 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
PROJECT (pivgenerator)
find_package(OpenCV REQUIRED)
find_package(Python2 COMPONENTS Development NumPy)
set( NAME_SRC
    main.cpp
)

set( NAME_HEADERS       
     # include/header.h
)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
link_directories( ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
add_executable( pivgenerator ${NAME_SRC} ${NAME_HEADERS} )

target_link_libraries( pivgenerator ${OpenCV_LIBS} Python2::Python Python2::NumPy    )
