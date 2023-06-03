set(CMAKE_CXX_STANDAND 17)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(InferenceEngine REQUIRED)
include_directories(${InferenceEngine_INCLUDE_DIRS})

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w" )


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})


aux_source_directory(kalman_filter kalman_src)
aux_source_directory(detect_armour detect_src)

message("${kalman_src}")
add_library(modules SHARED ${kalman_src} ${detect_src} )

target_link_libraries(modules
    ${OpenCV_LIBS}
    fmt::fmt 
    ${InferenceEngine_LIBRARIES}
)
