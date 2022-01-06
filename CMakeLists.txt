project(facebox)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_INSTALL_PREFIX /bin)
set(CMAKE_BUILD_TYPE Debug)
find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRECTORIES}
    include/
)
add_executable(main main.cpp src/detector.cpp src/priorbox.cpp)
target_link_libraries(main ${OpenCV_LIBS})
add_executable(test src/test.cpp)
target_link_libraries(test ${OpenCV_LIBS})
add_executable(test_onnx test_onnx.cpp)
target_link_libraries(test_onnx ${OpenCV_LIBS})
