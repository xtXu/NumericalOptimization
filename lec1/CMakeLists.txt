cmake_minimum_required(VERSION 3.5)

project(project LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)

include_directories(include)

add_library(lsgd src/lsgd.cpp)
target_link_libraries(lsgd Eigen3::Eigen)

add_executable(hello src/test.cpp)
target_link_libraries(hello Eigen3::Eigen)

add_executable(main src/main.cpp)
target_link_libraries(main Eigen3::Eigen lsgd)
