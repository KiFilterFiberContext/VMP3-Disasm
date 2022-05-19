cmake_minimum_required(VERSION 3.9)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

project(devirt)
set (CMAKE_CXX_STANDARD 20)

include_directories("/usr/local/include" "/usr/lib/llvm-13/include" "include")

add_executable(devirt src/main.cpp src/vmp.cpp src/ir.cpp)
target_link_directories(devirt PUBLIC /usr/local/lib /usr/lib/llvm-13/lib)

target_link_libraries(devirt LLVM triton LIEF)