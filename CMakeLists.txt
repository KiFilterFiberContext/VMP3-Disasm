cmake_minimum_required(VERSION 3.9)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

project(vmp-disasm)
set (CMAKE_CXX_STANDARD 20)

include_directories("/usr/local/include" "/usr/lib/llvm-13/include" "include")

add_executable(vmp-disasm src/main.cpp src/arch.cpp src/vmp.cpp)
target_link_directories(vmp-disasm PUBLIC /usr/local/lib /usr/lib/llvm-13/lib)

target_link_libraries(vmp-disasm LLVM triton LIEF)
