cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(test_exported_pt2)

execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED CONFIG)

add_executable(test_exported_pt2 test_exported_pt2.cpp)

target_link_libraries(test_exported_pt2 "${TORCH_LIBRARIES}")
set_property(TARGET test_exported_pt2 PROPERTY CXX_STANDARD 17)
