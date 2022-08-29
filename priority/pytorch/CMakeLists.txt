cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(jm-test)

# include_directories(~/joo/HGD_project/pytorch/include)
# include_directories(~/.local/lib/python3.6/site-packages/torch/include/torch/csrc/api/include)
# include_directories(~/.local/lib/python3.6/site-packages/torch/include)

# link_directories(/usr/local/lib)
# link_directories(~/.local/lib/python3.6/site-packages/torch/lib)


# set(CMAKE_PREFIX_PATH /home/kmsjames/very-big-storage/hojin/pytorch/newlibtorch/libtorch)
# set(CMAKE_PREFIX_PATH /home/kmsjames/very-big-storage/hojin/pytorch/perfisol/libtorch)
set(CMAKE_PREFIX_PATH ~/.local/lib/python3.6/site-packages/torch)
find_package(Torch REQUIRED)

set(COMMON_DIR /home/nvidia/joo/HGD_project/priority/pytorch/include)

file(GLOB SOURCES "src/*")
#set(SOURCES src/multithreadtest.cpp) #manual add

include_directories(${COMMON_DIR})

add_executable(jm-test ${SOURCES})
target_link_libraries(jm-test "${TORCH_LIBRARIES}")
set_property(TARGET jm-test PROPERTY CXX_STANDARD 14)
