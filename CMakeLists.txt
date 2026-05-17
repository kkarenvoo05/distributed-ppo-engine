cmake_minimum_required(VERSION 3.22)

project(distributed_ppo_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "70;80;89;90")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(ppo_kernels SHARED
  src/gae_kernel.cu
  src/normalize_kernel.cu
  src/gather_kernel.cu
  src/ppo_kernels.cpp
)

target_include_directories(ppo_kernels PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(ppo_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(ppo_kernels PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF
)

add_executable(gae_test src/gae_test.cpp)
target_link_libraries(gae_test PRIVATE ppo_kernels)
target_include_directories(gae_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_smoke tests/test_smoke.cu)
target_link_libraries(test_smoke PRIVATE ppo_kernels)
target_include_directories(test_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(test_smoke PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>)

add_executable(bench_gae benchmarks/bench_gae.cu)
target_link_libraries(bench_gae PRIVATE ppo_kernels)
target_include_directories(bench_gae PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(bench_gae PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>)

