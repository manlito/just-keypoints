cmake_minimum_required(VERSION 3.16...3.31)
project(just_keypoints VERSION 0.1.0 LANGUAGES CXX)

option(JKP_EMBED_WEIGHTS "Compile the model weights into the library" ON)
option(JKP_BUILD_CLI "Build the jkp command line tool" ON)
option(JKP_BUILD_TESTS "Build tests" ON)
set(JKP_CUDA "AUTO" CACHE STRING "CUDA backend: AUTO (build it when a CUDA compiler is found), ON or OFF")
set_property(CACHE JKP_CUDA PROPERTY STRINGS AUTO ON OFF)
set(JKP_EMBED_MODELS "aliked-n16" CACHE STRING "Models to embed when JKP_EMBED_WEIGHTS is ON (names in weights/)")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA backend: needs only the CUDA toolkit (nvcc and its runtime library).
if(NOT JKP_CUDA STREQUAL "OFF")
  if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    # Toolkit installed but not on PATH (common outside login shells).
    find_program(JKP_NVCC nvcc HINTS ENV CUDA_PATH PATHS /opt/cuda /usr/local/cuda PATH_SUFFIXES bin)
    if(JKP_NVCC)
      set(CMAKE_CUDA_COMPILER "${JKP_NVCC}")
    endif()
  endif()
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(JKP_HAVE_CUDA ON)
  elseif(JKP_CUDA STREQUAL "ON")
    message(FATAL_ERROR "JKP_CUDA=ON but no CUDA compiler was found (set CUDACXX or CMAKE_CUDA_COMPILER)")
  endif()
endif()

# Weights: either compiled in, or left out (then a path or memory blob must be given at run time).
set(embed_models "")
if(JKP_EMBED_WEIGHTS)
  foreach(model IN LISTS JKP_EMBED_MODELS)
    list(APPEND embed_models "${model}=${CMAKE_CURRENT_SOURCE_DIR}/weights/${model}.jkpw")
    list(APPEND embed_deps "${CMAKE_CURRENT_SOURCE_DIR}/weights/${model}.jkpw")
  endforeach()
endif()
set(embed_cpp "${CMAKE_CURRENT_BINARY_DIR}/embedded_weights.cpp")
string(REPLACE ";" "|" embed_models_arg "${embed_models}")  # a list does not survive the command line
add_custom_command(
  OUTPUT "${embed_cpp}"
  COMMAND ${CMAKE_COMMAND} "-DOUT=${embed_cpp}" "-DMODELS=${embed_models_arg}" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake"
  DEPENDS ${embed_deps} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake"
  VERBATIM)

add_library(jkp
  src/core/api.cpp
  src/core/ops.cpp
  src/core/weights.cpp
  src/models/aliked/aliked.cpp
  src/models/raco/raco.cpp
  src/backend/cpu/cpu_ops.cpp
  "${embed_cpp}")
target_include_directories(jkp PUBLIC include PRIVATE src)
find_package(Threads REQUIRED)
target_link_libraries(jkp PRIVATE Threads::Threads)
if(NOT MSVC)
  # -ffp-contract=off: keep float32 results independent of FMA availability (see PLAN.md, notes from M1).
  target_compile_options(jkp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-ffp-contract=off>)
endif()

if(JKP_HAVE_CUDA)
  target_sources(jkp PRIVATE src/backend/cuda/cuda_ops.cu)
  target_compile_definitions(jkp PRIVATE JKP_HAVE_CUDA)
  set_target_properties(jkp PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  # --fmad=false: no fused multiply-add in kernels, so they compute what the CPU backend computes.
  target_compile_options(jkp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  message(STATUS "jkp backends: cpu, cuda")
else()
  message(STATUS "jkp backends: cpu")
endif()

if(JKP_BUILD_CLI)
  add_executable(jkp_cli cli/main.cpp)
  set_target_properties(jkp_cli PROPERTIES OUTPUT_NAME jkp)
  target_include_directories(jkp_cli PRIVATE src)
  target_link_libraries(jkp_cli PRIVATE jkp)
endif()

if(JKP_BUILD_TESTS)
  enable_testing()
  add_executable(golden_test tests/golden_test.cpp)
  target_include_directories(golden_test PRIVATE src)
  target_link_libraries(golden_test PRIVATE jkp)
  # Golden files are generated locally: python tools/dump_golden.py <images>
  file(GLOB golden_files "${CMAKE_CURRENT_SOURCE_DIR}/build/golden/*.jkpt")
  set(test_backends cpu)
  if(JKP_HAVE_CUDA)
    list(APPEND test_backends cuda)
    target_compile_definitions(golden_test PRIVATE JKP_TEST_CUDA)
    # CUDA against the CPU backend on generated images; needs no golden files.
    add_executable(backend_test tests/backend_test.cpp)
    target_include_directories(backend_test PRIVATE src)
    target_link_libraries(backend_test PRIVATE jkp)
    add_test(NAME cuda_vs_cpu
             COMMAND backend_test "${CMAKE_CURRENT_SOURCE_DIR}/weights/aliked-n16.jkpw" cuda)
  endif()
  foreach(golden IN LISTS golden_files)
    get_filename_component(name "${golden}" NAME_WLE)
    foreach(backend IN LISTS test_backends)
      add_test(NAME golden_${backend}_${name}
               COMMAND golden_test "${CMAKE_CURRENT_SOURCE_DIR}/weights/aliked-n16.jkpw" "${golden}" ${backend})
    endforeach()
  endforeach()

  # RaCo: python tools/dump_golden.py --model raco <images>
  add_executable(raco_golden_test tests/raco_golden_test.cpp)
  target_include_directories(raco_golden_test PRIVATE src)
  target_link_libraries(raco_golden_test PRIVATE jkp)
  file(GLOB raco_golden_files "${CMAKE_CURRENT_SOURCE_DIR}/build/golden/raco/*.jkpt")
  foreach(golden IN LISTS raco_golden_files)
    get_filename_component(name "${golden}" NAME_WLE)
    add_test(NAME golden_raco_cpu_${name}
             COMMAND raco_golden_test "${CMAKE_CURRENT_SOURCE_DIR}/weights/raco.jkpw" "${golden}")
  endforeach()
endif()
