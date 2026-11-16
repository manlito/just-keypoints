cmake_minimum_required(VERSION 3.16...3.31)
project(just_keypoints VERSION 0.1.0 LANGUAGES CXX)

option(JKP_EMBED_WEIGHTS "Compile the model weights into the library" ON)
option(JKP_BUILD_CLI "Build the jkp command line tool" ON)
option(JKP_BUILD_TESTS "Build tests" ON)
set(JKP_CUDA "AUTO" CACHE STRING "CUDA backend: AUTO (build it when a CUDA compiler is found), ON or OFF")
set_property(CACHE JKP_CUDA PROPERTY STRINGS AUTO ON OFF)
set(JKP_METAL "AUTO" CACHE STRING "Metal backend (Apple platforms): AUTO (build it when Metal.framework is found), ON or OFF")
set_property(CACHE JKP_METAL PROPERTY STRINGS AUTO ON OFF)
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

# Metal backend: needs only Metal.framework and Foundation. Shaders are compiled at run time, so no
# Metal toolchain is needed to build.
if(NOT JKP_METAL STREQUAL "OFF")
  if(APPLE)
    find_library(JKP_METAL_FRAMEWORK Metal)
    find_library(JKP_FOUNDATION_FRAMEWORK Foundation)
  endif()
  if(JKP_METAL_FRAMEWORK AND JKP_FOUNDATION_FRAMEWORK)
    enable_language(OBJCXX)
    set(JKP_HAVE_METAL ON)
  elseif(JKP_METAL STREQUAL "ON")
    message(FATAL_ERROR "JKP_METAL=ON but Metal.framework was not found (Apple platforms only)")
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
  target_compile_options(jkp PRIVATE $<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-Wall;-Wextra;-ffp-contract=off>)
endif()

if(JKP_HAVE_CUDA)
  target_sources(jkp PRIVATE src/backend/cuda/cuda_ops.cu)
  target_compile_definitions(jkp PRIVATE JKP_HAVE_CUDA)
  set_target_properties(jkp PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  # --fmad=false: no fused multiply-add in kernels, so they compute what the CPU backend computes.
  target_compile_options(jkp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
endif()

if(JKP_HAVE_METAL)
  set(metal_src "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal/kernels.metal")
  set(metal_cpp "${CMAKE_CURRENT_BINARY_DIR}/metal_kernels.cpp")
  add_custom_command(
    OUTPUT "${metal_cpp}"
    COMMAND ${CMAKE_COMMAND} "-DIN=${metal_src}" "-DOUT=${metal_cpp}" -DNAME=kJkpMetalSource
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake"
    DEPENDS "${metal_src}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake"
    VERBATIM)
  target_sources(jkp PRIVATE src/backend/metal/metal_ops.mm "${metal_cpp}")
  target_compile_definitions(jkp PRIVATE JKP_HAVE_METAL)
  target_compile_options(jkp PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(jkp PRIVATE "${JKP_METAL_FRAMEWORK}" "${JKP_FOUNDATION_FRAMEWORK}")
endif()

set(jkp_backends cpu)
if(JKP_HAVE_CUDA)
  list(APPEND jkp_backends cuda)
endif()
if(JKP_HAVE_METAL)
  list(APPEND jkp_backends metal)
endif()
string(REPLACE ";" ", " jkp_backends_text "${jkp_backends}")
message(STATUS "jkp backends: ${jkp_backends_text}")

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
  set(test_backends ${jkp_backends})
  if(JKP_HAVE_CUDA)
    target_compile_definitions(golden_test PRIVATE JKP_TEST_CUDA)
  endif()
  if(JKP_HAVE_METAL)
    target_compile_definitions(golden_test PRIVATE JKP_TEST_METAL)
  endif()
  # Each GPU backend against the CPU backend on generated images; needs no golden files.
  set(gpu_backends ${jkp_backends})
  list(REMOVE_ITEM gpu_backends cpu)
  if(gpu_backends)
    add_executable(backend_test tests/backend_test.cpp)
    target_include_directories(backend_test PRIVATE src)
    target_link_libraries(backend_test PRIVATE jkp)
    if(JKP_HAVE_CUDA)
      target_compile_definitions(backend_test PRIVATE JKP_TEST_CUDA)
    endif()
    if(JKP_HAVE_METAL)
      target_compile_definitions(backend_test PRIVATE JKP_TEST_METAL)
    endif()
    # FP16 against FP32 on the same backend: generated images plus the images of the golden files.
    add_executable(precision_test tests/precision_test.cpp)
    target_include_directories(precision_test PRIVATE src)
    target_link_libraries(precision_test PRIVATE jkp)
    if(JKP_HAVE_CUDA)
      target_compile_definitions(precision_test PRIVATE JKP_TEST_CUDA)
    endif()
    if(JKP_HAVE_METAL)
      target_compile_definitions(precision_test PRIVATE JKP_TEST_METAL)
    endif()
    foreach(backend IN LISTS gpu_backends)
      add_test(NAME ${backend}_vs_cpu
               COMMAND backend_test "${CMAKE_CURRENT_SOURCE_DIR}/weights/aliked-n16.jkpw" ${backend})
      add_test(NAME ${backend}_fp16_vs_fp32
               COMMAND precision_test "${CMAKE_CURRENT_SOURCE_DIR}/weights/aliked-n16.jkpw" ${backend} ${golden_files})
    endforeach()
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
