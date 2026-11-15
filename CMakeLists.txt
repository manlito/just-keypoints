cmake_minimum_required(VERSION 3.16)
project(just_keypoints VERSION 0.1.0 LANGUAGES CXX)

option(JKP_EMBED_WEIGHTS "Compile the model weights into the library" ON)
option(JKP_BUILD_CLI "Build the jkp command line tool" ON)
option(JKP_BUILD_TESTS "Build tests" ON)
set(JKP_EMBED_MODELS "aliked-n16" CACHE STRING "Models to embed when JKP_EMBED_WEIGHTS is ON (names in weights/)")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Weights: either compiled in, or left out (then a path or memory blob must be given at run time).
set(embed_models "")
if(JKP_EMBED_WEIGHTS)
  foreach(model IN LISTS JKP_EMBED_MODELS)
    list(APPEND embed_models "${model}=${CMAKE_CURRENT_SOURCE_DIR}/weights/${model}.jkpw")
    list(APPEND embed_deps "${CMAKE_CURRENT_SOURCE_DIR}/weights/${model}.jkpw")
  endforeach()
endif()
set(embed_cpp "${CMAKE_CURRENT_BINARY_DIR}/embedded_weights.cpp")
string(REPLACE ";" "\;" embed_models_arg "${embed_models}")
add_custom_command(
  OUTPUT "${embed_cpp}"
  COMMAND ${CMAKE_COMMAND} "-DOUT=${embed_cpp}" "-DMODELS=${embed_models_arg}" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake"
  DEPENDS ${embed_deps} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake"
  VERBATIM)

add_library(jkp
  src/core/api.cpp
  src/core/weights.cpp
  src/models/aliked/aliked.cpp
  src/backend/cpu/cpu_ops.cpp
  "${embed_cpp}")
target_include_directories(jkp PUBLIC include PRIVATE src)
find_package(Threads REQUIRED)
target_link_libraries(jkp PRIVATE Threads::Threads)
if(NOT MSVC)
  # -ffp-contract=off: keep float32 results independent of FMA availability (see PLAN.md, notes from M1).
  target_compile_options(jkp PRIVATE -Wall -Wextra -ffp-contract=off)
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
  foreach(golden IN LISTS golden_files)
    get_filename_component(name "${golden}" NAME_WE)
    add_test(NAME golden_${name}
             COMMAND golden_test "${CMAKE_CURRENT_SOURCE_DIR}/weights/aliked-n16.jkpw" "${golden}")
  endforeach()
endif()
