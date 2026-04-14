cmake_minimum_required(VERSION 3.16)
project(hs_engine_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find pybind11 (installed via pip)
execute_process(
    COMMAND python -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

# Include headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
# effects.cpp is auto-generated: python scripts/generate_cpp_effects.py
set(ENGINE_SOURCES
    src/event_system.cpp
    src/generated_effects.cpp
    src/auras.cpp
    src/combat.cpp
)

# pybind11 module
pybind11_add_module(hs_engine_cpp
    bindings/pybind_module.cpp
    ${ENGINE_SOURCES}
)

# Compiler flags
if(MSVC)
    target_compile_options(hs_engine_cpp PRIVATE /O2 /W4)
else()
    target_compile_options(hs_engine_cpp PRIVATE -O2 -g -Wall -Wextra -mbmi2 -mpopcnt)
endif()
