cmake_minimum_required(VERSION 3.15)
project(fit_module LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Find Python interpreter and development libraries
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

# 2. Ask Python where nanobind's CMake files live
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 3. Find nanobind
find_package(nanobind CONFIG REQUIRED)

# 4. Find System Eigen3
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# 5. Build module
nanobind_add_module(fit_module fit/trajectory.cpp)

# Fix here: Use the proper namespace target Eigen3::Eigen
target_link_libraries(fit_module PRIVATE Eigen3::Eigen)

