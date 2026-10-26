cmake_minimum_required(VERSION 3.20)
project(scoring_ablation_experiment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# This isolated build reuses the engine targets without changing its build tree.
# All scoring and RMSD helpers needed here are host code.
set(ENABLE_CUDA OFF CACHE BOOL "This experiment only uses host scoring" FORCE)
add_subdirectory(../.. engine EXCLUDE_FROM_ALL)

add_library(scoring_ablation_support ablation.cpp)
target_include_directories(scoring_ablation_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(scoring_ablation_support PUBLIC docking_gpu_output)

add_executable(scoring_ablation scoring_ablation.cpp)
target_link_libraries(scoring_ablation PRIVATE scoring_ablation_support)

add_executable(scoring_ablation_test ablation_test.cpp)
target_link_libraries(scoring_ablation_test PRIVATE scoring_ablation_support)
enable_testing()
add_test(NAME scoring_ablation_synthetic COMMAND scoring_ablation_test)
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME scoring_ablation_workflow
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/workflow_test.py)
endif()
