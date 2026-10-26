cmake_minimum_required(VERSION 3.20)
project(conformer_ensemble_experiment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Reuse the engine's host libraries in this experiment's own build directory.
set(ENABLE_CUDA OFF CACHE BOOL "This experiment only uses the CPU engine" FORCE)
add_subdirectory(../.. engine EXCLUDE_FROM_ALL)

add_executable(conformer_docking conformer_docking.cpp)
target_link_libraries(conformer_docking PRIVATE docking_gpu_output)

add_library(conformer_hybrid_ranking hybrid_ranking.cpp)
target_include_directories(conformer_hybrid_ranking PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(conformer_hybrid_ranking PUBLIC docking_rescoring)
add_executable(conformer_hybrid conformer_hybrid.cpp)
target_link_libraries(conformer_hybrid PRIVATE conformer_hybrid_ranking docking_gpu_output)
add_executable(conformer_hybrid_ranking_test hybrid_ranking_test.cpp)
target_link_libraries(conformer_hybrid_ranking_test PRIVATE conformer_hybrid_ranking)

enable_testing()
add_test(NAME conformer_hybrid_ranking COMMAND conformer_hybrid_ranking_test)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME conformer_ensemble_synthetic
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/conformer_test.py)
# Missing Python RDKit must not look like a fully validated experiment.
add_test(NAME conformer_ensemble_rdkit_available
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/prepare_conformers.py --check)
add_test(NAME conformer_hybrid_io
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/hybrid_test.py)
