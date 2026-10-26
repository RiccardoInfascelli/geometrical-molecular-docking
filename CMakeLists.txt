cmake_minimum_required(VERSION 3.20)
project(geometric_docking LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_CUDA "Build the CUDA docking backend" OFF)
option(ENABLE_OPENMP "Enable optional host triplet-query parallelism" OFF)
option(BUILD_EXPERIMENTS "Build the diagnostic and experimental drivers" ON)
include(CTest)
set(DOCKING_TEST_COMPLEX "" CACHE PATH
    "Optional authorized complex for the real CPU/CUDA integration test")

if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()
find_package(Eigen3 REQUIRED)
add_subdirectory(../chem-interactions ${CMAKE_BINARY_DIR}/chem-interactions)

# Shared host code. These CUDA data adapters do not require a CUDA compiler.
add_library(docking_core
    src/molecule_io.cpp
    src/interaction_areas.cpp
    src/geometry.cpp
    src/rigid_alignment.cpp
    src/triplet_generation.cpp
    src/matching_areas.cpp
    src/matching_grid.cpp
    src/protein_grid.cpp
    src/protein_preparation.cpp
    src/protein_area_crop.cpp
    src/ligand_preparation.cpp
    src/pose_generation.cpp
    src/stage_profile.cpp
    src/pose_targets.cpp
    src/interaction_scoring.cpp
    src/clash.cpp
    src/diagnostics.cpp
    src/pose_ranking.cpp
    src/run_config.cpp
    src/pipeline.cpp
    src/report.cpp
    src/pose_output.cpp
    src/benchmark_csv.cpp
    src/cuda/cuda_triplet_index.cpp
    src/cuda/cuda_geometry.cpp
    src/cuda/cuda_grid.cpp
    src/cuda/cuda_fast_scoring.cpp)
target_include_directories(docking_core PUBLIC include)
target_link_libraries(docking_core PUBLIC libinteraction Eigen3::Eigen)
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(docking_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(docking_core PUBLIC DOCKING_ENABLE_OPENMP)
endif()

# Compile each scorer once; applications and isolated experiments share it.
add_library(docking_rescoring
    src/pose_rescoring.cpp src/vinardo_scoring.cpp src/pose_score_ranking.cpp)
target_include_directories(docking_rescoring PUBLIC include)
target_link_libraries(docking_rescoring PUBLIC docking_core)

add_library(docking_gpu_output src/gpu_docking_output.cpp)
target_include_directories(docking_gpu_output PUBLIC include)
target_link_libraries(docking_gpu_output PUBLIC docking_rescoring)

add_library(docking_gpu_config src/gpu_docking_config.cpp)
target_include_directories(docking_gpu_config PUBLIC include)
target_link_libraries(docking_gpu_config PUBLIC docking_gpu_output)

if(ENABLE_CUDA)
    add_library(docking_cuda
        src/cuda/cuda_backend.cu
        src/cuda/cuda_triplet_index.cu
        src/cuda/cuda_triplet_lookup.cu
        src/cuda/cuda_geometry_upload.cu
        src/cuda/cuda_kabsch.cu
        src/cuda/cuda_grid_upload.cu
        src/cuda/cuda_clash.cu
        src/cuda/cuda_compaction.cu
        src/cuda/cuda_fast_score.cu
        src/cuda/cuda_pose_pipeline.cpp
        src/cuda/cuda_screening_batch.cu)
    target_include_directories(docking_cuda PUBLIC include)
    target_link_libraries(docking_cuda PUBLIC
        libinteraction Eigen3::Eigen CUDA::cudart CUDA::cusolver)
    set_target_properties(docking_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

    add_library(docking_gpu_search src/cuda/gpu_pose_search.cpp)
    target_include_directories(docking_gpu_search PUBLIC include)
    target_link_libraries(docking_gpu_search PUBLIC docking_core docking_cuda)

    add_library(docking_cuda_rescoring
        src/cuda/cuda_weighted_interaction_scoring.cpp src/cuda/cuda_weighted_interaction_scoring.cu
        src/cuda/cuda_vinardo_scoring.cpp src/cuda/cuda_vinardo_scoring.cu
        src/cuda/cuda_pose_ranking.cu src/cuda/cuda_pose_rescoring.cpp)
    target_include_directories(docking_cuda_rescoring PUBLIC include)
    target_link_libraries(docking_cuda_rescoring PUBLIC docking_cuda docking_rescoring)
    set_target_properties(docking_cuda_rescoring PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    # Keep the existing CPU-compatible operation boundaries in these scorers only.
    set_source_files_properties(
        src/cuda/cuda_weighted_interaction_scoring.cu src/cuda/cuda_vinardo_scoring.cu
        src/cuda/cuda_pose_ranking.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")

    add_library(docking_gpu_screen src/cuda/gpu_docking.cpp)
    target_include_directories(docking_gpu_screen PUBLIC include)
    target_link_libraries(docking_gpu_screen PUBLIC
        docking_gpu_search docking_gpu_output docking_cuda_rescoring)
endif()

# Operational applications: preserve target names, CLI and scientific defaults.
add_executable(geometric_docking apps/docking_main.cpp)
target_link_libraries(geometric_docking PRIVATE docking_core)
add_executable(multi_ligand_screen apps/multi_ligand_screen.cpp)
target_link_libraries(multi_ligand_screen PRIVATE docking_core)
bsl_set_cxx_flags(multi_ligand_screen)

if(ENABLE_CUDA)
    add_executable(multi_ligand_screen_gpu apps/multi_ligand_screen_gpu.cpp)
    target_link_libraries(multi_ligand_screen_gpu PRIVATE
        docking_core docking_cuda docking_gpu_search docking_rescoring)
    bsl_set_cxx_flags(multi_ligand_screen_gpu)
    add_executable(gpu_docking apps/gpu_docking_main.cpp)
    target_link_libraries(gpu_docking PRIVATE docking_gpu_screen docking_gpu_config)
    bsl_set_cxx_flags(gpu_docking)
endif()

# Historical references and diagnostic protocols remain separate executables.
if(BUILD_EXPERIMENTS)
    set(GEOMETRY_EXPERIMENTS
        triplet_failure_diagnostic clash_threshold_experiment
        pose_target_shift_experiment pocket_area_experiment
        area_accessibility_experiment clearance_pruning_experiment
        pocket_openness_experiment)
    foreach(experiment IN LISTS GEOMETRY_EXPERIMENTS)
        add_executable(${experiment} experiments/geometry/${experiment}_main.cpp)
        target_link_libraries(${experiment} PRIVATE docking_core)
        if(NOT experiment STREQUAL "triplet_failure_diagnostic")
            bsl_set_cxx_flags(${experiment})
        endif()
    endforeach()
    add_executable(protein_index_cutoff_bench experiments/geometry/protein_index_cutoff_bench.cpp)
    target_link_libraries(protein_index_cutoff_bench PRIVATE docking_core)
    bsl_set_cxx_flags(protein_index_cutoff_bench)
    add_executable(offline_profile_experiment experiments/profiling/offline_profile_experiment_main.cpp)
    target_link_libraries(offline_profile_experiment PRIVATE docking_core)
    bsl_set_cxx_flags(offline_profile_experiment)
    add_executable(query_preparation_bench experiments/openmp_queries/query_preparation_bench.cpp)
    target_link_libraries(query_preparation_bench PRIVATE docking_core)
    add_executable(ligand_diameter_stats experiments/analysis/ligand_diameter_stats.cpp)
    target_link_libraries(ligand_diameter_stats PRIVATE docking_core)
    bsl_set_cxx_flags(ligand_diameter_stats)
    add_executable(crystal_affinity_interactions experiments/analysis/crystal_affinity_interactions.cpp)
    target_link_libraries(crystal_affinity_interactions PRIVATE docking_core)
    add_executable(serial_docking experiments/reference/serial_main.cpp src/serial_docking.cpp)
    target_link_libraries(serial_docking PRIVATE docking_core)
    bsl_set_cxx_flags(serial_docking)
    add_executable(vinardo_rescore_experiment experiments/scoring/vinardo_rescore_experiment.cpp)
    target_link_libraries(vinardo_rescore_experiment PRIVATE docking_rescoring)
    if(ENABLE_CUDA)
        add_executable(multi_ligand_screen_gpu_batch
            experiments/screening_batch/multi_ligand_screen_gpu_batch.cpp)
        target_link_libraries(multi_ligand_screen_gpu_batch PRIVATE docking_core docking_cuda)
        bsl_set_cxx_flags(multi_ligand_screen_gpu_batch)
    endif()
endif()

if(BUILD_TESTING)
    set(CPU_CORE_TESTS
        gpu_fast_scoring_test gpu_grid_test gpu_triplet_index_test gpu_geometry_test
        rigid_alignment_test pose_generation_test pose_ranking_test molecule_io_test)
    foreach(test IN LISTS CPU_CORE_TESTS)
        add_executable(${test} tests/cpu/${test}.cpp)
        target_link_libraries(${test} PRIVATE docking_core)
    endforeach()
    add_executable(pose_score_ranking_test tests/cpu/pose_score_ranking_test.cpp)
    target_link_libraries(pose_score_ranking_test PRIVATE docking_rescoring)
    add_executable(pose_rescoring_test tests/cpu/pose_rescoring_test.cpp)
    target_link_libraries(pose_rescoring_test PRIVATE docking_rescoring)
    add_executable(gpu_docking_config_test tests/cpu/gpu_docking_config_test.cpp)
    target_link_libraries(gpu_docking_config_test PRIVATE docking_gpu_config)
    add_executable(gpu_docking_output_test tests/cpu/gpu_docking_output_test.cpp)
    target_link_libraries(gpu_docking_output_test PRIVATE docking_gpu_output)
    set(CPU_TESTS ${CPU_CORE_TESTS} pose_score_ranking_test pose_rescoring_test
        gpu_docking_config_test gpu_docking_output_test)

    if(ENABLE_CUDA)
        set(CUDA_CORE_TESTS
            cuda_smoke_test cuda_triplet_lookup_test cuda_geometry_upload_test
            cuda_kabsch_test cuda_clash_test cuda_fast_score_test cuda_pose_pipeline_test)
        foreach(test IN LISTS CUDA_CORE_TESTS)
            add_executable(${test} tests/cuda/${test}.cpp)
            target_link_libraries(${test} PRIVATE docking_cuda)
        endforeach()
        add_executable(cuda_pose_rescoring_test tests/cuda/cuda_pose_rescoring_test.cpp)
        target_link_libraries(cuda_pose_rescoring_test PRIVATE docking_cuda_rescoring docking_gpu_output)
        add_executable(cuda_screening_batch_test tests/cuda/cuda_screening_batch_test.cpp)
        target_link_libraries(cuda_screening_batch_test PRIVATE docking_core docking_cuda)
        add_executable(cuda_real_pipeline_test tests/integration/cuda_real_pipeline_test.cpp)
        target_link_libraries(cuda_real_pipeline_test PRIVATE docking_core docking_cuda)
        add_executable(cpu_cuda_docking_test tests/integration/cpu_cuda_docking_test.cpp)
        target_link_libraries(cpu_cuda_docking_test PRIVATE docking_gpu_screen)
        set(CUDA_TESTS ${CUDA_CORE_TESTS} cuda_pose_rescoring_test cuda_screening_batch_test)
    endif()

    # Isolated experimental builds register their own tests, not unbuilt engine targets.
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        foreach(test IN LISTS CPU_TESTS)
            add_test(NAME ${test} COMMAND ${test})
            set_tests_properties(${test} PROPERTIES LABELS "cpu;synthetic")
        endforeach()
        add_test(NAME rigid_alignment_quaternion_test COMMAND rigid_alignment_test)
        set_tests_properties(rigid_alignment_quaternion_test PROPERTIES
            ENVIRONMENT "DOCKING_KABSCH_METHOD=quaternion" LABELS "cpu;synthetic")
        # Tests start with fixed selectors; an interactive shell must not choose their protocol.
        set_tests_properties(${CPU_TESTS} PROPERTIES ENVIRONMENT
            "DOCKING_KABSCH_METHOD=svd;DOCKING_CPU_CLASH_METHOD=original;DOCKING_CPU_SCREENING_ONLY=0;DOCKING_QUERY_METHOD=reference;DOCKING_PROFILE_STAGES=0")
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            foreach(script IN ITEMS test_clearance_pruning test_gpu_rescoring_comparison)
                add_test(NAME ${script} COMMAND ${Python3_EXECUTABLE} -B
                    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/${script}.py)
                set_tests_properties(${script} PROPERTIES LABELS "cpu;synthetic;python")
            endforeach()
        endif()
        if(ENABLE_CUDA)
            foreach(test IN LISTS CUDA_TESTS)
                add_test(NAME ${test} COMMAND ${test})
                set_tests_properties(${test} PROPERTIES LABELS "cuda;synthetic"
                    ENVIRONMENT "DOCKING_KABSCH_METHOD=svd;DOCKING_QUERY_METHOD=reference;DOCKING_PROFILE_STAGES=0")
            endforeach()
            add_test(NAME cuda_kabsch_quaternion_test COMMAND cuda_kabsch_test)
            set_tests_properties(cuda_kabsch_quaternion_test PROPERTIES
                ENVIRONMENT "DOCKING_KABSCH_METHOD=quaternion" LABELS "cuda;synthetic")
            if(NOT DOCKING_TEST_COMPLEX STREQUAL "")
                add_test(NAME cuda_real_pipeline_test COMMAND cuda_real_pipeline_test ${DOCKING_TEST_COMPLEX})
                set_tests_properties(cuda_real_pipeline_test PROPERTIES LABELS "cuda;integration")
                add_test(NAME cpu_cuda_docking_test COMMAND cpu_cuda_docking_test "${DOCKING_TEST_COMPLEX}")
                set_tests_properties(cpu_cuda_docking_test PROPERTIES
                    LABELS "cuda;integration"
                    ENVIRONMENT "DOCKING_KABSCH_METHOD=svd;DOCKING_CPU_CLASH_METHOD=original;DOCKING_CPU_SCREENING_ONLY=0;DOCKING_QUERY_METHOD=reference;DOCKING_PROFILE_STAGES=0")
                add_test(NAME cpu_cuda_docking_quaternion_test COMMAND cpu_cuda_docking_test "${DOCKING_TEST_COMPLEX}")
                set_tests_properties(cpu_cuda_docking_quaternion_test PROPERTIES
                    LABELS "cuda;integration"
                    ENVIRONMENT "DOCKING_KABSCH_METHOD=quaternion;DOCKING_CPU_CLASH_METHOD=original;DOCKING_CPU_SCREENING_ONLY=0;DOCKING_QUERY_METHOD=reference;DOCKING_PROFILE_STAGES=0")
                if(Python3_Interpreter_FOUND)
                    add_test(NAME gpu_docking_library_test COMMAND ${Python3_EXECUTABLE} -B
                        ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/gpu_docking_library_test.py
                        $<TARGET_FILE:gpu_docking> "${DOCKING_TEST_COMPLEX}")
                    set_tests_properties(gpu_docking_library_test PROPERTIES LABELS "cuda;integration;python")
                endif()
            endif()
        endif()
    endif()
endif()
