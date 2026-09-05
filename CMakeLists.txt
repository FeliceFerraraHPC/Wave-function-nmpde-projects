cmake_minimum_required(VERSION 3.12.0)
project(WaveBenchmark LANGUAGES CXX C)

include(common/cmake-common.cmake)

add_executable(WaveBenchmark
    src/main.cpp
    src/WaveSolverTheta.cpp
    src/WaveSolverMatFree.cpp
    src/WaveSolverDG.cpp
)
deal_ii_setup_target(WaveBenchmark)

add_custom_target(clean-output
    COMMAND ${CMAKE_COMMAND} -E rm -f
        ${CMAKE_BINARY_DIR}/solution_*.vtu
        ${CMAKE_BINARY_DIR}/solution_*.pvtu
        ${CMAKE_BINARY_DIR}/convergence.csv
    COMMENT "Cleaning VTU/PVTU/CSV output files"
)
