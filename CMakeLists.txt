cmake_minimum_required(VERSION 3.12.0)
project(project_title LANGUAGES CXX C)

include(common/cmake-common.cmake)

add_executable(WaveEquation
	src/main.cpp
	src/WaveEquation.cpp
)
deal_ii_setup_target(WaveEquation)

add_custom_target(clean-output
	COMMAND ${CMAKE_COMMAND} -E rm -f ${CMAKE_BINARY_DIR}/solution_*.vtu ${CMAKE_BINARY_DIR}/solution_*.pvtu
	COMMENT "Cleaning VTU/PVTU output files"
)

