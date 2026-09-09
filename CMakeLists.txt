cmake_minimum_required(VERSION 3.12.0)
project(project_title LANGUAGES CXX C)

include(common/cmake-common.cmake)

# ----------------------------------------------------------------------------
# WaveBenchmark: unified driver for the three wave-equation solvers
# (Theta-scheme, matrix-free CG, matrix-free DG/SIPG). See main.cpp for the
# --mode / --dim / --refine / --time / --solver / --gamma / --output CLI
# options.
#
# WaveSolverBase.hpp and WaveFunctions.hpp are header-only (templates +
# inline functions), so they don't need their own .cpp source.
# ----------------------------------------------------------------------------
add_executable(WaveBenchmark
	src/main.cpp
	src/WaveSolverTheta.cpp
	src/WaveSolverMatFree.cpp
	src/WaveSolverDG.cpp
)
deal_ii_setup_target(WaveBenchmark)

add_custom_target(clean-output
	COMMAND ${CMAKE_COMMAND} -E rm -f
		${CMAKE_BINARY_DIR}/cg_solution*.vtu
		${CMAKE_BINARY_DIR}/cg_solution*.pvtu
		${CMAKE_BINARY_DIR}/dg_solution*.vtu
		${CMAKE_BINARY_DIR}/dg_solution*.pvtu
		${CMAKE_BINARY_DIR}/theta_solution*.vtu
		${CMAKE_BINARY_DIR}/theta_solution*.pvtu
		${CMAKE_BINARY_DIR}/energy_*.csv
		${CMAKE_BINARY_DIR}/convergence_*.csv
	COMMENT "Cleaning VTU/PVTU/CSV output files"
)