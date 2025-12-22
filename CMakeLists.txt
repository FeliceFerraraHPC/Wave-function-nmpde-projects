cmake_minimum_required(VERSION 3.12.0)
project(project_title LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(WaveEquation src/WaveEquation.cpp)
deal_ii_setup_target(WaveEquation)
