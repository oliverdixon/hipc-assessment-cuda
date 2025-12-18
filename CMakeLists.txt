cmake_minimum_required(VERSION 4.0)
project(HIPC_Assessment_CUDA LANGUAGES CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(Airfoil_CUDA
    Source/airfoil.cu
    Source/instance.cu
    Source/instance.h
    Source/types.h
)

set_property(TARGET Airfoil_CUDA PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_executable(OccupanyTest Source/occupany_test.cu)
