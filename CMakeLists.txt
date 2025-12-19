cmake_minimum_required(VERSION 4.0)
project(HIPC_Assessment_CUDA LANGUAGES CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(Airfoil_CUDA
        Source/AirfoilDriver.cu
        Source/DeviceRegion.cu
        Source/DeviceRegion.cuh
        Source/HostRegion.cu
        Source/HostRegion.cuh
        Source/Metadata.cuh
        Source/SafeCUDA.cuh
)

set_property(TARGET Airfoil_CUDA PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(Airfoil_CUDA PRIVATE $<$<CONFIG:Debug>:-g -G>)

add_executable(OccupanyTest Source/occupany_test.cu)
add_executable(ReductionTest Source/reduction_test.cu)
