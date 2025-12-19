//
// Created by owd on 19/12/2025.
//

#include <iostream>

#include "SafeCUDA.cuh"

template<class T>
__global__ void reduce_sum_kernel(const T* const input, T* const output, const std::size_t input_size)
{
    extern __shared__ T shared_data[];

    const unsigned int thread_idx = threadIdx.x;
    const unsigned int input_idx = blockIdx.x * blockDim.x * 2 + thread_idx;

    // Fold the first two elements into a sum and copy to shared memory.
    if (input_idx + blockDim.x < input_size)
        shared_data[thread_idx] = input[input_idx] + input[input_idx + blockDim.x];
    else if (input_idx < input_size)
        shared_data[thread_idx] = input[input_idx];
    else
        shared_data[thread_idx] = 0;

    __syncthreads();

    // Do the reduction in shared memory.
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (thread_idx < stride)
            shared_data[thread_idx] += shared_data[thread_idx + stride];
        __syncthreads();
    }

    // Write the result for this block to global memory.
    if (thread_idx == 0)
        output[blockIdx.x] = shared_data[0];
}

template<class T>
static T reduce_sum(T * const input, const std::size_t value_count, T * const output, const std::size_t block_count)
{
    std::size_t remaining_blocks = value_count;

    T * reduction_input = input;
    T * reduction_output = output;

    for (std::size_t round_idx = 0; remaining_blocks > 1; ++round_idx) {
        const std::size_t input_size = remaining_blocks;
        remaining_blocks = (input_size + (2 * block_count - 1)) / (2 * block_count);

        if (input_size != 1) {
            reduce_sum_kernel<<<remaining_blocks, block_count, block_count * sizeof(T)>>>(reduction_input,
                reduction_output, input_size);

            if (round_idx & 1) {
                reduction_input = input;
                reduction_output = output;
            } else {
                reduction_input = output;
                reduction_output = input;
            }
        }
    }

    T sum_value;
    SafeCUDA(cudaMemcpy(&sum_value, reduction_input, sizeof(T), cudaMemcpyDeviceToHost));
    return sum_value;
}

int main()
{
    static constexpr std::size_t size = 5;
    static constexpr std::size_t block_count = 4;

    auto * host_array = new float[size];
    float * device_array;

    for (std::size_t idx = 0; idx < size; ++idx)
        host_array[idx] = static_cast<float>(idx) + 1.0f;

    SafeCUDA(cudaMalloc(&device_array, sizeof(float) * size));
    SafeCUDA(cudaMemcpy(device_array, host_array, size * sizeof(float), cudaMemcpyHostToDevice));

    float * helper_array;
    SafeCUDA(cudaMalloc(&helper_array, (size + (2 * block_count - 1)) / (2 * block_count) * sizeof(float)));

    std::cout << "Answer: " << reduce_sum(device_array, size, helper_array, block_count) << std::endl;

    cudaFree(helper_array);
    cudaFree(device_array);
    delete[] host_array;

    return 0;
}
