//
// Created by owd on 18/12/2025.
//

#ifndef HIPC_ASSESSMENT_CUDA_SAFECUDA_CUH
#define HIPC_ASSESSMENT_CUDA_SAFECUDA_CUH

#include <stdexcept>

#ifndef TOSTRING
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#endif

#define SafeCUDA(expr) \
    do { \
        cudaError_t status = expr; \
        if (status != cudaSuccess) \
            throw std::runtime_error(std::string(__FILE__) + ':' + TOSTRING(__LINE__) + ": expression " + #expr + \
                " failed with error " + cudaGetErrorString(status)); \
    } while (0)

#endif // HIPC_ASSESSMENT_CUDA_SAFECUDA_CUH
