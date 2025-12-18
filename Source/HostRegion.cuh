//
// Created by owd on 18/12/2025.
//

#ifndef HIPC_ASSESSMENT_CUDA_HOSTREGION_CUH
#define HIPC_ASSESSMENT_CUDA_HOSTREGION_CUH

#include <memory>

#include "Metadata.cuh"
#include "types.h"

namespace owd
{

class HostRegion
{
public:
    explicit HostRegion(std::shared_ptr<Metadata> metadata);

    ~HostRegion() noexcept;

    void receive_velocity_x(const compute_t * velocity_x_source) const;

    void receive_velocity_y(const compute_t * velocity_y_source) const;

    void receive_pressure(const compute_t * pressure_source) const;

    void vtk_serialise(std::ostream& ostream) const;

private:
    std::shared_ptr<Metadata> metadata;

    compute_t * velocity_x;
    compute_t * velocity_y;
    compute_t * pressure;
};

} // namespace owd

#endif // HIPC_ASSESSMENT_CUDA_HOSTREGION_CUH
