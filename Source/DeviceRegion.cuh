//
// Created by owd on 18/12/2025.
//

#ifndef HIPC_ASSESSMENT_CUDA_DEVICEREGION_CUH
#define HIPC_ASSESSMENT_CUDA_DEVICEREGION_CUH

#include <memory>

#include "types.h"

namespace owd
{

class DeviceRegion;

class Metadata;
class HostRegion;

class DeviceRegion
{
public:
    explicit DeviceRegion(std::shared_ptr<Metadata> metadata);

    ~DeviceRegion() noexcept;

    DeviceRegion(DeviceRegion &&) noexcept = delete;
    DeviceRegion(const DeviceRegion &) noexcept = delete;

    void populate_host_region(const HostRegion& host_region) const;

private:
    std::shared_ptr<Metadata> metadata;

    static constexpr dim3 block_size = { 8, 8, 1 }; // TODO: dynamic block size with cudaOccupancyMaxPotentialBlockSize
    const dim3 grid_size;

    static constexpr compute_t initial_velocity_x = 1.0;
    static constexpr compute_t initial_velocity_y = 0.0;
    static constexpr compute_t initial_pressure = 0.0;

    compute_t * velocity_x;
    compute_t * velocity_y;
    compute_t * tentative_velocity_x;
    compute_t * tentative_velocity_y;
    compute_t * pressure;
    compute_t * poisson_source;
    cell_flags * flags;
    bounds * v_body_bounds;
};

} // namespace owd

#endif // HIPC_ASSESSMENT_CUDA_DEVICEREGION_CUH
