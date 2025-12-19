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

    void apply_boundary_conditions() const;

    void update_velocities() const;

    void compute_tentative_velocities() const;

    void compute_poisson_source() const;

    void populate_host_region(const HostRegion& host_region) const;

private:
    std::shared_ptr<Metadata> metadata;

    static constexpr dim3 block_size = { 8, 8, 1 }; // TODO: dynamic block size with cudaOccupancyMaxPotentialBlockSize
    const dim3 grid_size;

    static constexpr compute_t initial_velocity_x = 1.0;
    static constexpr compute_t initial_velocity_y = 0.0;
    static constexpr compute_t initial_pressure = 0.0;

    compute_t * velocity_x = nullptr;
    compute_t * velocity_y = nullptr;
    compute_t * tentative_velocity_x = nullptr;
    compute_t * tentative_velocity_y = nullptr;
    compute_t * pressure = nullptr;
    compute_t * poisson_source = nullptr;
    cell_flags * flags = nullptr;
    bounds * v_body_bounds = nullptr;
};

} // namespace owd

#endif // HIPC_ASSESSMENT_CUDA_DEVICEREGION_CUH
