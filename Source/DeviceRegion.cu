//
// Created by owd on 18/12/2025.
//

#include "DeviceRegion.cuh"
#include "HostRegion.cuh"
#include "SafeCUDA.cuh"

namespace owd
{

namespace kernels
{

__global__ void set_body_bounds(const Metadata * const metadata, bounds * const v_body_bounds)
{
    const std::size_t x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_idx >= metadata->extents.x)
        return;

    const unsigned int resolution = metadata->resolution;
    const compute_t x = static_cast<compute_t>(x_idx) / resolution - 0.5;

    bounds bounds = {
        .begin = 0,
        .end = 0
    };

    if (x >= 0.0 || x <= 1.0) {
        const compute_t maximum_camber = metadata->naca_specifier.maximum_camber / 100.0;
        const compute_t edge_distance = metadata->naca_specifier.edge_distance / 10.0;
        const compute_t thickness = metadata->naca_specifier.maximum_thickness / 100.0;

        const compute_t x_sq = x * x;

        const compute_t mean_camber_line_y = x <= edge_distance ?
            maximum_camber / (edge_distance * edge_distance) * (2.0 * edge_distance * x - x_sq) :
            maximum_camber / ((1.0 - edge_distance) * (1.0 - edge_distance)) *
                (1.0 - 2.0 * edge_distance + 2.0 * edge_distance * x - x_sq);

        const compute_t norm = x <= edge_distance
                ? 2.0 * maximum_camber / (edge_distance * edge_distance) * (edge_distance - x)
                : 2.0 * maximum_camber / ((1.0 - edge_distance) * (1.0 - edge_distance)) * (edge_distance - x);

        const compute_t midline_distance = 5.0 * thickness * cos(atan(norm)) *
            (0.2969 * sqrt(x) - 0.1260 * x - 0.3516 * x_sq + 0.2843 * x * x_sq - 0.1015 * x_sq * x_sq);

        bounds.begin = floor((mean_camber_line_y - midline_distance + metadata->problem_size.y / 2.0) * resolution);
        bounds.end = ceil((mean_camber_line_y + midline_distance + metadata->problem_size.y / 2.0) * resolution);
    }

    v_body_bounds[x_idx] = bounds;
}

__global__ void set_boundaries(const Metadata * const metadata, compute_t * const velocity_x,
    compute_t * const velocity_y, compute_t * const pressure, cell_flags * const flags,
    const bounds * const v_body_bounds)
{
    const dim2 idx(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    );

    if (idx.x < metadata->extents.x && idx.y < metadata->extents.y) {
        const indexer_t array_idx = idx.x + metadata->extents.x * idx.y;

        velocity_x[array_idx] = Metadata::initial_velocity_x;
        velocity_y[array_idx] = Metadata::initial_velocity_y;
        pressure[array_idx] = Metadata::initial_pressure;

        const bounds& body_bounds = v_body_bounds[idx.x];

        flags[array_idx] =
            idx.x == 0 || idx.x == metadata->extents.x - 1 ||
            idx.y == 0 || idx.y == metadata->extents.y - 1 ||
                (idx.y >= body_bounds.begin && idx.y < body_bounds.end) ? CELL_BOUNDARY : CELL_FLUID;
    }
}

__global__ void set_neighbouring_flags(const Metadata * const metadata, cell_flags * const flags)
{
    const dim2 idx(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    );

    if (idx.x >= metadata->extents.x || idx.y >= metadata->extents.y)
        return;

    const indexer_t row_increment = metadata->extents.x;
    const indexer_t v_basis = row_increment * idx.y;
    const indexer_t flat_idx = idx.x + v_basis;

    if (!(flags[flat_idx] & CELL_FLUID)) {
        if (idx.x > 0 && flags[flat_idx - 1] & CELL_FLUID)
            flags[flat_idx] = static_cast<cell_flags>(flags[flat_idx] | CELL_FLUID_WEST);
        if (idx.x < metadata->extents.x - 1 && flags[flat_idx + 1] & CELL_FLUID)
            flags[flat_idx] = static_cast<cell_flags>(flags[flat_idx] | CELL_FLUID_EAST);
        if (idx.y > 0 && flags[flat_idx - row_increment] & CELL_FLUID)
            flags[flat_idx] = static_cast<cell_flags>(flags[flat_idx] | CELL_FLUID_SOUTH);
        if (idx.y < metadata->extents.y - 1 && flags[flat_idx + row_increment] & CELL_FLUID)
            flags[flat_idx] = static_cast<cell_flags>(flags[flat_idx] | CELL_FLUID_NORTH);
    }
}

}

DeviceRegion::DeviceRegion(std::shared_ptr<Metadata> metadata) :
    metadata(std::move(metadata)),
    grid_size(
        std::ceil(static_cast<float>(this->metadata->extents.x) / block_size.x),
        std::ceil(static_cast<float>(this->metadata->extents.y) / block_size.y),
        1
    )
{
    const std::size_t allocation_extent = this->metadata->allocation_count;
    const std::size_t allocation_extent_bytes = this->metadata->allocation_byte_count;

    SafeCUDA(cudaMalloc(&velocity_x, allocation_extent_bytes));
    SafeCUDA(cudaMalloc(&velocity_y, allocation_extent_bytes));
    SafeCUDA(cudaMalloc(&tentative_velocity_x, allocation_extent_bytes));
    SafeCUDA(cudaMalloc(&tentative_velocity_y, allocation_extent_bytes));
    SafeCUDA(cudaMalloc(&pressure, allocation_extent_bytes));
    SafeCUDA(cudaMalloc(&poisson_source, allocation_extent_bytes));
    SafeCUDA(cudaMalloc(&flags, sizeof(cell_flags) * allocation_extent));
    SafeCUDA(cudaMalloc(&v_body_bounds, sizeof(bounds) * this->metadata->extents.x));

    const Metadata * const metadata_ro_ptr = this->metadata.get();

    kernels::set_body_bounds<<<1, this->metadata->extents.x>>>(metadata_ro_ptr, v_body_bounds);
    kernels::set_boundaries<<<grid_size, block_size>>>(metadata_ro_ptr, velocity_x, velocity_y, pressure, flags,
        v_body_bounds);
    kernels::set_neighbouring_flags<<<grid_size, block_size>>>(metadata_ro_ptr, flags);
}

DeviceRegion::~DeviceRegion() noexcept
{
    /*
     * Do not use safe_cuda here, as the destructor shouldn't be throwing exceptions. The CUDA runtime will report any
     * important failures.
     */

    cudaFree(v_body_bounds);
    cudaFree(flags);
    cudaFree(poisson_source);
    cudaFree(pressure);
    cudaFree(tentative_velocity_y);
    cudaFree(tentative_velocity_x);
    cudaFree(velocity_y);
    cudaFree(velocity_x);
}

void DeviceRegion::populate_host_region(const HostRegion &host_region) const
{
    host_region.receive_velocity_x(velocity_x);
    host_region.receive_velocity_y(velocity_y);
    host_region.receive_pressure(pressure);
}

} // namespace owd
