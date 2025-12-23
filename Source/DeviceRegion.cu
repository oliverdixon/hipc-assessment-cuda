//
// Created by owd on 18/12/2025.
//

#include "DeviceRegion.cuh"
#include "HostRegion.cuh"
#include "SafeCUDA.cuh"

namespace owd
{

enum SORPhase
{
    SOR_RED = 0,
    SOR_BLACK = 1
};

namespace kernels
{

__global__ void set_body_bounds(const Metadata * const metadata, bounds * const v_body_bounds)
{
    const std::size_t x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_idx >= metadata->extents.x)
        return;

    const unsigned int resolution = metadata->resolution;
    const compute_t x = static_cast<compute_t>(x_idx) / resolution - 0.5;

    bounds bounds{};

    if (x >= 0.0 && x <= 1.0) {
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

    if (idx.x >= metadata->extents.x || idx.y >= metadata->extents.y)
        return;

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
        if (idx.y < metadata->extents.y - 1 && flags[flat_idx + row_increment] & CELL_FLUID)
            flags[flat_idx] = static_cast<cell_flags>(flags[flat_idx] | CELL_FLUID_SOUTH);
        if (idx.y > 0 && flags[flat_idx - row_increment] & CELL_FLUID)
            flags[flat_idx] = static_cast<cell_flags>(flags[flat_idx] | CELL_FLUID_NORTH);
    }
}

__global__ void apply_west_east_boundary_conditions(const Metadata * const metadata, compute_t * const velocity_x,
    compute_t * const velocity_y)
{
    const dim2 idx{
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    if (idx.y >= metadata->extents.y)
        return;

    // Fluid freely flows in from the west
    const indexer_t west_basis = metadata->extents.x * idx.y;
    velocity_x[west_basis] = velocity_x[west_basis + 1];
    velocity_y[west_basis] = velocity_y[west_basis + 1];

    // Fluid freely flows out to the east
    const indexer_t east_basis = west_basis + metadata->extents.x - 1;
    velocity_x[east_basis - 1] = velocity_x[east_basis - 2];
    velocity_y[east_basis] = velocity_y[east_basis - 1];
}

__global__ void apply_north_south_boundary_conditions(const Metadata * const metadata, compute_t * const velocity_x,
    compute_t * const velocity_y)
{
    const dim2 idx{
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    if (idx.x >= metadata->extents.x || idx.y >= metadata->extents.y)
        return;

    /*
     * At the north boundary, the vertical velocity approaches zero. Fluid approaches freely. The basis is our position
     * on the north row.
     */
    const indexer_t north_basis = idx.x;
    velocity_x[north_basis] = velocity_x[north_basis + metadata->extents.x];
    velocity_y[north_basis] = 0.0;

    // Ditto for the south boundary. The basis is our position on the south row.
    const indexer_t south_basis = north_basis + (metadata->extents.y - 1) * metadata->extents.x;
    velocity_x[south_basis] = velocity_x[south_basis - metadata->extents.x];
    velocity_y[south_basis - metadata->extents.x] = 0.0;
}

__global__ void apply_inflow_boundary_conditions(const Metadata * const metadata, compute_t * const velocity_x,
    compute_t * const velocity_y)
{
    const dim2 idx{
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    if (idx.y >= metadata->extents.y)
        return;

    // Fix the western-edge velocities such that there is a continual flow of fluid into the simulation space.
    const indexer_t west_anchored_idx = metadata->extents.x * idx.y;
    velocity_x[west_anchored_idx] = Metadata::initial_velocity_x;
    velocity_y[west_anchored_idx] = 2 * Metadata::initial_velocity_y - velocity_y[west_anchored_idx + 1];
}

__global__ void apply_obstacle_boundary_conditions(const Metadata * const metadata, compute_t * const velocity_x,
    compute_t * const velocity_y, const cell_flags * const flags)
{
    const dim2 idx{
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    if (idx.x < 1 || idx.x >= metadata->extents.x - 1 || idx.y < 1 || idx.y >= metadata->extents.y - 1)
        return; // Kernel invocation is not in the interior.

    const indexer_t idx_central = idx.x + idx.y * metadata->extents.x;

    if (flags[idx_central] & CELL_FLUID)
        return; // Obstacle boundary conditions apply only to non-fluid cells.

    const indexer_t idx_north = idx.x + metadata->extents.x * (idx.y - 1);
    const indexer_t idx_south = idx.x + metadata->extents.x * (idx.y + 1);
    const indexer_t idx_west = idx.x - 1 + metadata->extents.x * idx.y;
    const indexer_t idx_east = idx.x + 1 + metadata->extents.x * idx.y;

    const indexer_t idx_northeast = idx.x + 1 + metadata->extents.x * (idx.y - 1);
    const indexer_t idx_southwest = idx.x - 1 + metadata->extents.x * (idx.y + 1);
    const indexer_t idx_northwest = idx.x - 1 + metadata->extents.x * (idx.y - 1);

    switch (flags[idx_central]) {
    case CELL_FLUID_NORTH:
        velocity_y[idx_central] = 0.0;
        velocity_x[idx_central] = -velocity_x[idx_south];
        velocity_x[idx_west] = -velocity_x[idx_southwest];
        break;
    case CELL_FLUID_EAST:
        velocity_x[idx_central] = 0.0;
        velocity_y[idx_central] = -velocity_y[idx_east];
        velocity_y[idx_north] = -velocity_y[idx_northeast];
        break;
    case CELL_FLUID_SOUTH:
        velocity_y[idx_north] = 0.0;
        velocity_x[idx_central] = -velocity_x[idx_north];
        velocity_x[idx_west] = -velocity_x[idx_northwest];
        break;
    case CELL_FLUID_WEST:
        velocity_x[idx_west] = 0.0;
        velocity_y[idx_central] = -velocity_y[idx_west];
        velocity_y[idx_north] = -velocity_y[idx_northwest];
        break;
    case CELL_FLUID_NORTHEAST:
        velocity_y[idx_central] = 0.0;
        velocity_x[idx_central] = 0.0;
        velocity_y[idx_north] = -velocity_y[idx_northeast];
        velocity_x[idx_west] = -velocity_x[idx_southwest];
        break;
    case CELL_FLUID_SOUTHEAST:
        velocity_y[idx_north] = 0.0;
        velocity_x[idx_central] = 0.0;
        velocity_y[idx_central] = -velocity_y[idx_east];
        velocity_x[idx_west] = -velocity_x[idx_northwest];
        break;
    case CELL_FLUID_SOUTHWEST:
        velocity_y[idx_north] = 0.0;
        velocity_x[idx_west] = 0.0;
        velocity_y[idx_central] = -velocity_y[idx_west];
        velocity_x[idx_central] = -velocity_x[idx_north];
        break;
    case CELL_FLUID_NORTHWEST:
        velocity_y[idx_central] = 0.0;
        velocity_x[idx_west] = 0.0;
        velocity_y[idx_north] = -velocity_y[idx_northwest];
        velocity_x[idx_central] = -velocity_x[idx_south];
        break;
    default:;
    }
}

__global__ void update_velocities(const Metadata * const metadata, compute_t * const velocity_x,
    compute_t * const velocity_y, const compute_t * const tentative_velocity_x,
    const compute_t * const tentative_velocity_y, const compute_t * const pressure, const cell_flags * const flags)
{
    const dim2 idx{
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    if (idx.x >= metadata->extents.x || idx.y >= metadata->extents.y)
        return;

    const compute_t pressure_diff_factor = metadata->timestep_duration * metadata->resolution;
    const indexer_t idx_central = idx.x + metadata->extents.x * idx.y;

    if (flags[idx_central] & CELL_FLUID) {
        if (flags[idx_central + 1] & CELL_FLUID)
            velocity_x[idx_central] = tentative_velocity_x[idx_central] -
                (pressure[idx_central + 1] - pressure[idx_central]) * pressure_diff_factor;

        if (flags[idx_central + metadata->extents.x] & CELL_FLUID)
            velocity_y[idx_central] = tentative_velocity_y[idx_central] -
                (pressure[idx_central + metadata->extents.x] - pressure[idx_central]) * pressure_diff_factor;
    }
}

__global__ void compute_tentative_velocities(const Metadata * const metadata, const compute_t * const velocity_x,
    const compute_t * const velocity_y, compute_t * const tentative_velocity_x, compute_t * const tentative_velocity_y,
    const cell_flags * const flags)
{
    const dim2 idx{
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    if (idx.x < 1 || idx.x > metadata->extents.x - 2 || idx.y < 1 || idx.y > metadata->extents.y - 2)
        return;

    static constexpr compute_t reynolds = 500.0;
    static constexpr double gamma = 0.9; // Upwind differencing factor in PDE discretisation

    const indexer_t idx_central = idx.x + metadata->extents.x * idx.y;

    const indexer_t idx_north = idx.x + metadata->extents.x * (idx.y - 1);
    const indexer_t idx_south = idx.x + metadata->extents.x * (idx.y + 1);
    const indexer_t idx_west = idx.x - 1 + metadata->extents.x * idx.y;
    const indexer_t idx_east = idx.x + 1 + metadata->extents.x * idx.y;

    const indexer_t idx_northeast = idx.x + 1 + metadata->extents.x * (idx.y - 1);
    const indexer_t idx_southwest = idx.x - 1 + metadata->extents.x * (idx.y + 1);

    const compute_t quarter_resolution = metadata->resolution / 4.0;
    const compute_t sq_resolution = metadata->resolution * metadata->resolution;

    // TODO: check this. Why only checking east when else comment indicates "adjacent cells"?
    if (flags[idx_central] & CELL_FLUID && flags[idx_east] & CELL_FLUID) {
        const compute_t self_advection_x =
            (
                (velocity_x[idx_central] + velocity_x[idx_east]) *
                (velocity_x[idx_central] + velocity_x[idx_east]) +
                gamma * fabs(velocity_x[idx_central] + velocity_x[idx_east]) *
                (velocity_x[idx_central] - velocity_x[idx_east]) -
                (velocity_x[idx_west] + velocity_x[idx_central]) *
                (velocity_x[idx_west] + velocity_x[idx_central]) -
                gamma * fabs(velocity_x[idx_west] + velocity_x[idx_central]) *
                (velocity_x[idx_west] - velocity_x[idx_central])
            ) * quarter_resolution;

        const compute_t cross_advection_y =
            (
                (velocity_y[idx_central] + velocity_y[idx_east]) *
                (velocity_x[idx_central] + velocity_x[idx_south]) +
                gamma * fabs(velocity_y[idx_central] + velocity_y[idx_east]) *
                (velocity_x[idx_central] - velocity_x[idx_south]) -
                (velocity_y[idx_north] + velocity_y[idx_northeast]) *
                (velocity_x[idx_north] + velocity_x[idx_central]) -
                gamma * fabs(velocity_y[idx_north] + velocity_y[idx_northeast]) *
                (velocity_x[idx_north] - velocity_x[idx_central])
            ) * quarter_resolution;

        const compute_t diffusion =
            (
                velocity_x[idx_east] -
                2.0 * velocity_x[idx_central] +
                velocity_x[idx_west] +
                velocity_x[idx_south] -
                2.0 * velocity_x[idx_central] +
                velocity_x[idx_north]
            ) * sq_resolution;

        tentative_velocity_x[idx_central] = velocity_x[idx_central] + metadata->timestep_duration *
            (diffusion / reynolds - self_advection_x - cross_advection_y);
    } else
        // If both adjacent cells are not fluids, the velocity is unchanged.
        tentative_velocity_x[idx_central] = velocity_x[idx_central];

    if (flags[idx_central] & CELL_FLUID && flags[idx_south] & CELL_FLUID) {
        const compute_t cross_advection_x =
            (
                (velocity_x[idx_central] + velocity_x[idx_south]) *
                (velocity_y[idx_central] + velocity_y[idx_east]) +
                gamma * fabs(velocity_x[idx_central] + velocity_x[idx_south]) *
                (velocity_y[idx_central] - velocity_y[idx_east]) -
                (velocity_x[idx_west] + velocity_x[idx_southwest]) *
                (velocity_y[idx_west] + velocity_y[idx_central]) -
                gamma * fabs(velocity_x[idx_west] + velocity_x[idx_southwest]) *
                (velocity_y[idx_west] - velocity_y[idx_central])
            ) * quarter_resolution;

        const compute_t self_advection_y =
            (
                (velocity_y[idx_central] + velocity_y[idx_south]) *
                (velocity_y[idx_central] + velocity_y[idx_south]) +
                gamma * fabs(velocity_y[idx_central] + velocity_y[idx_south]) *
                (velocity_y[idx_central] - velocity_y[idx_south]) -
                (velocity_y[idx_north] + velocity_y[idx_central]) *
                (velocity_y[idx_north] + velocity_y[idx_central]) -
                gamma * fabs(velocity_y[idx_north] + velocity_y[idx_central]) *
                (velocity_y[idx_north] - velocity_y[idx_central])
            ) * quarter_resolution;

        const compute_t diffusion =
            (
                velocity_y[idx_east] -
                2.0 * velocity_y[idx_central] +
                velocity_y[idx_west] +
                velocity_y[idx_south] -
                2.0 * velocity_y[idx_central] +
                velocity_y[idx_north]
            ) * sq_resolution;

        tentative_velocity_y[idx_central] = velocity_y[idx_central] + metadata->timestep_duration *
            (diffusion / reynolds - cross_advection_x - self_advection_y);

    } else
        // If both adjacent cells are not fluids, the velocity is unchanged.
        tentative_velocity_y[idx_central] = velocity_y[idx_central];
}

__global__ void compute_poisson_source(const Metadata * const metadata, const compute_t * const velocity_x,
    const compute_t * const velocity_y, compute_t * const tentative_velocity_x, compute_t * const tentative_velocity_y,
    compute_t * const pressure, compute_t * const poisson_source, const cell_flags * const flags)
{
    const dim2 idx{
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    if (idx.x >= metadata->extents.x || idx.y >= metadata->extents.y)
        return;

    if (idx.x == 0) {
        const indexer_t west_anchored_idx = metadata->extents.x * idx.y;
        tentative_velocity_x[west_anchored_idx] = velocity_x[west_anchored_idx];
        pressure[west_anchored_idx] = pressure[west_anchored_idx + 1];
    }

    else if (idx.x == metadata->extents.x - 1) {
        const indexer_t east_anchored_idx = idx.x + idx.y * metadata->extents.x;
        tentative_velocity_x[east_anchored_idx] = velocity_x[east_anchored_idx];
        pressure[east_anchored_idx] = pressure[east_anchored_idx - 1];
    }

    if (idx.y == 0) {
        const indexer_t north_anchored_idx = idx.x;
        tentative_velocity_y[north_anchored_idx] = velocity_y[north_anchored_idx];
        pressure[north_anchored_idx] = pressure[north_anchored_idx + metadata->extents.x];
    }

    else if (idx.y == metadata->extents.y - 1) {
        const indexer_t south_anchored_idx = idx.x + metadata->extents.x * (idx.y - 1);
        tentative_velocity_y[south_anchored_idx] = velocity_y[south_anchored_idx];
        pressure[south_anchored_idx] = pressure[south_anchored_idx - metadata->extents.x];
    }

    const auto idx_central = idx.x + idx.y * metadata->extents.x;

    if (flags[idx_central] & CELL_FLUID) {
        const compute_t x_tent_vel_diff = (tentative_velocity_x[idx_central] - tentative_velocity_x[idx_central - 1]) *
            metadata->resolution;

        const compute_t y_tent_vel_diff = (tentative_velocity_y[idx_central] -
            tentative_velocity_y[idx_central + metadata->extents.x]) * metadata->resolution;

        poisson_source[idx_central] = (x_tent_vel_diff + y_tent_vel_diff) / metadata->timestep_duration;
    }
}

__global__ void compute_local_residuals(const Metadata * const metadata, const compute_t * const pressure,
    const compute_t * const poisson_source, const cell_flags * const flags, compute_t * const residual_dest,
    unsigned int * const fluid_dest)
{
    const dim2 idx{
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    if (idx.x >= metadata->extents.x || idx.y >= metadata->extents.y)
        return;

    const unsigned int step_sq = metadata->resolution * metadata->resolution;
    const indexer_t idx_central = idx.x + metadata->extents.x * idx.y;

    const indexer_t idx_east = idx.x + 1 + metadata->extents.x * idx.y;
    const indexer_t idx_west = idx.x - 1 + metadata->extents.x * idx.y;
    const indexer_t idx_north = idx.x + metadata->extents.x * (idx.y - 1);
    const indexer_t idx_south = idx.x + metadata->extents.x * (idx.y + 1);

    if (flags[idx_central] & CELL_FLUID) {
        const double epsilon_east = flags[idx_east] & CELL_FLUID ? 1.0 : 0.0;
        const double epsilon_west = flags[idx_west] & CELL_FLUID ? 1.0 : 0.0;
        const double epsilon_north = flags[idx_north] & CELL_FLUID ? 1.0 : 0.0;
        const double epsilon_south = flags[idx_south] & CELL_FLUID ? 1.0 : 0.0;

        const double x_residual = (
            epsilon_east * (pressure[idx_east] - pressure[idx_central]) -
            epsilon_west * (pressure[idx_central] - pressure[idx_west])
        ) * step_sq;

        const double y_residual = (
            epsilon_north * (pressure[idx_south] - pressure[idx_central]) -
            epsilon_south * (pressure[idx_central] - pressure[idx_north])
        ) * step_sq;

        residual_dest[idx_central] = x_residual + y_residual - poisson_source[idx_central];
        fluid_dest[idx_central] = true;
    } else {
        residual_dest[idx_central] = 0.0;
        fluid_dest[idx_central] = false;
    }
}

template<class T>
__global__ void reduce_sum_kernel(const T* const input, T* const output, const std::size_t input_size)
{
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory[];
    T* shared_data = reinterpret_cast<T*>(shared_memory);

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

__global__ void perform_sor_cycle(const Metadata * const metadata, compute_t * const pressure,
    const compute_t * const poisson_source, const cell_flags * const flags, const SORPhase phase)
{
    const dim2 idx{
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    if (idx.x >= metadata->extents.x - 1 || idx.y >= metadata->extents.y - 1)
        return;

    if ((idx.x + idx.y & 1) != phase)
        // Kernel invocation not applicable for the requested phase.
        return;
    
    const indexer_t idx_central = idx.x + metadata->extents.x * idx.y;

    if (!(flags[idx_central] & CELL_FLUID))
        return;

    static constexpr compute_t omega = 1.7; // TODO move
    const compute_t step_sq = metadata->resolution * metadata->resolution;

    const compute_t epsilon_east = flags[idx_central + 1] & CELL_FLUID ? 1.0 : 0.0;
    const compute_t epsilon_west = flags[idx_central - 1] & CELL_FLUID ? 1.0 : 0.0;
    const compute_t epsilon_north = flags[idx_central - metadata->extents.x] & CELL_FLUID ? 1.0 : 0.0;
    const compute_t epsilon_south = flags[idx_central + metadata->extents.x] & CELL_FLUID ? 1.0 : 0.0;

    const compute_t weight = omega / ((epsilon_east + epsilon_west + epsilon_north + epsilon_south) * step_sq);
    const compute_t x_spatial = pressure[idx_central + 1] * epsilon_east + pressure[idx_central - 1] * epsilon_west;
    const compute_t y_spatial = pressure[idx_central + metadata->extents.x] * epsilon_south +
        pressure[idx_central - metadata->extents.x] * epsilon_north;

    pressure[idx_central] = (1.0 - omega) * pressure[idx_central] + weight *
        (step_sq * (x_spatial + y_spatial) - poisson_source[idx_central]);
}

} // namespace kernels

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

    // TODO: reduction array output sizes can be reduced by a factor of block count.

    SafeCUDA(cudaMalloc(&local_residuals_input, allocation_extent_bytes));
    SafeCUDA(cudaMalloc(&fluid_cell_markers_input, sizeof(unsigned int) * allocation_extent));
    SafeCUDA(cudaMalloc(&local_residuals_output, sizeof(compute_t) * allocation_extent));
    SafeCUDA(cudaMalloc(&fluid_cell_markers_output, sizeof(unsigned int) * allocation_extent));

    SafeCUDA(cudaMalloc(&v_body_bounds, sizeof(bounds) * this->metadata->extents.x));

    const Metadata * const metadata_ro_ptr = this->metadata.get();

    // Initialise the problem grids to a suitable initial state for the iterative solvers.
    kernels::set_body_bounds<<<1, this->metadata->extents.x>>>(metadata_ro_ptr, v_body_bounds);
    kernels::set_boundaries<<<grid_size, block_size>>>(metadata_ro_ptr, velocity_x, velocity_y, pressure, flags,
        v_body_bounds);
    kernels::set_neighbouring_flags<<<grid_size, block_size>>>(metadata_ro_ptr, flags);
}

DeviceRegion::~DeviceRegion() noexcept
{
    /*
     * Do not use SafeCUDA here, as the destructor shouldn't be throwing exceptions. The CUDA runtime will report any
     * important failures.
     */

    cudaFree(v_body_bounds);

    cudaFree(fluid_cell_markers_output);
    cudaFree(local_residuals_output);
    cudaFree(fluid_cell_markers_input);
    cudaFree(local_residuals_input);

    cudaFree(flags);
    cudaFree(poisson_source);
    cudaFree(pressure);
    cudaFree(tentative_velocity_y);
    cudaFree(tentative_velocity_x);
    cudaFree(velocity_y);
    cudaFree(velocity_x);
}

void DeviceRegion::apply_boundary_conditions() const
{
    const auto metadata_ptr = metadata.get();

    kernels::apply_west_east_boundary_conditions<<<grid_size, block_size>>>(metadata_ptr, velocity_x, velocity_y);
    kernels::apply_north_south_boundary_conditions<<<grid_size, block_size>>>(metadata_ptr, velocity_x, velocity_y);
    kernels::apply_obstacle_boundary_conditions<<<grid_size, block_size>>>(metadata_ptr, velocity_x, velocity_y, flags);
    kernels::apply_inflow_boundary_conditions<<<grid_size, block_size>>>(metadata_ptr, velocity_x, velocity_y);
}

void DeviceRegion::update_velocities() const
{
    kernels::update_velocities<<<grid_size, block_size>>>(metadata.get(), velocity_x, velocity_y, tentative_velocity_x,
        tentative_velocity_y, pressure, flags);
}

void DeviceRegion::compute_tentative_velocities() const
{
    kernels::compute_tentative_velocities<<<grid_size, block_size>>>(metadata.get(), velocity_x, velocity_y,
        tentative_velocity_x, tentative_velocity_y, flags);
}

void DeviceRegion::compute_poisson_source() const
{
    kernels::compute_poisson_source<<<grid_size, block_size>>>(metadata.get(), velocity_x, velocity_y,
        tentative_velocity_x, tentative_velocity_y, pressure, poisson_source, flags);
}

compute_t DeviceRegion::compute_residual_norm_sq() const
{
    kernels::compute_local_residuals<<<grid_size, block_size>>>(metadata.get(), pressure, poisson_source, flags,
        local_residuals_input, fluid_cell_markers_input);

    constexpr std::size_t block_extent = block_size.x * block_size.y * block_size.z;
    const compute_t residual_sum = reduce_sum(local_residuals_input, metadata->allocation_count, local_residuals_output,
        block_extent);
    const unsigned int fluid_cell_count = reduce_sum(fluid_cell_markers_input, metadata->allocation_count,
        fluid_cell_markers_output, block_extent);

    return residual_sum / fluid_cell_count;
}

void DeviceRegion::perform_sor_cycle() const
{
    const auto metadata_ptr = metadata.get();

    kernels::perform_sor_cycle<<<grid_size, block_size>>>(metadata_ptr, pressure, poisson_source, flags, SOR_RED);
    kernels::perform_sor_cycle<<<grid_size, block_size>>>(metadata_ptr, pressure, poisson_source, flags, SOR_BLACK);
}

void DeviceRegion::populate_host_region(const HostRegion &host_region) const
{
    SafeCUDA(cudaDeviceSynchronize());
    host_region.receive_velocity_x(velocity_x);
    host_region.receive_velocity_y(velocity_y);
    host_region.receive_pressure(pressure);
}

template<class T>
T DeviceRegion::reduce_sum(T * const input, const std::size_t value_count, T * const output,
    const std::size_t block_extent) const
{
    std::size_t remaining_blocks = value_count;

    T * reduction_input = input;
    T * reduction_output = output;

    for (std::size_t round_idx = 0; remaining_blocks > 1; ++round_idx) {
        const std::size_t input_size = remaining_blocks;
        remaining_blocks = (input_size + (2 * block_extent - 1)) / (2 * block_extent);

        if (input_size != 1) {
            kernels::reduce_sum_kernel<<<remaining_blocks, block_extent, block_extent * sizeof(T)>>>(reduction_input,
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

} // namespace owd
