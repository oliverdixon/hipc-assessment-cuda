//
// Created by od641 on 09/01/2026.
//

#ifndef HIPC_ASSESSMENT_CUDA_SERIALPROTOTYPE_HPP
#define HIPC_ASSESSMENT_CUDA_SERIALPROTOTYPE_HPP

#define IDX(h_idx, v_idx) ((h_idx) + (v_idx) * metadata->extents.x)

#define IDX_NORTH(h_idx, v_idx) (h_idx + (v_idx - 1) * metadata->extents.x)
#define IDX_SOUTH(h_idx, v_idx) (h_idx + (v_idx + 1) * metadata->extents.x)
#define IDX_WEST(h_idx, v_idx) ((h_idx + 1) + v_idx * metadata->extents.x)
#define IDX_EAST(h_idx, v_idx) ((h_idx - 1) + v_idx * metadata->extents.x)

namespace owd
{

class SerialPrototype
{
public:
    explicit SerialPrototype(std::shared_ptr<Metadata> metadata_source) :
        metadata(std::move(metadata_source)),
        fluid_cell_count((metadata->extents.x - 2) * (metadata->extents.y - 2))
    {
        const std::size_t allocation_extent = metadata->allocation_count;

        velocity_x = new compute_t[allocation_extent];
        velocity_y = new compute_t[allocation_extent];
        tentative_velocity_x = new compute_t[allocation_extent];
        tentative_velocity_y = new compute_t[allocation_extent];
        pressure = new compute_t[allocation_extent];
        poisson_source = new compute_t[allocation_extent];
        flags = new cell_flags[allocation_extent];

        for (indexer_t h_idx = 0; h_idx < metadata->extents.x; ++h_idx)
            for (indexer_t v_idx = 0; v_idx < metadata->extents.y; ++v_idx) {
                velocity_x[IDX(h_idx, v_idx)] = initial_velocity_x;
                velocity_y[IDX(h_idx, v_idx)] = initial_velocity_y;
                pressure[IDX(h_idx, v_idx)] = initial_pressure;
                flags[IDX(h_idx, v_idx)] = initial_flag;
            }

        const compute_t maximum_camber = static_cast<compute_t>(metadata->naca_specifier.maximum_camber) / 100.0f;
        const compute_t edge_distance = static_cast<compute_t>(metadata->naca_specifier.edge_distance) / 10.0f;
        const compute_t thickness = static_cast<compute_t>(metadata->naca_specifier.maximum_thickness) / 100.0f;

        const dim2 interior_extents = {
            metadata->extents.x - 1,
            metadata->extents.y - 1,
        };

        for (indexer_t h_idx = 0; h_idx < metadata->extents.x; ++h_idx) {
            // Compute the vertical index boundaries of the airfoil body at the fixed horizontal index.
            const bounds v_idx_boundaries = get_airfoil_v_bounds(maximum_camber, edge_distance, thickness, h_idx);

            // Populate the airfoil body with boundary markers.
            for (indexer_t v_idx = v_idx_boundaries.begin; v_idx < v_idx_boundaries.end; ++v_idx)
                flags[IDX(h_idx, v_idx)] = CELL_BOUNDARY;
        }
        
        for (indexer_t h_idx = 0; h_idx < interior_extents.x; ++h_idx) {
            flags[IDX(h_idx, 0)] = CELL_BOUNDARY;
            flags[IDX(h_idx, interior_extents.y - 1)] = CELL_BOUNDARY;
        }
        
        for (indexer_t v_idx = 0; v_idx < interior_extents.y; ++v_idx) {
            flags[IDX(0, v_idx)] = CELL_BOUNDARY;
            flags[IDX(interior_extents.x - 1, v_idx)] = CELL_BOUNDARY;
        }
        
        for (indexer_t h_idx = 1; h_idx < interior_extents.x; ++h_idx)
            for (indexer_t v_idx = 1; v_idx < interior_extents.y; ++v_idx)
                if (!(flags[IDX(h_idx, v_idx)] & CELL_FLUID)) {
                    --fluid_cell_count;

                    if (flags[IDX_WEST(h_idx, v_idx)] & CELL_FLUID)
                        flags[IDX(h_idx, v_idx)] = static_cast<cell_flags>(flags[IDX(h_idx, v_idx)] | CELL_FLUID_EAST);

                    if (flags[IDX_EAST(h_idx, v_idx)] & CELL_FLUID)
                        flags[IDX(h_idx, v_idx)] = static_cast<cell_flags>(flags[IDX(h_idx, v_idx)] | CELL_FLUID_WEST);

                    if (flags[IDX_SOUTH(h_idx, v_idx)] & CELL_FLUID)
                        // mismatch intentional
                        flags[IDX(h_idx, v_idx)] = static_cast<cell_flags>(flags[IDX(h_idx, v_idx)] | CELL_FLUID_NORTH);

                    if (flags[IDX_NORTH(h_idx, v_idx)] & CELL_FLUID)
                        flags[IDX(h_idx, v_idx)] = static_cast<cell_flags>(flags[IDX(h_idx, v_idx)] | CELL_FLUID_SOUTH);
                }
    }

    ~SerialPrototype() noexcept
    {
        delete[] velocity_x;
        delete[] velocity_y;
        delete[] tentative_velocity_x;
        delete[] tentative_velocity_y;
        delete[] pressure;
        delete[] poisson_source;
        delete[] flags;
    }

    explicit SerialPrototype(SerialPrototype &&) noexcept = delete;
    explicit SerialPrototype(const SerialPrototype &) noexcept = delete;

    void populate_host_region(const HostRegion& host_region) const
    {
        host_region.copy_vx(velocity_x);
        host_region.copy_vy(velocity_y);
        host_region.copy_p(pressure);
        host_region.copy_flags(flags);
    }

    void apply_boundary_conditions() const
    {
        const dim2 interior_extents = {
            metadata->extents.x - 1,
            metadata->extents.y - 1,
        };

        for (indexer_t v_idx = 0; v_idx < metadata->extents.y; ++v_idx) {
            // Fluid freely flows in from the west
            velocity_x[IDX(0, v_idx)] = velocity_x[IDX(1, v_idx)];
            velocity_y[IDX(0, v_idx)] = velocity_y[IDX(1, v_idx)];

            // Fluid freely flows out to the east
            velocity_x[IDX(metadata->extents.x - 2, v_idx)] = velocity_x[IDX(metadata->extents.x - 3, v_idx)];
            velocity_y[IDX(metadata->extents.x - 1, v_idx)] = velocity_y[IDX(metadata->extents.x - 2, v_idx)];
        }

        for (indexer_t h_idx = 0; h_idx < interior_extents.x; ++h_idx) {
            /*
             * The vertical velocity approaches zero at the north and south boundaries, but fluid flows freely in the
             * horizontal direction. */
            velocity_y[IDX(h_idx, metadata->extents.y - 2)] = 0.0;
            velocity_x[IDX(h_idx, metadata->extents.y - 1)] = velocity_x[IDX(h_idx, metadata->extents.y - 2)];

            velocity_y[IDX(h_idx, 0)] = 0.0;
            velocity_x[IDX(h_idx, 0)] = velocity_x[IDX(h_idx, 1)];
        }

        /*
         * Apply no-slip boundary conditions to cells that are adjacent to internal obstacle cells. This forces the
         * velocities to tend towards zero in these cells. This portion is not parallelised as the number of boundary
         * cells is small, and establishment of boundary conditions requires writes into neighbouring cells.
         */
        for (indexer_t h_idx = 1; h_idx < interior_extents.x; ++h_idx)
            for (indexer_t v_idx = 1; v_idx < interior_extents.y; ++v_idx)
                if (!(flags[IDX(h_idx, v_idx)] & CELL_FLUID))
                    switch (flags[IDX(h_idx, v_idx)]) {
                    case CELL_FLUID_NORTH:
                        velocity_y[IDX(h_idx, v_idx)] = 0.0;
                        velocity_x[IDX(h_idx, v_idx)] = -velocity_x[IDX(h_idx, v_idx) + 1];
                        velocity_x[IDX(h_idx - 1, v_idx)] = -velocity_x[IDX(h_idx - 1, v_idx + 1)];
                        break;
                    case CELL_FLUID_EAST:
                        velocity_x[IDX(h_idx, v_idx)] = 0.0;
                        velocity_y[IDX(h_idx, v_idx)] = -velocity_y[IDX(h_idx + 1, v_idx)];
                        velocity_y[IDX(h_idx, v_idx) - 1] = -velocity_y[IDX(h_idx + 1, v_idx - 1)];
                        break;
                    case CELL_FLUID_SOUTH:
                        velocity_y[IDX(h_idx, v_idx) - 1] = 0.0;
                        velocity_x[IDX(h_idx, v_idx)] = -velocity_x[IDX(h_idx, v_idx) - 1];
                        velocity_x[IDX(h_idx - 1, v_idx)] = -velocity_x[IDX(h_idx - 1, v_idx - 1)];
                        break;
                    case CELL_FLUID_WEST:
                        velocity_x[IDX(h_idx - 1, v_idx)] = 0.0;
                        velocity_y[IDX(h_idx, v_idx)] = -velocity_y[IDX(h_idx - 1, v_idx)];
                        velocity_y[IDX(h_idx, v_idx) - 1] = -velocity_y[IDX(h_idx - 1, v_idx - 1)];
                        break;
                    case CELL_FLUID_NORTHEAST:
                        velocity_y[IDX(h_idx, v_idx)] = 0.0;
                        velocity_x[IDX(h_idx, v_idx)] = 0.0;
                        velocity_y[IDX(h_idx, v_idx) - 1] = -velocity_y[IDX(h_idx + 1, v_idx - 1)];
                        velocity_x[IDX(h_idx - 1, v_idx)] = -velocity_x[IDX(h_idx - 1, v_idx + 1)];
                        break;
                    case CELL_FLUID_SOUTHEAST:
                        velocity_y[IDX(h_idx, v_idx) - 1] = 0.0;
                        velocity_x[IDX(h_idx, v_idx)] = 0.0;
                        velocity_y[IDX(h_idx, v_idx)] = -velocity_y[IDX(h_idx + 1, v_idx)];
                        velocity_x[IDX(h_idx - 1, v_idx)] = -velocity_x[IDX(h_idx - 1, v_idx - 1)];
                        break;
                    case CELL_FLUID_SOUTHWEST:
                        velocity_y[IDX(h_idx, v_idx) - 1] = 0.0;
                        velocity_x[IDX(h_idx - 1, v_idx)] = 0.0;
                        velocity_y[IDX(h_idx, v_idx)] = -velocity_y[IDX(h_idx - 1, v_idx)];
                        velocity_x[IDX(h_idx, v_idx)] = -velocity_x[IDX(h_idx, v_idx) - 1];
                        break;
                    case CELL_FLUID_NORTHWEST:
                        velocity_y[IDX(h_idx, v_idx)] = 0.0;
                        velocity_x[IDX(h_idx - 1, v_idx)] = 0.0;
                        velocity_y[IDX(h_idx, v_idx) - 1] = -velocity_y[IDX(h_idx - 1, v_idx - 1)];
                        velocity_x[IDX(h_idx, v_idx)] = -velocity_x[IDX(h_idx, v_idx) + 1];
                        break;
                    default:;
                    }

        /*
         * If we're on a western boundary, fix the western-edge velocities such that there is a continual flow of fluid
         * into the simulation space.
         */
        velocity_y[0] = 2 * metadata->initial_velocity_y - velocity_y[IDX(1, 0)];

        for (indexer_t v_idx = 1; v_idx < interior_extents.y; ++v_idx) {
            velocity_x[IDX(0, v_idx)] = metadata->initial_velocity_x;
            velocity_y[IDX(0, v_idx)] = 2 * metadata->initial_velocity_y - velocity_y[IDX(1, v_idx)];
        }
    }

    void compute_tentative_velocities() const
    {
        static const compute_t reynolds = 500.0;
        static const compute_t gamma = 0.9; // Upwind differencing factor in PDE discretisation

        const compute_t quarter_resolution = metadata->resolution / 4.0;
        const compute_t sq_resolution = metadata->resolution * metadata->resolution;

        const dim2 interior_extents = {
            metadata->extents.x - 1,
            metadata->extents.y - 1,
        };

        // X tentative velocities
        for (indexer_t h_idx = 1; h_idx < interior_extents.x - 1; ++h_idx)
            for (indexer_t v_idx = 1; v_idx < interior_extents.y; ++v_idx)
                if (flags[IDX(h_idx, v_idx)] & CELL_FLUID && flags[IDX(h_idx + 1, v_idx)] & CELL_FLUID) {

                    const compute_t self_advection_x =
                        (
                            (velocity_x[IDX(h_idx, v_idx)] + velocity_x[IDX(h_idx + 1, v_idx)]) *
                            (velocity_x[IDX(h_idx, v_idx)] + velocity_x[IDX(h_idx + 1, v_idx)]) +
                            gamma * fabs(velocity_x[IDX(h_idx, v_idx)] + velocity_x[IDX(h_idx + 1, v_idx)]) *
                            (velocity_x[IDX(h_idx, v_idx)] - velocity_x[IDX(h_idx + 1, v_idx)]) -
                            (velocity_x[IDX(h_idx - 1, v_idx)] + velocity_x[IDX(h_idx, v_idx)]) *
                            (velocity_x[IDX(h_idx - 1, v_idx)] + velocity_x[IDX(h_idx, v_idx)]) -
                            gamma * fabs(velocity_x[IDX(h_idx - 1, v_idx)] + velocity_x[IDX(h_idx, v_idx)]) *
                            (velocity_x[IDX(h_idx - 1, v_idx)] - velocity_x[IDX(h_idx, v_idx)])
                        ) * quarter_resolution;

                    const compute_t cross_advection_y =
                        (
                            (velocity_y[IDX(h_idx, v_idx)] + velocity_y[IDX(h_idx + 1, v_idx)]) *
                            (velocity_x[IDX(h_idx, v_idx)] + velocity_x[IDX(h_idx, v_idx + 1)]) +
                            gamma * fabs(velocity_y[IDX(h_idx, v_idx)] + velocity_y[IDX(h_idx + 1, v_idx)]) *
                            (velocity_x[IDX(h_idx, v_idx)] - velocity_x[IDX(h_idx, v_idx + 1)]) -
                            (velocity_y[IDX(h_idx, v_idx - 1)] + velocity_y[IDX(h_idx + 1, v_idx - 1)]) *
                            (velocity_x[IDX(h_idx, v_idx - 1)] + velocity_x[IDX(h_idx, v_idx)]) -
                            gamma * fabs(velocity_y[IDX(h_idx, v_idx - 1)] +
                                velocity_y[IDX(h_idx + 1, v_idx - 1)]) *
                            (velocity_x[IDX(h_idx, v_idx - 1)] - velocity_x[IDX(h_idx, v_idx)])
                        ) * quarter_resolution;

                    const compute_t diffusion =
                        (
                            velocity_x[IDX(h_idx + 1, v_idx)] -
                            2.0 * velocity_x[IDX(h_idx, v_idx)] +
                            velocity_x[IDX(h_idx - 1, v_idx)] +
                            velocity_x[IDX(h_idx, v_idx + 1)] -
                            2.0 * velocity_x[IDX(h_idx, v_idx)] +
                            velocity_x[IDX(h_idx, v_idx - 1)]
                        ) * sq_resolution;

                    tentative_velocity_x[IDX(h_idx, v_idx)] = velocity_x[IDX(h_idx, v_idx)] + metadata->timestep_duration *
                        (diffusion / reynolds - self_advection_x - cross_advection_y);

                } else
                    // If both adjacent cells are not fluids, the velocity is unchanged.
                    tentative_velocity_x[IDX(h_idx, v_idx)] = velocity_x[IDX(h_idx, v_idx)];

        // Y velocities
        for (indexer_t h_idx = 1; h_idx < interior_extents.x; ++h_idx)
            for (indexer_t v_idx = 1; v_idx < interior_extents.y - 1; ++v_idx)
                if (flags[IDX(h_idx, v_idx)] & CELL_FLUID && flags[IDX(h_idx, v_idx + 1)] & CELL_FLUID) {

                    const compute_t cross_advection_x =
                        (
                            (velocity_x[IDX(h_idx, v_idx)] + velocity_x[IDX(h_idx, v_idx + 1)]) *
                            (velocity_y[IDX(h_idx, v_idx)] + velocity_y[IDX(h_idx + 1, v_idx)]) +
                            gamma * fabs(velocity_x[IDX(h_idx, v_idx)] + velocity_x[IDX(h_idx, v_idx + 1)]) *
                            (velocity_y[IDX(h_idx, v_idx)] - velocity_y[IDX(h_idx + 1, v_idx)]) -
                            (velocity_x[IDX(h_idx - 1, v_idx)] + velocity_x[IDX(h_idx - 1, v_idx + 1)]) *
                            (velocity_y[IDX(h_idx - 1, v_idx)] + velocity_y[IDX(h_idx, v_idx)]) -
                            gamma * fabs(velocity_x[IDX(h_idx - 1, v_idx)] +
                                velocity_x[IDX(h_idx - 1, v_idx + 1)]) *
                            (velocity_y[IDX(h_idx - 1, v_idx)] - velocity_y[IDX(h_idx, v_idx)])
                        ) * quarter_resolution;

                    const compute_t self_advection_y =
                        (
                            (velocity_y[IDX(h_idx, v_idx)] + velocity_y[IDX(h_idx, v_idx + 1)]) *
                            (velocity_y[IDX(h_idx, v_idx)] + velocity_y[IDX(h_idx, v_idx + 1)]) +
                            gamma * fabs(velocity_y[IDX(h_idx, v_idx)] + velocity_y[IDX(h_idx, v_idx + 1)]) *
                            (velocity_y[IDX(h_idx, v_idx)] - velocity_y[IDX(h_idx, v_idx + 1)]) -
                            (velocity_y[IDX(h_idx, v_idx - 1)] + velocity_y[IDX(h_idx, v_idx)]) *
                            (velocity_y[IDX(h_idx, v_idx - 1)] + velocity_y[IDX(h_idx, v_idx)]) -
                            gamma * fabs(velocity_y[IDX(h_idx, v_idx - 1)] + velocity_y[IDX(h_idx, v_idx)]) *
                            (velocity_y[IDX(h_idx, v_idx - 1)] - velocity_y[IDX(h_idx, v_idx)])
                        ) * quarter_resolution;

                    const compute_t diffusion =
                        (
                            velocity_y[IDX(h_idx + 1, v_idx)] -
                            2.0 * velocity_y[IDX(h_idx, v_idx)] +
                            velocity_y[IDX(h_idx - 1, v_idx)] +
                            velocity_y[IDX(h_idx, v_idx + 1)] -
                            2.0 * velocity_y[IDX(h_idx, v_idx)] +
                            velocity_y[IDX(h_idx, v_idx - 1)]
                        ) * sq_resolution;

                    tentative_velocity_y[IDX(h_idx, v_idx)] = velocity_y[IDX(h_idx, v_idx)] + metadata->timestep_duration *
                        (diffusion / reynolds - cross_advection_x - self_advection_y);

                } else
                    // If both adjacent cells are not fluids, the velocity is unchanged.
                    tentative_velocity_y[IDX(h_idx, v_idx)] = velocity_y[IDX(h_idx, v_idx)];
    }
    
    void update_velocities() const
    {
        const compute_t pressure_diff_factor = metadata->timestep_duration * metadata->resolution;

        // X velocities
        for (indexer_t h_idx = 1; h_idx < metadata->extents.x - 2; ++h_idx)
            for (indexer_t v_idx = 1; v_idx < metadata->extents.y - 1; ++v_idx)
                if (flags[IDX(h_idx, v_idx)] & CELL_FLUID && flags[IDX(h_idx + 1, v_idx)] & CELL_FLUID)
                    velocity_x[IDX(h_idx, v_idx)] = tentative_velocity_x[IDX(h_idx, v_idx)] -
                        (pressure[IDX(h_idx + 1, v_idx)] - pressure[IDX(h_idx, v_idx)]) * pressure_diff_factor;

        // Y velocities
        for (indexer_t h_idx = 1; h_idx < metadata->extents.x - 1; ++h_idx)
            for (indexer_t v_idx = 1; v_idx < metadata->extents.y - 2; ++v_idx)
                if (flags[IDX(h_idx, v_idx)] & CELL_FLUID && flags[IDX(h_idx, v_idx + 1)] & CELL_FLUID)
                    velocity_y[IDX(h_idx, v_idx)] = tentative_velocity_y[IDX(h_idx, v_idx)] -
                        (pressure[IDX(h_idx, v_idx + 1)] - pressure[IDX(h_idx, v_idx)]) * pressure_diff_factor;
    }

private:
    std::shared_ptr<Metadata> metadata;

    static constexpr compute_t initial_velocity_x = 1.0;
    static constexpr compute_t initial_velocity_y = 0.0;
    static constexpr compute_t initial_pressure = 0.0;
    static constexpr cell_flags initial_flag = CELL_FLUID;

    compute_t * velocity_x = nullptr;
    compute_t * velocity_y = nullptr;
    compute_t * tentative_velocity_x = nullptr;
    compute_t * tentative_velocity_y = nullptr;
    compute_t * pressure = nullptr;
    compute_t * poisson_source = nullptr;
    cell_flags * flags = nullptr;
    
    unsigned int fluid_cell_count;

    bounds get_airfoil_v_bounds(const compute_t maximum_camber, const compute_t edge_distance,
        const compute_t thickness, const indexer_t h_idx) const
    {
        bounds boundaries;

        /*
         * Position along chord, normalised to [0, 1]. From here, 'x' is translated into the co-ordinate space of the
         * global problem, and not the region.
         */
        const compute_t x = static_cast<compute_t>(h_idx) / static_cast<compute_t>(metadata->resolution) - 0.5f;

        if (x < 0.0 || x > 1.0)
            return boundaries;

        /*
         * The midline distance is the half-thickness from the fixed 'x' to the horizontal central line of the airfoil.
         * It is the Euclidean distance from the 'x' co-ordinate to the midline. This is NACA standard formulae.
         */
        const compute_t x_sq = x * x;
        compute_t midline_distance = 5.0 * thickness *
                (0.2969 * sqrt(x) - 0.1260 * x - 0.3516 * x_sq + 0.2843 * x * x_sq - 0.1015 * x_sq * x_sq);

        /*
         * Compute the 'y' co-ordinate of the mean camber line, given the fixed 'x' position. This is NACA standard
         * formulae, represented as a piecewise map over 'x' in intervals [0, p] and (p, 1], where 'p' is the edge
         * distance.
         */
        const compute_t mean_camber_line_y = x <= edge_distance
                ? maximum_camber / (edge_distance * edge_distance) * (2.0 * edge_distance * x - x_sq)
                : // 0 <= x <= p
                maximum_camber / ((1.0 - edge_distance) * (1.0 - edge_distance)) * // p < x <= 1
                        (1.0 - 2.0 * edge_distance + 2.0 * edge_distance * x - x_sq);

        /*
         * Use standard calculus formulae to find the numerical derivative of the mean camber line 'y' co-ordinate.
         * Thickness is applied perpendicular to the mean camber line. Use standard geometric formulae to compute the 'y'
         * co-ordinates for the upper and lower camber surface lines.
         */
        const compute_t norm = x <= edge_distance
                ? 2.0 * maximum_camber / (edge_distance * edge_distance) * (edge_distance - x)
                : 2.0 * maximum_camber / ((1.0 - edge_distance) * (1.0 - edge_distance)) * (edge_distance - x);

        midline_distance *= cos(atan(norm));

        const compute_t upper_camber_y = mean_camber_line_y + midline_distance;
        const compute_t lower_camber_y = mean_camber_line_y - midline_distance;

        boundaries.begin = floor((lower_camber_y + metadata->problem_size.y / 2.0) * metadata->resolution);
        boundaries.end = ceil((upper_camber_y + metadata->problem_size.y / 2.0) * metadata->resolution);
        return boundaries;
    }
};

} // namespace owd

#endif // HIPC_ASSESSMENT_CUDA_SERIALPROTOTYPE_HPP
