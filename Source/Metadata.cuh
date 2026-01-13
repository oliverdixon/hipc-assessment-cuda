//
// Created by owd on 18/12/2025.
//

#ifndef HIPC_ASSESSMENT_CUDA_METADATA_CUH
#define HIPC_ASSESSMENT_CUDA_METADATA_CUH

#include "types.h"

namespace owd
{

struct Metadata
{
    struct NACASpecifier
    {
        NACASpecifier(const unsigned char maximum_camber, const unsigned char edge_distance,
                const unsigned char maximum_thickness) noexcept :
            maximum_camber(maximum_camber),
            edge_distance(edge_distance),
            maximum_thickness(maximum_thickness)
        { }

        unsigned char maximum_camber;
        unsigned char edge_distance;
        unsigned char maximum_thickness;
    };

    Metadata(const unsigned int resolution, const NACASpecifier naca_specifier, const compute_dim2 problem_size,
            const compute_t timestep_duration) noexcept :
        extents(
            static_cast<indexer_t>(std::ceil(problem_size.x * resolution) + 2),
            static_cast<indexer_t>(std::ceil(problem_size.y * resolution) + 2)
        ),
        resolution(resolution),
        naca_specifier(naca_specifier),
        problem_size(problem_size),
        allocation_count(extents.x * extents.y),
        allocation_byte_count(allocation_count * sizeof(compute_t)),
        timestep_duration(timestep_duration)
    { }

    const dim2 extents;
    const unsigned int resolution;
    const NACASpecifier naca_specifier;
    const compute_dim2 problem_size;
    const std::size_t allocation_count;
    const std::size_t allocation_byte_count;

    static constexpr compute_t initial_velocity_x = 1.0;
    static constexpr compute_t initial_velocity_y = 0.0;
    static constexpr compute_t initial_pressure = 0.0;

    compute_t timestep_duration;
};

} // namespace owd

#endif // HIPC_ASSESSMENT_CUDA_METADATA_CUH
