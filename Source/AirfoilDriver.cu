//
// Created by owd on 18/12/2025.
//

#include <fstream>
#include <iostream>

#include "DeviceRegion.cuh"
#include "HostRegion.cuh"
#include "SafeCUDA.cuh"

namespace owd
{

static std::shared_ptr<Metadata> metadata_create()
{
    Metadata * metadata;
    SafeCUDA(cudaMallocManaged(&metadata, sizeof(Metadata)));
    new (metadata) Metadata(128, Metadata::NACASpecifier(2, 4, 12), compute_dim2(4.0, 1.0), 0.003);

    return {
        metadata,
        [](Metadata * const target)
        {
            target->~Metadata();
            cudaFree(target);
        }
    };
}

}

int main()
{
    const auto metadata = owd::metadata_create();
    const owd::DeviceRegion device_region(metadata);

    const compute_t max_simulation_runtime = metadata->timestep_duration * 10; // TODO
    static constexpr indexer_t sor_max_iterations = 100;
    static constexpr compute_t sor_residual_epsilon = 0.001;
    static constexpr indexer_t output_freq = 100;

    compute_t simulation_runtime = 0.0;
    indexer_t step_iteration = 0;

    while (simulation_runtime < max_simulation_runtime) {
        device_region.apply_boundary_conditions();
        device_region.compute_tentative_velocities();
        device_region.compute_poisson_source();

        compute_t residual = std::numeric_limits<compute_t>::max();

        for (indexer_t sor_iteration = 0; sor_iteration < sor_max_iterations; ++sor_iteration) {
            device_region.perform_sor_cycle();
#if 0 // TODO
            residual = device_region.compute_residual_norm_sq();
            if (std::fabs(residual) < sor_residual_epsilon * sor_residual_epsilon)
                break; // SOR has converged.
#endif
        }

        device_region.update_velocities();
        simulation_runtime += metadata->timestep_duration;

        if (step_iteration % output_freq == 0)
            std::cout << "Step: " << step_iteration << ", Time: " << simulation_runtime << ", Residual: " << residual <<
                std::endl;

        ++step_iteration;
    }

    const owd::HostRegion host_region(metadata);
    device_region.populate_host_region(host_region);

    std::ofstream output_file("out/flows.vtr");
    host_region.vtk_serialise(output_file);

    return 0;
}
