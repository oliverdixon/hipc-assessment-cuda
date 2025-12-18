//
// Created by owd on 18/12/2025.
//

#include <fstream>

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

    return std::shared_ptr<Metadata>(
        metadata,
        [](Metadata * const target)
        {
            target->~Metadata();
            cudaFree(target);
        }
    );
}

}

int main()
{
    const auto metadata = owd::metadata_create();
    const owd::DeviceRegion device_region(metadata);

    // Do work...

    const owd::HostRegion host_region(metadata);
    device_region.populate_host_region(host_region);

    std::ofstream output_file("out/flows.vtr");
    host_region.vtk_serialise(output_file);

    return 0;
}
