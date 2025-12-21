//
// Created by owd on 18/12/2025.
//

#include "HostRegion.cuh"

#include <iostream>

#include "SafeCUDA.cuh"

namespace owd
{

HostRegion::HostRegion(std::shared_ptr<Metadata> metadata) :
    metadata(std::move(metadata))
{
    const std::size_t allocation_count = this->metadata->allocation_count;

    velocity_x = new compute_t[allocation_count];
    velocity_y = new compute_t[allocation_count];
    pressure = new compute_t[allocation_count];
}

HostRegion::~HostRegion()
{
    delete[] pressure;
    delete[] velocity_y;
    delete[] velocity_x;
}

void HostRegion::receive_velocity_x(const compute_t * const velocity_x_source) const
{
    SafeCUDA(cudaMemcpy(velocity_x, velocity_x_source, metadata->allocation_byte_count, cudaMemcpyDeviceToHost));
}

void HostRegion::receive_velocity_y(const compute_t * const velocity_y_source) const
{
    SafeCUDA(cudaMemcpy(velocity_y, velocity_y_source, metadata->allocation_byte_count, cudaMemcpyDeviceToHost));
}

void HostRegion::receive_pressure(const compute_t * const pressure_source) const
{
    SafeCUDA(cudaMemcpy(pressure, pressure_source, metadata->allocation_byte_count, cudaMemcpyDeviceToHost));
}

void HostRegion::vtk_serialise(std::ostream &ostream) const
{
    // Prologue
    ostream << "<?xml version=\"1.0\"?>\n";
    ostream << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";

    const auto h_pixel_count = metadata->extents.x - 1;
    const auto v_pixel_count = metadata->extents.y - 1;

    // Grid consisting of singular piece
    ostream << "\t<RectilinearGrid WholeExtent=\"0 " << h_pixel_count << " 0 " << v_pixel_count <<
        " 0 0\" GhostLevel=\"0\">\n";
    ostream << "\t\t<Piece Extent=\"0 " << h_pixel_count << " 0 " << v_pixel_count << " 0 0\">\n";

    // Physical positions of X and Y co-ordinates
    ostream << "\t\t\t<Coordinates>\n";

    const auto problem_space_width = metadata->problem_size.x;
    ostream << "\t\t\t\t<DataArray type=\"Float64\" name=\"X\" format=\"ascii\" RangeMin=\"0\" RangeMax=\"" <<
        problem_space_width << "\">\n";
    for (indexer_t h_idx = 0; h_idx <= h_pixel_count; ++h_idx)
        ostream << problem_space_width / h_pixel_count * h_idx << ' ';

    ostream << "\n\t\t\t\t</DataArray>\n";

    const auto problem_space_height = metadata->problem_size.y;
    ostream << "\t\t\t\t<DataArray type=\"Float64\" name=\"Y\" format=\"ascii\" RangeMin=\"0\" RangeMax=\"" <<
        problem_space_height << "\">\n";
    for (indexer_t v_idx = 0; v_idx <= v_pixel_count; ++v_idx)
        ostream << problem_space_height / v_pixel_count * v_idx << ' ';

    ostream << "\n\t\t\t\t</DataArray>\n";
    ostream << "\t\t\t\t<DataArray type=\"Float64\" name=\"Z\" format=\"ascii\">0.0</DataArray>\n";
    ostream << "\t\t\t</Coordinates>\n";

    // Velocity vectors
    ostream << "\t\t\t<PointData Vectors=\"uv\">\n";
    ostream << "\t\t\t\t<DataArray type=\"Float64\" Name=\"uv\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    const auto row_extent = metadata->extents.x;

    for (indexer_t v_idx = 0; v_idx <= v_pixel_count; ++v_idx) {
        const indexer_t v_basis = v_idx * row_extent;
        for (indexer_t h_idx = 0; h_idx <= h_pixel_count; ++h_idx)
            ostream << velocity_x[v_basis + h_idx] << ' ' << velocity_y[v_basis + h_idx] << " 0\n";
    }

    ostream << "\t\t\t\t</DataArray>\n";
    ostream << "\t\t\t</PointData>\n";

    // Pressure scalars
    ostream << "\t\t\t<CellData Scalars=\"p\">\n";
    ostream << "\t\t\t\t<DataArray type=\"Float64\" format=\"ascii\" Name=\"p\">\n";

    for (indexer_t v_idx = 0; v_idx < v_pixel_count; ++v_idx) {
        const indexer_t v_basis = v_idx * row_extent;
        for (indexer_t h_idx = 0; h_idx < h_pixel_count; ++h_idx)
            ostream << pressure[v_basis + h_idx] << ' ';
        ostream << '\n';
    }

    ostream << "\t\t\t\t</DataArray>\n";
    ostream << "\t\t\t</CellData>\n";

    // Epilogue
    ostream << "\t\t</Piece>\n";
    ostream << "\t</RectilinearGrid>\n";
    ostream << "</VTKFile>\n";
}

} // namespace owd
