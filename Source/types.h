//
// Created by od641 on 10/12/2025.
//

#ifndef HIPC_ASSESSMENT_TYPES_H
#define HIPC_ASSESSMENT_TYPES_H

typedef double compute_t;
typedef unsigned int indexer_t;

struct bounds
{
    __host__ __device__ bounds() { }

    __host__ __device__ bounds(const indexer_t begin, const indexer_t end) noexcept :
        begin(begin),
        end(end)
    { }

    indexer_t begin = 0;
    indexer_t end = 0;
};

struct dim2
{
    __host__ __device__ dim2(const indexer_t x, const indexer_t y) noexcept :
        x(x),
        y(y)
    { }

    indexer_t x;
    indexer_t y;
};

struct compute_dim2
{
    __host__ __device__ compute_dim2(const compute_t x, const compute_t y) noexcept :
        x(x),
        y(y)
    { }

    compute_t x;
    compute_t y;
};

enum cell_flags
{
    CELL_BOUNDARY = 0, /**< Boundary cell */

    CELL_FLUID_NORTH = 1, /**< Boundary cell with fluid to the north */
    CELL_FLUID_SOUTH = 1 << 1, /**< Boundary cell with fluid to the south */
    CELL_FLUID_WEST = 1 << 2, /**< Boundary cell with fluid to the west */
    CELL_FLUID_EAST = 1 << 3, /**< Boundary cell with fluid to the east */

    CELL_FLUID_NORTHWEST = CELL_FLUID_NORTH | CELL_FLUID_WEST,
    CELL_FLUID_SOUTHWEST = CELL_FLUID_SOUTH | CELL_FLUID_WEST,
    CELL_FLUID_NORTHEAST = CELL_FLUID_NORTH | CELL_FLUID_EAST,
    CELL_FLUID_SOUTHEAST = CELL_FLUID_SOUTH | CELL_FLUID_EAST,
    CELL_FLUID_ALL = CELL_FLUID_NORTH | CELL_FLUID_SOUTH | CELL_FLUID_EAST | CELL_FLUID_WEST,

    CELL_FLUID = 1 << 4, /**< Fluid cell */
};

#endif // HIPC_ASSESSMENT_TYPES_H
