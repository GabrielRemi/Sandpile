#pragma once
#include <Eigen/Core>

template <typename T>
using vector = Eigen::VectorX<T>;

inline uint64_t ravel_index(const vector<uint8_t>& multi_index, const uint8_t grid)
{
    uint64_t result = 0, power = 1;

    for (ssize_t i = multi_index.size() - 1; i >= 0; --i)
    {
        result += static_cast<uint64_t>(multi_index(i)) * power;
        power *= static_cast<uint64_t>(grid);
    }

    return result;
}

inline vector<uint8_t> unravel_index(uint64_t index, const uint8_t dim, const uint8_t grid)
{
    vector<uint8_t> result(dim);

    for (ssize_t i = dim - 1; i >= 0; --i)
    {
        result(i) = static_cast<uint8_t>(index % grid);
        index = static_cast<uint64_t>(floor(static_cast<double>(index) / static_cast<double>(grid)));
    }

    return result;
}

inline uint64_t shift_ravelled_index(const uint64_t index, const uint8_t dim, const uint8_t grid, const int64_t shift,
                                     const uint8_t shift_dim)
{
    const int64_t index_shift = shift * static_cast<int64_t>(pow(grid, dim - shift_dim - 1));
    if (index_shift < 0)
    {
        return index + index_shift;
    }
    else
    {
        return index + static_cast<uint64_t>(index_shift);
    }
}

template <typename T>
std::vector<vector<uint8_t>> get_critical_points(vector<T>& cfg, const uint8_t dim, const uint8_t grid,
                                                 const uint8_t crit_slope)
{
    std::vector<vector<uint8_t>> points(0);

    for (ssize_t i = 0; i < cfg.size(); ++i)
    {
        if (cfg(i) > static_cast<T>(crit_slope))
        {
            points.push_back(unravel_index(static_cast<uint64_t>(i), dim, grid));
        }
    }

    return points;
}

template <typename T>
void op_bound_system_relax(vector<T>& cfg, vector<uint8_t>& position_index, const uint8_t grid)
{
    const auto dim = static_cast<uint8_t>(position_index.size());
    auto raveled_index = ravel_index(position_index, grid);

    T boundary_indices = 0;
    for (ssize_t i = 0; i < dim; ++i)
    {
        if (position_index(i) == 0)
        {
            cfg(raveled_index) = 0;
            return;
        }
        else if (position_index(i) == grid - 1)
        {
            ++boundary_indices;
        }
    }

    cfg(raveled_index) += static_cast<T>(-2 * dim + boundary_indices);

    for (ssize_t i = 0; i < dim; ++i)
    {
        cfg(shift_ravelled_index(raveled_index, dim, grid, -1, static_cast<uint8_t>(i))) += 1;

        if (position_index(i) < (grid - 1))
        {
            cfg(shift_ravelled_index(raveled_index, dim, grid, 1, static_cast<uint8_t>(i))) += 1;
        }
    }
}

template <typename T>
void cl_bound_system_relax(vector<T>& cfg, vector<uint8_t>& position_index, const uint8_t grid)
{
    const auto dim = static_cast<uint8_t>(position_index.size());

    T boundary_indices = 0;
    auto ravelled_index = ravel_index(position_index, grid);
    for (ssize_t i = 0; i < dim; ++i)
    {
        if (position_index(i) == 0 || position_index(i) == (grid - 1))
        {
            cfg(ravelled_index) = 0;
            return;
        }
    }

    cfg(ravelled_index) += static_cast<T>(-2 * dim);
    for (ssize_t i = 0; i < dim; ++i)
    {
        cfg(shift_ravelled_index(ravelled_index, dim, grid, -1, static_cast<uint8_t>(i))) += 1;

        cfg(shift_ravelled_index(ravelled_index, dim, grid, 1, static_cast<uint8_t>(i))) += 1;
    }
}
