#pragma once
#include <Eigen/Dense>

template <typename T>
using vector = Eigen::VectorX<T>;
// template <typename T> using vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

struct AvalancheData
{
    uint32_t time_step;
    uint32_t size;
    uint32_t time;
    double reach;
    vector<uint16_t> dissipation_rate;

    explicit AvalancheData(const uint32_t _t) : time_step(_t)
    {
        size = 0;
        time = 0;
        reach = 0;
        dissipation_rate = vector<uint16_t>(0);
    }
};

inline uint64_t ravel_index(const vector<uint8_t>& multi_index, const uint8_t grid)
{
    uint64_t result = 0, curr_pow = 0;

    for (ssize_t i = multi_index.size() - 1; i >= 0; --i)
    {
        result += static_cast<uint64_t>(multi_index(i)) *
            static_cast<uint64_t>(std::pow(static_cast<uint64_t>(grid), curr_pow));
        curr_pow += 1;
    }

    return result;
}

inline vector<uint8_t> unravel_index(uint64_t index, uint8_t dim, uint8_t grid)
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
    // return static_cast<uint64_t>(static_cast<int16_t>(index) +
    //     shift * static_cast<int16_t>(pow(grid, dim - shift_dim - 1)));
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
        auto slope = cfg(i);
        if (static_cast<int16_t>(slope) > static_cast<int16_t>(crit_slope))
        {
            points.push_back(unravel_index(static_cast<uint64_t>(i), dim, grid));
        }
    }

    return points;
}

template <typename T>
void op_bound_system_relax(vector<T>& cfg, vector<uint8_t>& position_index, const uint8_t grid)
{
    T dim = static_cast<T>(position_index.size());
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

    cfg(ravel_index(position_index, grid)) += static_cast<T>(-2 * dim + boundary_indices);

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
    T dim = static_cast<T>(position_index.size());

    T boundary_indices = 0;
    auto ravelled_index = ravel_index(position_index, grid);
    for (ssize_t i = 0; i < dim; ++i)
    {
        if (position_index(i) == 0 || position_index(i) == grid - 1)
        {
            cfg(ravelled_index) = 0;
            return;
        }
    }

    cfg(ravel_index(position_index, grid)) += static_cast<T>(-2 * dim);
    for (ssize_t i = 0; i < dim; ++i)
    {
        cfg(shift_ravelled_index(ravelled_index, dim, grid, -1, static_cast<uint8_t>(i))) += 1;

        cfg(shift_ravelled_index(ravelled_index, dim, grid, 1, static_cast<uint8_t>(i))) += 1;
    }
}
