#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
// #include <cmath>
#include <print>
#include <iostream>

namespace py = pybind11;

struct SystemMeta
{
    uint8_t dim;
    uint8_t grid;
    uint8_t crit_slope;
    bool closed_boundary;

    SystemMeta(uint8_t _d, uint8_t _g, uint8_t _c, uint8_t _b) : dim(_d), grid(_g), crit_slope(_c), closed_boundary(_b) {}
};

uint64_t ravel_index(py::array_t<uint8_t> &multi_index, uint8_t grid)
{
    uint64_t result = 0, curr_pow = 0;

    auto buf = multi_index.unchecked<1>();
    for (ssize_t i = buf.shape(0) - 1; i >= 0; --i)
    {
        result += static_cast<uint64_t>(buf(i)) * static_cast<uint64_t>(std::pow(static_cast<uint64_t>(grid), curr_pow));
        // result += buf(i) * std::pow(grid, curr_pow);
        curr_pow += 1;
    }

    return result;
}

py::array_t<uint8_t> unravel_index(uint64_t index, uint8_t dim, uint8_t grid)
{
    py::array_t<uint8_t> result(dim);
    auto buf = result.mutable_unchecked<1>();

    for (ssize_t i = dim - 1; i >= 0; --i)
    {
        buf(i) = static_cast<uint8_t>(index % grid);
        index = static_cast<uint64_t>(floor(index / grid));
    }

    return result;
}

template <typename T>
// py::array_t<py::array_t<uint8_t>> get_critical_points(py::array_t<T> cfg, SystemMeta &meta)
std::vector<py::array_t<uint8_t>> get_critical_points(py::array_t<T> &cfg, SystemMeta &meta)
{
    std::vector<py::array_t<uint8_t>> points(0);
    auto buf = cfg.template unchecked<1>();

    for (ssize_t i = 0; i < buf.shape(0); ++i)
    {
        auto slope = buf(i);
        if (static_cast<uint32_t>(slope) > static_cast<uint32_t>(meta.crit_slope))
        {
            std::cout << i << std::endl;
            points.push_back(unravel_index(static_cast<uint64_t>(i), meta.dim, meta.grid));
        }
    }

    return points;
    // TODO do not make a copy of the vector
}

template <typename T>
void op_bound_system_relax(py::array_t<T> &cfg, py::array_t<uint8_t> &position_index, uint8_t grid)
{
    auto cfg_buf = cfg.template mutable_unchecked<1>();

    auto pos_buf = position_index.mutable_unchecked<1>();
    T dim = static_cast<T>(pos_buf.shape(0));

    T boundary_indices = 0;
    for (ssize_t i = 0; i < dim; ++i)
    {
        if (pos_buf(i) == 0)
        {
            cfg_buf(ravel_index(position_index, grid)) = 0;
            return;
        }
        else if (pos_buf(i) == grid - 1)
        {
            ++boundary_indices;
        }
    }

    cfg_buf(ravel_index(position_index, grid)) += static_cast<T>(-2 * dim + boundary_indices);

    for (ssize_t i = 0; i < dim; ++i)
    {
        pos_buf(i) -= 1;
        cfg_buf(ravel_index(position_index, grid)) += 1;

        pos_buf(i) += 1;
        if (pos_buf(i) < (grid - 1))
        {
            pos_buf(i) += 1;
            cfg_buf(ravel_index(position_index, grid)) += 1;
            pos_buf(i) -= 1;
        }
    }
}

template <typename T>
void cl_bound_system_relax(py::array_t<T> &cfg, py::array_t<uint8_t> &position_index, uint8_t grid)
{

    auto cfg_buf = cfg.template mutable_unchecked<1>();

    auto pos_buf = position_index.mutable_unchecked<1>();
    T dim = static_cast<T>(pos_buf.shape(0));

    T boundary_indices = 0;
    for (ssize_t i = 0; i < dim; ++i)
    {
        if (pos_buf(i) == 0 || pos_buf(i) == grid - 1)
        {
            cfg_buf(ravel_index(position_index, grid)) = 0;
            return;
        }
    }

    cfg_buf(ravel_index(position_index, grid)) += static_cast<T>(-2 * dim);
    for (ssize_t i = 0; i < dim; ++i)
    {
        pos_buf(i) -= 1;
        cfg_buf(ravel_index(position_index, grid)) += 1;

        pos_buf(i) += 2;
        cfg_buf(ravel_index(position_index, grid)) += 1;

        pos_buf(i) -= 1;
    }
}

PYBIND11_MODULE(avalanche, m)
{
    m.def("ravel_index", &ravel_index);
    m.def("unravel_index", &unravel_index);
    m.def("get_critical_points", &get_critical_points<int8_t>);
    m.def("get_critical_points", &get_critical_points<int16_t>);
    m.def("op_bound_system_relax", &op_bound_system_relax<int8_t>);
    m.def("op_bound_system_relax", &op_bound_system_relax<int16_t>);
    m.def("cl_bound_system_relax", &cl_bound_system_relax<int8_t>);
    m.def("cl_bound_system_relax", &cl_bound_system_relax<int16_t>);

    py::class_<SystemMeta>(m, "SystemMeta")
        .def(py::init<uint8_t, uint8_t, uint8_t, bool>())
        .def_readwrite("dim", &SystemMeta::dim)
        .def_readwrite("grid", &SystemMeta::grid)
        .def_readwrite("crit_slope", &SystemMeta::crit_slope)
        .def_readwrite("closed_bounday", &SystemMeta::closed_boundary);
}