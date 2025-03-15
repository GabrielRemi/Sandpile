#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
// #include <cmath>
#include <functional>
#include <print>
#include <iostream>
#include <fstream>

namespace py = pybind11;

struct SystemMeta
{
    uint8_t dim;
    uint8_t grid;
    uint8_t crit_slope;
    bool closed_boundary;

    SystemMeta(uint8_t _d, uint8_t _g, uint8_t _c, uint8_t _b) : dim(_d), grid(_g), crit_slope(_c), closed_boundary(_b) {}
};

struct AvalancheData
{
    uint32_t time_step;
    uint32_t size;
    uint32_t time;
    double reach;
    py::array_t<uint8_t> dissipation_rate;

    AvalancheData(uint32_t _t) : time_step(_t)
    {
        size = 0;
        time = 0;
        reach = 0;
        dissipation_rate = py::array_t<uint8_t>(0);
    }
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
std::vector<py::array_t<uint8_t>> get_critical_points(py::array_t<T> &cfg, SystemMeta &meta)
{
    std::vector<py::array_t<uint8_t>> points(0);
    auto buf = cfg.template unchecked<1>();

    for (ssize_t i = 0; i < buf.shape(0); ++i)
    {
        auto slope = buf(i);
        if (static_cast<uint32_t>(slope) > static_cast<uint32_t>(meta.crit_slope))
        {
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

template <typename T>
AvalancheData relax_avalanche(uint32_t time_step, py::array_t<T> &start_cfg, py::array_t<uint8_t> &start_point,
                              SystemMeta &system)
{
    std::vector<uint8_t> dissipation_rate(0);
    AvalancheData avalanche(time_step);
    auto file = std::ofstream("data.log");

    int max_step = 500;
    int i = 0;
    std::function<void(py::array_t<T> &, py::array_t<uint8_t> &, uint8_t)> relax;

    if (system.closed_boundary)
    {
        relax = cl_bound_system_relax<T>;
    }
    else
    {
        relax = op_bound_system_relax<T>;
    }

    for (i = 0; i < max_step; ++i)
    {
        auto critical_points = get_critical_points(start_cfg, system);
        if (critical_points.size() == 0)
        {
            break;
        }
        // printing
        file << std::format("iteration [{}], number of points [{}]", i, critical_points.size()) << std::endl;

        avalanche.size += static_cast<uint32_t>(critical_points.size());
        avalanche.time += 1;
        dissipation_rate.push_back(static_cast<uint8_t>(critical_points.size()));

        for (auto &critical_point : critical_points)
        {
            // printing
            auto cfg_buf = start_cfg.template unchecked<1>();
            auto index = ravel_index(critical_point, system.grid);
            file << std::format("[{}] value: {}", index, static_cast<int>(cfg_buf(index))) << std::endl;

            auto buf = critical_point.template unchecked<1>();
            auto start_buf = start_point.unchecked<1>();
            double temp = 0.;
            for (ssize_t j = 0; j < buf.shape(0); ++j)
            {
                temp += static_cast<double>(pow(start_buf(j) - buf(j), 2));
            }
            temp = static_cast<double>(sqrt(temp));
            avalanche.reach = std::max(avalanche.reach, temp);

            relax(start_cfg, critical_point, system.grid);
        }
        // for (long unsigned int i = 0; i < critical_points.size(); ++i)
        // {
        //     relax(start_cfg, critical_points[i], system.grid);
        //     // break;
        // }
        if (i == 4)
        {
            break;
        }
    }

    if (i == (max_step - 1))
    {
        throw std::runtime_error("Max number of step iterations reached");
        file << "Error!!!" << std::endl;
    }

    avalanche.dissipation_rate = py::array_t<uint8_t>(dissipation_rate.size(), dissipation_rate.data());
    return avalanche;
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
    m.def("relax_avalanche", &relax_avalanche<int8_t>);
    m.def("relax_avalanche", &relax_avalanche<int16_t>);

    py::class_<SystemMeta>(m, "SystemMeta")
        .def(py::init<uint8_t, uint8_t, uint8_t, bool>())
        .def_readwrite("dim", &SystemMeta::dim)
        .def_readwrite("grid", &SystemMeta::grid)
        .def_readwrite("crit_slope", &SystemMeta::crit_slope)
        .def_readwrite("closed_bounday", &SystemMeta::closed_boundary);

    // Only for testing purposes. It is not going to be used on the python side
    py::class_<AvalancheData>(m, "AvalancheData")
        .def(py::init<uint32_t>())
        .def_readwrite("time_step", &AvalancheData::time_step)
        .def_readwrite("size", &AvalancheData::size)
        .def_readwrite("time", &AvalancheData::time)
        .def_readwrite("reach", &AvalancheData::reach)
        .def_readwrite("dissipation_rate", &AvalancheData::dissipation_rate);
}