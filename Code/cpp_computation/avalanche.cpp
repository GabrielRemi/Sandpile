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

uint64_t ravel_index(py::array_t<uint8_t> multi_index, uint8_t grid)
{
    uint64_t result = 0, curr_pow = 0;

    auto buf = multi_index.unchecked<1>();
    for (ssize_t i = buf.shape(0) - 1; i >= 0; --i)
    {
        result += static_cast<uint64_t>(buf(i)) * std::pow(static_cast<uint64_t>(grid), curr_pow);
        // result += buf(i) * std::pow(grid, curr_pow);
        curr_pow += 1;
    }

    return result;
}

template <typename T>
std::vector<py::array_t<uint8_t>> get_critical_points(py::array_t<T>, SystemMeta &meta)
{
    std::vector<py::array_t<uint8_t>> points;
    points.push_back(py::array_t<uint8_t>({1, 2, 3}));

    // TODO stuff

    return points;
}

PYBIND11_MODULE(avalanche, m)
{
    m.def("ravel_index", &ravel_index);
    m.def("get_critical_points", &get_critical_points<int8_t>);
    m.def("get_critical_points", &get_critical_points<int16_t>);

    py::class_<SystemMeta>(m, "SystemMeta")
        .def(py::init<uint8_t, uint8_t, uint8_t, bool>())
        .def_readwrite("dim", &SystemMeta::dim)
        .def_readwrite("grid", &SystemMeta::grid)
        .def_readwrite("crit_slope", &SystemMeta::crit_slope)
        .def_readwrite("closed_bounday", &SystemMeta::closed_boundary);
}