#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "include/avalanche.hpp"

namespace py = pybind11;

PYBIND11_MODULE(sandpile, m) {
    py::class_<AvalancheData>(m, "AvalancheData")
        .def(py::init<uint32_t>())
        .def_readwrite("time_step", &AvalancheData::time_step)
        .def_readwrite("size", &AvalancheData::size)
        .def_readwrite("time", &AvalancheData::time)
        .def_readwrite("reach", &AvalancheData::reach)
        .def_readwrite("dissipation_rate", &AvalancheData::dissipation_rate);
}