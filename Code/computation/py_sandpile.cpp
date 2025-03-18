#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include <sandpile.hpp>

namespace py = pybind11;

template <typename T>
void bind_sandpile(py::module_& m, const std::string name)
{
    py::class_<Sandpile<T>>(m, name.c_str())
        .def(py::init<uint8_t, uint8_t, uint8_t>(),
             py::arg("dim"),
             py::arg("grid"),
             py::arg("crit_slope"),
             R"(
A Class for simulating Sandpiles. Can be used with any dimension and grid size, as long
as the number of points inside the grid is small enough for to work.

:param dim: Dimension of the system
:param grid: Grid size per dimension
:param crit_slope: Critical Slope. If the slope value on lattice point is above this value, the system relaxes.
)")
        .def(py::init<uint8_t, uint8_t, uint8_t, bool, bool>(),
             py::arg("dim"),
             py::arg("grid"),
             py::arg("crit_slope"),
             py::arg("has_open_boundary"),
             py::arg("has_conservative_perturbation"),
             R"(
A Class for simulating Sandpiles. Can be used with any dimension and grid size, as long
as the number of points inside the grid is small enough for to work.

:param dim: Dimension of the system
:param grid: Grid size per dimension
:param crit_slope: Critical Slope. If the slope value on lattice point is above this value, the system relaxes.
:param has_open_boundary: Specify if the system uses open boundaries for relaxation.
:param has_conservative_perturbation: Specify if the system uses conservative perturbations for relaxations.
)")
        .def_readwrite("dim", &Sandpile<T>::dim)
        .def_readwrite("grid", &Sandpile<T>::grid)
        .def_readwrite("crit_slope", &Sandpile<T>::crit_slope)
        .def_readwrite("time_cuf_off", &Sandpile<T>::time_cut_off,
                       "All avalanche data registered before this time step will be ignored")
        .def("get_average_slopes", &Sandpile<T>::get_average_slopes,
             "Array of average slopes calculated during the last simulation.",
             py::return_value_policy::reference_internal)
        .def("get_size", &Sandpile<T>::get_size, py::return_value_policy::reference_internal)
        .def("get_time", &Sandpile<T>::get_time, py::return_value_policy::reference_internal)
        .def("get_reach", &Sandpile<T>::get_reach, py::return_value_policy::reference_internal)
        // This does not work
        // .def("dissipation_rate", &Sandpile<T>::dissipation_rate)
        .def("generate_total_dissipation_rate", &Sandpile<T>::generate_total_dissipation_rate,
             py::return_value_policy::reference_internal,
             py::arg("time_steps"),
             py::arg("seed") = std::nullopt,
             R"(
Generate the total dissipation rate by randomly placing the individual dissipation rates on
a grid.

:param time_steps: defines the time scale of the total dissipation rate.
:param seed: Uses this seed for random number generation.

:return: total dissipation rate
)")
        .def("get_has_open_boundary", &Sandpile<T>::get_has_open_boundary)
        .def("get_has_conservative_perturbation", &Sandpile<T>::get_has_conservative_perturbation)
        // .def("set_has_open_boundary", &Sandpile<T>::set_has_open_boundary)
        // .def("set_has_conservative_perturbation", &Sandpile<T>::set_has_conservative_perturbation)
        .def("initialize_system", &Sandpile<T>::initialize_system,
             py::arg("time_steps"),
             py::arg("start_cfg") = std::nullopt,
             py::arg("seed") = std::nullopt,
             R"(
Initialize the system to start a simulation. You need to run this function every time
you start a manual simulation with the `step()` function.

:param time_steps: The number of time steps to simulate.
:param start_cfg: Initial configuration of the system.
:param seed: Random seed for reproducibility.
)")
        .def("step", &Sandpile<T>::step,
             py::arg("perturb_position") = std::nullopt,
             R"(
Make one perturbation step of the system.

:param perturb_position: The position of the perturbation. If not specified, perturb the system at random.
)")
        .def("simulate", &Sandpile<T>::simulate,
             py::arg("time_steps"),
             py::arg("start_cfg") = std::nullopt,
             py::arg("seed") = std::nullopt,
             R"(
Simulates the system over a given number of time steps.

:param time_steps: The number of time steps to simulate.
:param start_cfg: Initial configuration of the system.
:param seed: Random seed for reproducibility.

Example:
```python
from cpp_computation import Sandpile
system = Sandpile(2, 20, 7)
system.simulate(100)
print(system.average_slopes)
)");
}

PYBIND11_MODULE(sandpile, m)
{
    // py::class_<AvalancheData>(m, "AvalancheData")
    //     .def(py::init<uint32_t>())
    //     .def_readwrite("time_step", &AvalancheData::time_step)
    //     .def_readwrite("size", &AvalancheData::size)
    //     .def_readwrite("time", &AvalancheData::time)
    //     .def_readwrite("reach", &AvalancheData::reach)
    //     .def_readwrite("dissipation_rate", &AvalancheData::dissipation_rate);
    bind_sandpile<int8_t>(m, "Sandpile8Bit");
    bind_sandpile<int16_t>(m, "Sandpile16Bit");
    // bind_sandpile<int32_t>(m, "Sandpile32Bit");
}
