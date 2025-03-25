#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include <sandpile.hpp>

#define TQDM_UPDATE_STEPS 1000

namespace py = pybind11;

template <typename T>
void simulate_single(Sandpile<T>& system, const uint32_t time_steps,
                     int tqdm_update_steps, const int position)
{
    const py::dict kwargs{};
    kwargs["total"] = tqdm_update_steps;
    kwargs["position"] = position;
    kwargs["ncols"] = 100;

    const auto tqdm = py::module::import("tqdm").attr("tqdm")(**kwargs);

    if (!hasattr(tqdm, "update"))
    {
        throw py::attribute_error("tqdm object does not have update function");
    }
    tqdm.attr("n") = 0;

    system.initialize_system(time_steps, std::nullopt, std::nullopt);
    const int tqdm_update_step = static_cast<int>(time_steps / tqdm_update_steps);

    for (uint32_t i = 1; i <= time_steps; ++i)
    {
        system.step(std::nullopt);
        if (i % static_cast<uint32_t>(tqdm_update_step) == 0)
        {
            auto _ = tqdm.attr("update")(1);
            tqdm.attr("refresh")();
        }
    }

    // tqdm.attr("update")(static_cast<int>(time_steps % tqdm_update_step > 0));
    tqdm.attr("close")();

    system.shrink_to_fit();
}

template <typename T>
void sandpile_simulate_worker(Sandpile<T>& system, const py::object& shared_value,
                              uint32_t time_steps, int tqdm_update_steps)
{
    system.initialize_system(time_steps, std::nullopt, std::nullopt);
    const int tqdm_update_step = static_cast<int>(time_steps / tqdm_update_steps);
    const int start_value = shared_value.attr("value").cast<int>();

    for (uint32_t i = 1; i <= time_steps; ++i)
    {
        system.step(std::nullopt);
        if (i % static_cast<uint32_t>(tqdm_update_step) == 0)
        {
            auto _ = shared_value.attr("get_lock")().attr("acquire")();
            // auto _ = lock.attr("acquire")();
            // shared_value.attr("value") = shared_value.attr("value").cast<int>() + tqdm_update_step;
            shared_value.attr("value") = shared_value.attr("value").cast<int>() + 1;
            // _ = lock.attr("release")();
            _ = shared_value.attr("get_lock")().attr("release")();
        }
    }

    // auto _ = shared_value.attr("get_lock")().attr("acquire")();
    // shared_value.attr("value") =
    //     shared_value.attr("value").cast<int>() + static_cast<int>((time_steps % tqdm_update_step) == 0);
    // _ = shared_value.attr("get_lock")().attr("release")();

    system.shrink_to_fit();
}

template <typename T>
void bind_sandpile(py::module_& m, const std::string name)
{
    py::class_<Sandpile<T>>(m, name.c_str())
        .def(py::init<uint8_t, uint8_t, uint8_t, bool, bool>(),
             py::arg("dim"),
             py::arg("grid"),
             py::arg("crit_slope"),
             py::arg("has_open_boundary") = true,
             py::arg("has_conservative_perturbation") = true,
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
        .def_readwrite("time_cut_off", &Sandpile<T>::time_cut_off,
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

template <typename T>
void bind_simulate_worker(py::module_& m, std::string name)
{
    m.def(name.c_str(), &sandpile_simulate_worker<T>,
          py::arg("system"),
          py::arg("shared_value"),
          py::arg("time_steps"),
          py::arg("tqdm_update_steps") = TQDM_UPDATE_STEPS
    );
}

template <typename T>
void bind_simulate_single(py::module_& m, std::string name)
{
    m.def(name.c_str(), &simulate_single<T>,
          py::arg("system"),
          // py::arg("tqdm"),
          py::arg("time_steps"),
          py::arg("tqdm_update_steps") = TQDM_UPDATE_STEPS,
          py::arg("position") = 0
    );
}

PYBIND11_MODULE(sandpile, m)
{
    bind_sandpile<int8_t>(m, "Sandpile8Bit");
    bind_sandpile<int16_t>(m, "Sandpile16Bit");
    bind_simulate_worker<int8_t>(m, "sandpile_simulate_worker");
    bind_simulate_worker<int16_t>(m, "sandpile_simulate_worker");
    bind_simulate_single<int8_t>(m, "sandpile_simulate_single");
    bind_simulate_single<int16_t>(m, "sandpile_simulate_single");
}
