#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <print>
#include <random>
#include <string>

namespace py = pybind11;

struct AvalancheData {
    uint32_t time_step;
    uint32_t size;
    uint32_t time;
    double reach;
    py::array_t<uint16_t> dissipation_rate;

    AvalancheData(uint32_t _t) : time_step(_t) {
        size = 0;
        time = 0;
        reach = 0;
        dissipation_rate = py::array_t<uint8_t>(0);
    }
};
template <typename T> struct Sandpile {
  private:
    std::function<void(py::array_t<T>, py::array_t<uint8_t>)> _perturb_func;
    std::vector<AvalancheData> _avalanches;
    void _perturb_conservative(py::array_t<T> cfg, py::array_t<uint8_t> position);
    void _perturb_non_conservative(py::array_t<T> cfg, py::array_t<uint8_t> position);
    void _step(std::optional<py::array_t<uint8_t>> perturb_position);
    std::mt19937 _gen;
    std::uniform_int_distribution<uint8_t> _dist;

  public:
    uint8_t dim;
    uint8_t grid;
    uint8_t crit_slope;
    bool open_boundary = true;
    bool conservative_perturbation = true;
    py::array_t<T> current_cfg;
    std::vector<double> average_slopes;

    Sandpile(uint8_t _d, uint8_t _g, uint8_t _c) : dim(_d), grid(_g), crit_slope(_c) {}
    Sandpile(uint8_t _d, uint8_t _g, uint8_t _c, bool _b, bool _p)
        : dim(_d), grid(_g), crit_slope(_c), open_boundary(_b), conservative_perturbation(_p) {}

    uint32_t size() { return static_cast<uint32_t>(pow(grid, dim)); }
    void initialize_system(uint32_t time_steps, std::optional<py::array_t<T>> start_cfg);
    void simulate(uint32_t time_steps, std::optional<py::array_t<T>> start_cfg);
};

template <typename T> std::string format_array(py::array_t<T> &arr) {
    auto buf = arr.template unchecked<1>();
    std::string result = "";
    for (ssize_t i = 0; i < (buf.shape(0) - 1); ++i) {
        result += std::format("{} ", buf(i));
    }
    result += std::format("{}", buf(buf.shape(0) - 1));

    return result;
}

uint64_t ravel_index(py::array_t<uint8_t> &multi_index, uint8_t grid) {
    uint64_t result = 0, curr_pow = 0;

    auto buf = multi_index.unchecked<1>();
    for (ssize_t i = buf.shape(0) - 1; i >= 0; --i) {
        result +=
            static_cast<uint64_t>(buf(i)) * static_cast<uint64_t>(std::pow(static_cast<uint64_t>(grid), curr_pow));
        curr_pow += 1;
    }

    return result;
}

py::array_t<uint8_t> unravel_index(uint64_t index, uint8_t dim, uint8_t grid) {
    py::array_t<uint8_t> result(dim);
    auto buf = result.mutable_unchecked<1>();

    for (ssize_t i = dim - 1; i >= 0; --i) {
        buf(i) = static_cast<uint8_t>(index % grid);
        index = static_cast<uint64_t>(floor(index / grid));
    }

    return result;
}

uint64_t shift_ravelled_index(uint64_t index, uint8_t dim, uint8_t grid, int16_t shift, uint8_t shift_dim) {
    return static_cast<uint64_t>(static_cast<int16_t>(index) +
                                 shift * static_cast<int16_t>(pow(grid, dim - shift_dim - 1)));
}

template <typename T> std::vector<py::array_t<uint8_t>> get_critical_points(py::array_t<T> &cfg, Sandpile<T> &system) {
    std::vector<py::array_t<uint8_t>> points(0);
    auto buf = cfg.template unchecked<1>();

    for (ssize_t i = 0; i < buf.shape(0); ++i) {
        auto slope = buf(i);
        if (static_cast<int16_t>(slope) > static_cast<int16_t>(system.crit_slope)) {
            points.push_back(unravel_index(static_cast<uint64_t>(i), system.dim, system.grid));
        }
    }

    return points;
    // TODO do not make a copy of the vector
}

template <typename T>
void op_bound_system_relax(py::array_t<T> &cfg, py::array_t<uint8_t> &position_index, uint8_t grid) {
    auto cfg_buf = cfg.template mutable_unchecked<1>();

    auto pos_buf = position_index.mutable_unchecked<1>();
    T dim = static_cast<T>(pos_buf.shape(0));

    T boundary_indices = 0;
    for (ssize_t i = 0; i < dim; ++i) {
        if (pos_buf(i) == 0) {
            cfg_buf(ravel_index(position_index, grid)) = 0;
            return;
        } else if (pos_buf(i) == grid - 1) {
            ++boundary_indices;
        }
    }

    cfg_buf(ravel_index(position_index, grid)) += static_cast<T>(-2 * dim + boundary_indices);

    for (ssize_t i = 0; i < dim; ++i) {
        pos_buf(i) -= 1;
        cfg_buf(ravel_index(position_index, grid)) += 1;

        pos_buf(i) += 1;
        if (pos_buf(i) < (grid - 1)) {
            pos_buf(i) += 1;
            cfg_buf(ravel_index(position_index, grid)) += 1;
            pos_buf(i) -= 1;
        }
    }
}

template <typename T>
void cl_bound_system_relax(py::array_t<T> &cfg, py::array_t<uint8_t> &position_index, uint8_t grid) {

    auto cfg_buf = cfg.template mutable_unchecked<1>();

    auto pos_buf = position_index.mutable_unchecked<1>();
    T dim = static_cast<T>(pos_buf.shape(0));

    T boundary_indices = 0;
    for (ssize_t i = 0; i < dim; ++i) {
        if (pos_buf(i) == 0 || pos_buf(i) == grid - 1) {
            cfg_buf(ravel_index(position_index, grid)) = 0;
            return;
        }
    }

    cfg_buf(ravel_index(position_index, grid)) += static_cast<T>(-2 * dim);
    for (ssize_t i = 0; i < dim; ++i) {
        pos_buf(i) -= 1;
        cfg_buf(ravel_index(position_index, grid)) += 1;

        pos_buf(i) += 2;
        cfg_buf(ravel_index(position_index, grid)) += 1;

        pos_buf(i) -= 1;
    }
}

template <typename T>
AvalancheData relax_avalanche(uint32_t time_step, py::array_t<T> &start_cfg, py::array_t<uint8_t> &start_point,
                              Sandpile<T> &system) {
    std::vector<uint16_t> dissipation_rate(0);
    // dissipation_rate.reserve(500);
    AvalancheData avalanche(time_step);

    int max_step = 5000;
    int i = 0;
    std::function<void(py::array_t<T> &, py::array_t<uint8_t> &, uint8_t)> relax;

    if (system.open_boundary) {
        relax = op_bound_system_relax<T>;
    } else {
        relax = cl_bound_system_relax<T>;
    }

    for (i = 0; i < max_step; ++i) {
        auto critical_points = get_critical_points(start_cfg, system);
        if (critical_points.size() == 0) {
            break;
        }

        avalanche.size += static_cast<uint32_t>(critical_points.size());
        avalanche.time += 1;
        dissipation_rate.push_back(static_cast<uint16_t>(critical_points.size()));

        for (auto &critical_point : critical_points) {

            auto buf = critical_point.template unchecked<1>();
            auto start_buf = start_point.unchecked<1>();
            double temp = 0.;
            for (ssize_t j = 0; j < buf.shape(0); ++j) {
                temp += static_cast<double>(pow(start_buf(j) - buf(j), 2));
            }
            temp = static_cast<double>(sqrt(temp));
            avalanche.reach = std::max(avalanche.reach, temp);

            relax(start_cfg, critical_point, system.grid);
        }
    }

    if (i == (max_step - 1)) {
        throw std::runtime_error("Max number of step iterations reached");
    }

    avalanche.dissipation_rate = py::array_t<uint16_t>(dissipation_rate.size(), dissipation_rate.data());
    return avalanche;
}

template <typename T>
void Sandpile<T>::initialize_system(uint32_t time_steps, std::optional<py::array_t<T>> start_cfg) {
    this->average_slopes.clear();
    this->_avalanches.clear();

    // specific seed for testing
    std::random_device rd;
    this->_gen = std::mt19937(rd());
    this->_dist = std::uniform_int_distribution<uint8_t>(0, this->grid - 1);

    if (conservative_perturbation) {
        _perturb_func = [this](py::array_t<T> cfg, py::array_t<uint8_t> position) {
            return this->_perturb_conservative(cfg, position);
        };
    } else {
        _perturb_func = [this](py::array_t<T> cfg, py::array_t<uint8_t> position) {
            return this->_perturb_non_conservative(cfg, position);
        };
    }
    if (!start_cfg) {
        this->current_cfg = py::array_t<T>(this->size());
        auto buf = this->current_cfg.template mutable_unchecked<1>();
        for (ssize_t i = 0; i < buf.shape(0); ++i) {
            buf(i) = 0;
        }
    } else {
        this->current_cfg = start_cfg.value();
    }

    this->average_slopes.reserve(time_steps);
    double average_slope = 0.;
    auto buf = this->current_cfg.template unchecked<1>();
    for (ssize_t i = 0; i < buf.shape(0); ++i) {
        average_slope += static_cast<double>(buf(i));
    }
    average_slope /= static_cast<double>(buf.shape(0));
    this->average_slopes.push_back(average_slope);
    this->_avalanches.reserve(time_steps / 2);
}
template <typename T> void Sandpile<T>::_perturb_conservative(py::array_t<T> cfg, py::array_t<uint8_t> position) {
    auto cfg_buf = cfg.template mutable_unchecked<1>();

    cfg_buf(ravel_index(position, this->grid)) += static_cast<T>(this->dim);

    auto pos_buf = position.mutable_unchecked<1>();
    for (ssize_t i = 0; i < pos_buf.shape(0); ++i) {
        if (pos_buf(i) == 0) {
            continue;
        }
        pos_buf(i) -= 1;
        cfg_buf(ravel_index(position, this->grid)) -= 1;
        pos_buf(i) += 1;
    }
}
template <typename T> void Sandpile<T>::_perturb_non_conservative(py::array_t<T> cfg, py::array_t<uint8_t> position) {
    auto cfg_buf = cfg.template mutable_unchecked<1>();

    cfg_buf(ravel_index(position, this->grid)) += static_cast<T>(this->dim);
}
template <typename T> void Sandpile<T>::_step(std::optional<py::array_t<uint8_t>> perturb_position) {
    // Generate randum perturbation position
    if (!perturb_position) {
        perturb_position = py::array_t<uint8_t>(this->dim);
        auto buf = perturb_position.value().mutable_unchecked<1>();
        for (ssize_t i = 0; i < buf.shape(0); ++i) {
            buf(i) = this->_dist(this->_gen);
        }
    }

    // Perturb the system and calculate the average slope
    _perturb_func(this->current_cfg, perturb_position.value());
    double average_slope = 0.;
    auto buf = this->current_cfg.template unchecked<1>();
    for (ssize_t i = 0; i < buf.shape(0); ++i) {
        average_slope += static_cast<double>(buf(i));
    }
    average_slope /= static_cast<double>(buf.shape(0));
    this->average_slopes.push_back(average_slope);

    // Relax the system
    auto cfg_buf = this->current_cfg.template unchecked<1>();
    if (cfg_buf(ravel_index(perturb_position.value(), this->grid)) > this->crit_slope) {
        auto avalanche = relax_avalanche(static_cast<uint32_t>(this->average_slopes.size() - 1), this->current_cfg,
                                         perturb_position.value(), *this);

        this->_avalanches.push_back(avalanche);
    }
}
template <typename T> void Sandpile<T>::simulate(uint32_t time_steps, std::optional<py::array_t<T>> start_cfg) {
    this->initialize_system(time_steps, start_cfg);
    // auto file = std::ofstream("data.log");
    for (uint32_t i = 0; i < time_steps; ++i) {
        this->_step(start_cfg);
        // file << format_array(this->current_cfg) << std::endl;
    }

    this->average_slopes.shrink_to_fit();
    this->_avalanches.shrink_to_fit();
}

template <typename T> void bind_sandpile(py::module_ &m) {
    py::class_<Sandpile<T>>(m, "Sandpile")
        .def(py::init<T, T, T>())
        .def(py::init<T, T, T, bool, bool>())
        .def_readwrite("dim", &Sandpile<T>::dim)
        .def_readwrite("grid", &Sandpile<T>::grid)
        .def_readwrite("crit_slope", &Sandpile<T>::crit_slope)
        .def_readwrite("closed_boundary", &Sandpile<T>::open_boundary)
        .def_readwrite("conservative_perturbation", &Sandpile<T>::conservative_perturbation)
        .def_readwrite("average_slopes", &Sandpile<T>::average_slopes)
        .def("simulate", &Sandpile<T>::simulate);
}

PYBIND11_MODULE(sandpile, m) {
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

    // py::class_<Sandpile<uint8_t>>(m, "Sandpile")
    //     .def(py::init<uint8_t, uint8_t, uint8_t>())
    //     .def(py::init<uint8_t, uint8_t, uint8_t, bool, bool>())
    //     .def_readwrite("dim", &Sandpile<uint8_t>::dim)
    //     .def_readwrite("grid", &Sandpile<uint8_t>::grid)
    //     .def_readwrite("crit_slope", &Sandpile<uint8_t>::crit_slope)
    //     .def_readwrite("closed_boundary", &Sandpile<uint8_t>::open_boundary)
    //     .def("simulate", &Sandpile<uint8_t>::simulate);
    bind_sandpile<int8_t>(m);
    // bind_sandpile<uint16_t>(m);

    // Only for testing purposes. It is not going to be used on the python side
    py::class_<AvalancheData>(m, "AvalancheData")
        .def(py::init<uint32_t>())
        .def_readwrite("time_step", &AvalancheData::time_step)
        .def_readwrite("size", &AvalancheData::size)
        .def_readwrite("time", &AvalancheData::time)
        .def_readwrite("reach", &AvalancheData::reach)
        .def_readwrite("dissipation_rate", &AvalancheData::dissipation_rate);
}