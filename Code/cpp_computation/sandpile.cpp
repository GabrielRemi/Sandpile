#pragma once
#include <sandpile.hpp>


template <typename T>
void Sandpile<T>::initialize_system(const uint32_t time_steps, std::optional<vector<T>> start_cfg,
                                    std::optional<int> seed)
{
    this->_average_slopes.clear();
    this->_avalanches.clear();

    // specific seed for testing
    if (!seed.has_value())
    {
        std::random_device rd;
        seed = rd();
    }
    this->_gen = std::mt19937(seed.value());
    this->_dist = std::uniform_int_distribution<uint8_t>(0, this->grid - 1);

    if (this->_has_conservative_perturbation)
    {
        _perturb_func = [this](vector<T>& cfg, const vector<uint8_t>& position)
        {
            return this->_perturb_conservative(cfg, position);
        };
    }
    else
    {
        _perturb_func = [this](vector<T>& cfg, const vector<uint8_t>& position)
        {
            return this->_perturb_non_conservative(cfg, position);
        };
    }
    if (this->_has_open_boundary)
    {
        _relax_func = [this](vector<T>& cfg, vector<uint8_t>& position)
        {
            return op_bound_system_relax(cfg, position, this->grid);
        };
    }
    else
    {
        _relax_func = [this](vector<T>& cfg, vector<uint8_t>& position)
        {
            return cl_bound_system_relax(cfg, position, this->grid);
        };
    }

    if (!start_cfg)
    {
        this->current_cfg = vector<T>(this->size());
        current_cfg.setZero();
    }
    else
    {
        this->current_cfg = start_cfg.value();
    }

    this->_average_slopes.reserve(time_steps);
    double average_slope = 0.;
    for (ssize_t i = 0; i < this->current_cfg.size(); ++i)
    {
        average_slope += static_cast<double>(this->current_cfg(i));
    }
    average_slope /= static_cast<double>(this->current_cfg.size());
    this->_average_slopes.push_back(average_slope);
    this->_avalanches.reserve(time_steps / 2);
}


template <typename T>
void Sandpile<T>::_perturb_conservative(vector<T>& cfg, const vector<uint8_t>& position)
{
    // auto cfg_buf = cfg.template mutable_unchecked<1>();

    auto r_index = ravel_index(position, this->grid);
    cfg(r_index) += static_cast<T>(this->dim);

    // auto pos_buf = position.mutable_unchecked<1>();
    for (ssize_t i = 0; i < position.size(); ++i)
    {
        if (position(i) == 0)
        {
            continue;
        }
        // position(i) -= 1;
        cfg(shift_ravelled_index(r_index, this->dim, this->grid, -1, static_cast<uint8_t>(i))) -= 1;
        // position(i) += 1;
    }
}

template <typename T>
void Sandpile<T>::_perturb_non_conservative(vector<T>& cfg, const vector<uint8_t>& position)
{
    // auto cfg_buf = cfg.template mutable_unchecked<1>();

    cfg(ravel_index(position, this->grid)) += static_cast<T>(this->dim);
}

// TODO
template <typename T>
AvalancheData Sandpile<T>::_relax_avalanche(const uint32_t time_step, vector<T>& start_cfg,
                                            vector<uint8_t>& start_point
)
{
    // std::vector<uint16_t> dissipation_rate(0);
    // dissipation_rate.reserve(500);
    AvalancheData avalanche;

    constexpr int max_step = 5000;
    int i = 0;

    for (i = 0; i < max_step; ++i)
    {
        auto critical_points = get_critical_points(start_cfg, this->dim, this->grid, this->crit_slope);
        if (critical_points.size() == 0)
        {
            break;
        }

        avalanche.size += static_cast<uint32_t>(critical_points.size());
        avalanche.time += 1;
        avalanche.dissipation_rate.push_back(static_cast<uint16_t>(critical_points.size()));

        for (auto& critical_point : critical_points)
        {
            // auto buf = critical_point.template unchecked<1>();
            // auto start_buf = start_point.unchecked<1>();
            double temp = 0.;
            for (ssize_t j = 0; j < critical_point.size(); ++j)
            {
                temp += static_cast<double>(pow(start_point(j) - critical_point(j), 2));
            }
            temp = sqrt(temp);
            avalanche.reach = std::max(avalanche.reach, temp);

            this->_relax_func(start_cfg, critical_point);
        }
    }

    if (i == (max_step - 1))
    {
        throw std::runtime_error("Max number of step iterations reached");
    }

    // avalanche.dissipation_rate = py::array_t<uint16_t>(dissipation_rate.size(), dissipation_rate.data());
    return avalanche;
}

template <typename T>
void Sandpile<T>::step(std::optional<vector<uint8_t>> perturb_position)
{
    // Generate random perturbation position
    if (!perturb_position)
    {
        perturb_position = vector<uint8_t>(this->dim);
        // auto buf = perturb_position.value().mutable_unchecked<1>();
        for (ssize_t i = 0; i < dim; ++i)
        {
            perturb_position.value()(i) = this->_dist(this->_gen);
        }
    }

    // Perturb the system and calculate the average slope
    _perturb_func(this->current_cfg, perturb_position.value());
    double average_slope = 0.;
    for (ssize_t i = 0; i < this->current_cfg.size(); ++i)
    {
        average_slope += static_cast<double>(this->current_cfg(i));
    }
    average_slope /= static_cast<double>(this->current_cfg.size());
    this->_average_slopes.push_back(average_slope);

    // Relax the system
    if (this->current_cfg(ravel_index(perturb_position.value(), this->grid)) > this->crit_slope)
    {
        auto avalanche = this->_relax_avalanche(static_cast<uint32_t>(this->_average_slopes.size() - 1),
                                                this->current_cfg,
                                                perturb_position.value());

        this->_avalanches.push_back(avalanche);
    }
}

template <typename T>
void Sandpile<T>::simulate(const uint32_t time_steps, std::optional<vector<T>> start_cfg,
                           const std::optional<int> seed)
{
    this->initialize_system(time_steps, start_cfg, seed);
    for (uint32_t i = 0; i < time_steps; ++i)
    {
        this->step(std::nullopt);
    }

    this->_average_slopes.shrink_to_fit();
    this->_avalanches.shrink_to_fit();
}
