#pragma once
#include <sandpile.hpp>


template <typename T>
void Sandpile<T>::initialize_system(const uint32_t time_steps, std::optional<vector<T>> start_cfg,
                                    std::optional<int> seed)
{
    this->_average_slopes.clear();
    this->_size.clear();
    this->_time.clear();
    this->_reach.clear();
    this->dissipation_rate.clear();

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
        _relax_func = [this](vector<T>& cfg, const uint64_t raveled_index, set& crit_points)
        {
            return Sandpile::_op_bound_system_relax(cfg, raveled_index, crit_points);
        };
    }
    else
    {
        _relax_func = [this](vector<T>& cfg, const uint64_t raveled_index, set& crit_points)
        {
            return Sandpile::_cl_bound_system_relax(cfg, raveled_index, crit_points);
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
    this->_size.reserve(time_steps / 2);
    this->_time.reserve(time_steps / 2);
    this->_reach.reserve(time_steps / 2);
    this->dissipation_rate.reserve(time_steps / 2);
}


template <typename T>
void Sandpile<T>::_perturb_conservative(vector<T>& cfg, const vector<uint8_t>& position)
{
    auto r_index = ravel_index(position, this->grid);
    cfg(r_index) += static_cast<T>(this->dim);

    // auto pos_buf = position.mutable_unchecked<1>();
    for (ssize_t i = 0; i < position.size(); ++i)
    {
        if (position(i) == 0)
        {
            continue;
        }
        cfg(shift_ravelled_index(r_index, this->dim, this->grid, -1, static_cast<uint8_t>(i))) -= 1;
    }
}

template <typename T>
void Sandpile<T>::_perturb_non_conservative(vector<T>& cfg, const vector<uint8_t>& position)
{
    cfg(ravel_index(position, this->grid)) += static_cast<T>(this->dim);
}

// Deprecated
template <typename T>
void Sandpile<T>::_op_bound_system_relax(vector<T>& cfg, uint64_t raveled_index, set& crit_points)
{
    auto position_index = unravel_index(raveled_index, this->dim, this->grid);

    T boundary_indices = 0;
    for (ssize_t i = 0; i < this->dim; ++i)
    {
        if (position_index(i) == 0)
        {
            cfg(raveled_index) = 0;
            return;
        }
        if (position_index(i) == this->grid - 1)
        {
            ++boundary_indices;
        }
    }

    cfg(raveled_index) += static_cast<T>(-2 * this->dim + boundary_indices);

    for (ssize_t i = 0; i < this->dim; ++i)
    {
        auto x = shift_ravelled_index(raveled_index, this->dim, this->grid, -1, static_cast<uint8_t>(i));
        T& s1 = cfg(x);
        s1 += 1;
        if (s1 > this->crit_slope)
        {
            crit_points.insert(x);
        }

        if (position_index(i) < (grid - 1))
        {
            x = shift_ravelled_index(raveled_index, dim, grid, 1, static_cast<uint8_t>(i));
            T& s2 = cfg(x);
            s2 += 1;
            if (s2 > this->crit_slope)
            {
                crit_points.insert(x);
            }
        }
    }
}

// Deprecated
template <typename T>
void Sandpile<T>::_cl_bound_system_relax(vector<T>& cfg, const uint64_t raveled_index, set& crit_points)
{
    // const auto dim = static_cast<uint8_t>(position_index.size());
    auto position_index = unravel_index(raveled_index, this->dim, this->grid);

    T boundary_indices = 0;
    // auto ravelled_index = ravel_index(position_index, grid);
    for (ssize_t i = 0; i < this->dim; ++i)
    {
        if (position_index(i) == 0 || position_index(i) == (this->grid - 1))
        {
            cfg(raveled_index) = 0;
            return;
        }
    }

    cfg(raveled_index) += static_cast<T>(-2 * this->dim);
    for (ssize_t i = 0; i < this->dim; ++i)
    {
        auto x = shift_ravelled_index(raveled_index, this->dim, this->grid, -1, static_cast<uint8_t>(i));
        T& s1 = cfg(x);
        s1 += 1;
        if (s1 > this->crit_slope)
        {
            crit_points.insert(x);
        }

        x = shift_ravelled_index(raveled_index, this->dim, this->grid, 1, static_cast<uint8_t>(i));
        T& s2 = cfg(x);
        s2 += 1;
        if (s2 > this->crit_slope)
        {
            crit_points.insert(x);
        }
    }
}

template <typename T>
void Sandpile<T>::_relax_avalanche(vector<T>& start_cfg, vector<uint8_t>& start_point)
{
    constexpr int max_step = 5000;
    int i = 0;
    uint32_t size{};
    uint32_t time{};
    double reach{};
    std::vector<uint16_t> dissipation_rate;
    dissipation_rate.reserve(DISSIPATION_RESERVE);

    auto raveled_index = ravel_index(start_point, this->grid);
    set crit_points{raveled_index};
    set temp_crit_points{};
    for (i = 0; i < max_step; ++i)
    {
        // auto critical_points = get_critical_points(start_cfg, this->dim, this->grid, this->crit_slope);
        if (crit_points.empty())
        {
            break;
        }

        size += static_cast<uint32_t>(crit_points.size());
        time += 1;
        dissipation_rate.push_back(static_cast<uint16_t>(crit_points.size()));

        for (uint64_t crit_point : crit_points)
        {
            double temp = 0.;
            auto crit_point_multi_index = unravel_index(crit_point, this->dim, this->grid);
            for (ssize_t j = 0; j < crit_point_multi_index.size(); ++j)
            {
                temp += (pow(start_point(j) - crit_point_multi_index(j), 2));
            }
            temp = sqrt(temp);
            reach = std::max(reach, temp);

            this->_relax_func(start_cfg, crit_point, temp_crit_points);
        }
        std::swap(crit_points, temp_crit_points);
        temp_crit_points.clear();
    }

    if (i == (max_step - 1))
    {
        throw std::runtime_error("Max number of step iterations reached");
    }

    if (this->_average_slopes.size() > this->time_cut_off)
    {
        this->_size.push_back(size);
        this->_time.push_back(time);
        this->_reach.push_back(reach);
        vector<uint16_t> dissipation_rate_eigen(dissipation_rate.size());
        std::ranges::move(dissipation_rate.begin(), dissipation_rate.end(), dissipation_rate_eigen.data());
        this->dissipation_rate.push_back(dissipation_rate_eigen);
    }
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
        this->_relax_avalanche(this->current_cfg, perturb_position.value());
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
    this->_size.shrink_to_fit();
    this->_time.shrink_to_fit();
    this->_reach.shrink_to_fit();
    this->dissipation_rate.shrink_to_fit();
}

template <typename T>
vector<uint32_t> Sandpile<T>::generate_total_dissipation_rate(const uint32_t time_steps, std::optional<int> seed)
{
    if (!seed.has_value())
    {
        std::random_device rd{};
        seed = rd();
    }
    std::mt19937 gen(seed.value());
    std::uniform_int_distribution<uint32_t> start_dist(0, time_steps - 1);

    vector<uint32_t> total{vector<uint32_t>::Zero(time_steps)};
    for (auto& dis : this->dissipation_rate)
    {
        const auto start = start_dist(gen);
        for (ssize_t j = 0; j < dis.size(); ++j)
        {
            total((j + start) % time_steps) += static_cast<uint32_t>(dis(j));
        }
    }

    return total;
}

template <typename T>
void Sandpile<T>::shrink_to_fit()
{
    this->_average_slopes.shrink_to_fit();
    this->_size.shrink_to_fit();
    this->_time.shrink_to_fit();
    this->_reach.shrink_to_fit();
    this->dissipation_rate.shrink_to_fit();
}
