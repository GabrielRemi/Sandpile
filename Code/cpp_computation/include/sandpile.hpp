#pragma once
#include <functional>
#include <avalanche.hpp>
#include <random>


#define DISSIPATION_RESERVE 10'000

template <typename T>
inline vector<T> from_std_vector(std::vector<T>& v)
{
    vector<T> w(v.size());
    std::ranges::move(v.begin(), v.end(), w.data());

    return w;
}

template <typename T>
struct Sandpile
{
private:
    std::function<void(vector<T>&, vector<uint8_t>&)> _perturb_func{};
    std::function<void(vector<T>&, vector<uint8_t>&)> _relax_func{};
    std::vector<double> _average_slopes{};
    void _perturb_conservative(vector<T>& cfg, const vector<uint8_t>& position);
    void _perturb_non_conservative(vector<T>& cfg, const vector<uint8_t>& position);
    void _relax_avalanche(vector<T>& start_cfg, vector<uint8_t>& start_point);

    // Data of the avalanches
    std::vector<uint32_t> _size {};
    std::vector<uint32_t> _time {};
    std::vector<double> _reach {};

    std::mt19937 _gen{};
    std::uniform_int_distribution<uint8_t> _dist{};
    bool _has_open_boundary{};
    bool _has_conservative_perturbation{};

public:
    uint8_t dim;
    uint8_t grid;
    uint8_t crit_slope;

    uint32_t time_cut_off = 0; // All avalanche data below this time step will be ignored

    std::vector<vector<uint16_t>> dissipation_rate{};

    vector<uint32_t> get_size() { return from_std_vector<uint32_t>(this->_size); }
    vector<uint32_t> get_time() { return from_std_vector<uint32_t>(this->_time); }
    vector<double> get_reach() { return from_std_vector<double>(this->_reach); }
    vector<double> get_average_slopes() { return from_std_vector(this->_average_slopes); }

    bool get_has_open_boundary() const { return _has_open_boundary; }
    bool get_has_conservative_perturbation() const { return _has_conservative_perturbation; }
    vector<T> current_cfg{};

    Sandpile(const uint8_t _d, const uint8_t _g, const uint8_t _c) : dim(_d), grid(_g), crit_slope(_c)
    {
        _has_open_boundary = true;
        _has_conservative_perturbation = true;
    }

    Sandpile(const uint8_t _d, const uint8_t _g, const uint8_t _c, const bool _b, const bool _p)
        : _has_open_boundary(_b), _has_conservative_perturbation(_p), dim(_d), grid(_g), crit_slope(_c)
    {
    }

    uint32_t size() const { return static_cast<uint32_t>(pow(grid, dim)); }
    void initialize_system(uint32_t time_steps, std::optional<vector<T>> start_cfg, std::optional<int> seed);
    void step(std::optional<vector<uint8_t>> perturb_position);
    void simulate(uint32_t time_steps, std::optional<vector<T>> start_cfg, std::optional<int> seed);

    vector<uint32_t> generate_total_dissipation_rate(uint32_t time_steps, std::optional<int> seed);
};


#include "../sandpile.cpp"
