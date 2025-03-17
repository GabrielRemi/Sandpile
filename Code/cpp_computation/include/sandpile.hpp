#pragma once
#include <functional>
#include <avalanche.hpp>
#include <random>

template <typename T>
struct Sandpile
{
private:
    std::function<void(vector<T>&, vector<uint8_t>&)> _perturb_func;
    std::function<void(vector<T>&, vector<uint8_t>&)> _relax_func;
    std::vector<AvalancheData> _avalanches;
    std::vector<double> _average_slopes;
    void _perturb_conservative(vector<T>& cfg, const vector<uint8_t>& position);
    void _perturb_non_conservative(vector<T>& cfg, const vector<uint8_t>& position);
    AvalancheData _relax_avalanche(uint32_t time_step, vector<T>& start_cfg, vector<uint8_t>& start_point);
    std::mt19937 _gen;
    std::uniform_int_distribution<uint8_t> _dist;
    bool _has_open_boundary;
    bool _has_conservative_perturbation;

public:
    uint8_t dim;
    uint8_t grid;
    uint8_t crit_slope;

    vector<double> get_average_slopes()
    {
        vector<double> average_slopes(this->_average_slopes.size());
        std::ranges::move(this->_average_slopes.begin(), this->_average_slopes.end(), average_slopes.data());

        return average_slopes;
    }

    bool get_has_open_boundary() const { return _has_open_boundary; }
    bool get_has_conservative_perturbation() const { return _has_conservative_perturbation; }
    vector<T> current_cfg;

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
};


#include "../sandpile.cpp"
