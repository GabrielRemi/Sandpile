#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_chronometer.hpp>
#include <Eigen/Core>
#include <avalanche.hpp>
#include <sandpile.hpp>
#include <vector>

// #include <sandpile.hpp>

#define N 10'000
TEST_CASE("benching")
{
    BENCHMARK("DIM 1 GRID 40")
    {
        auto system = Sandpile<uint8_t>(1, 40, 7, false, false);
        system.simulate(N, std::nullopt, 0);
    };

    BENCHMARK("DIM 2 GRID 40")
    {
        auto system = Sandpile<uint8_t>(2, 40, 7, false, false);
        system.simulate(N, std::nullopt, 0);
    };

    BENCHMARK("DIM 3 GRID 30")
    {
        auto system = Sandpile<uint8_t>(3, 30, 7, false, false);
        system.simulate(N, std::nullopt, 0);
    };

    BENCHMARK("DIM 4 GRID 20")
    {
        auto system = Sandpile<uint8_t>(4, 20, 7, false, false);
        system.simulate(N, std::nullopt, 0);
    };

    BENCHMARK("DIM 5 GRID 10")
    {
        auto system = Sandpile<uint8_t>(5, 10, 7, false, false);
        system.simulate(N, std::nullopt, 0);
    };
}
