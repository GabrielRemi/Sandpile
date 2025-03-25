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
    BENCHMARK("random one")
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        auto shape = static_cast<uint64_t>(pow(15, 6));
        std::uniform_int_distribution<uint64_t> dist(0, shape - 1);
        std::vector<uint8_t> cfg(shape);
        auto index = dist(gen);
        return index;
    };
    BENCHMARK("random multi")
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint8_t> dist(0, 14);
        std::vector<uint8_t> cfg(static_cast<int>(pow(15, 6)));
        vector<uint8_t> index(6);
        for (auto& i : index)
        {
            i = dist(gen);
        }
        return cfg;
    };
    // BENCHMARK("DIM 1 GRID 40")
    // {
    //     auto system = Sandpile<uint8_t>(1, 40, 7, false, false);
    //     system.simulate(N, std::nullopt, 0);
    // };
    //
    // BENCHMARK("DIM 2 GRID 40")
    // {
    //     auto system = Sandpile<uint8_t>(2, 40, 7, false, false);
    //     system.simulate(N, std::nullopt, 0);
    // };
    //
    // BENCHMARK("DIM 3 GRID 30")
    // {
    //     auto system = Sandpile<uint8_t>(3, 30, 7, false, false);
    //     system.simulate(N, std::nullopt, 0);
    // };
    //
    // BENCHMARK("DIM 4 GRID 20")
    // {
    //     auto system = Sandpile<uint8_t>(4, 20, 7, false, false);
    //     system.simulate(N, std::nullopt, 0);
    // };
    //
    // BENCHMARK("DIM 5 GRID 10")
    // {
    //     auto system = Sandpile<uint8_t>(5, 10, 7, false, false);
    //     system.simulate(N, std::nullopt, 0);
    // };
}
