#define CATCH_CONFIG_MAIN
// #include <catch2/catch.hpp>
#include <Eigen/Dense>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cinttypes>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <print>
#include <random>

#include "../include/avalanche.hpp"

namespace fs = std::filesystem;

TEST_CASE("Benchmarks")
{
    BENCHMARK_ADVANCED("Ravel Dim 2 Grid 40")(Catch::Benchmark::Chronometer meter)
    {
        std::mt19937 gen(0);
        std::uniform_int_distribution<uint8_t> ind_dist(0, 39);
        auto multi_index = Eigen::VectorX<uint8_t>(2);
        multi_index << ind_dist(gen), ind_dist(gen);
        meter.measure([&multi_index]() { return ravel_index(multi_index, 40); });
    };

    BENCHMARK_ADVANCED("Ravel Dim 6 Grid 40")(Catch::Benchmark::Chronometer meter)
    {
        std::mt19937 gen(0);
        auto multi_index = Eigen::VectorX<uint8_t>(6);
        std::uniform_int_distribution<uint8_t> ind_dist(0, 39);
        multi_index << ind_dist(gen), ind_dist(gen), ind_dist(gen), ind_dist(gen), ind_dist(gen), ind_dist(gen);
        meter.measure([&multi_index]() { return ravel_index(multi_index, 40); });
    };

    BENCHMARK_ADVANCED("Unravel Dim 2 Grid 40")(Catch::Benchmark::Chronometer meter)
    {
        std::mt19937 gen(0);
        std::uniform_int_distribution<uint8_t> ind_dist(0, static_cast<uint64_t>(pow(40, 2)) - 1);
        auto index = ind_dist(gen);
        meter.measure([&index]() { return unravel_index(index, 2, 40); });
    };

    BENCHMARK_ADVANCED("Unravel Dim 6 Grid 40")(Catch::Benchmark::Chronometer meter)
    {
        std::mt19937 gen(0);
        std::uniform_int_distribution<uint8_t> ind_dist(0, static_cast<uint64_t>(pow(40, 6)) - 1);
        auto index = ind_dist(gen);
        meter.measure([&index]() { return unravel_index(index, 2, 40); });
    };
}

TEST_CASE("Ravel Index")
{
    SECTION("Dim 1 Grid 5")
    {
        uint8_t dim = 1;
        uint8_t grid = 5;
        auto multi_index = Eigen::VectorX<uint8_t>(dim);
        multi_index << 4;

        REQUIRE(ravel_index(multi_index, grid) == 4);
    };

    SECTION("Dim 2 Grid 5")
    {
        uint8_t dim = 2;
        uint8_t grid = 5;
        auto multi_index = Eigen::VectorX<uint8_t>(dim);
        multi_index << 1, 4;

        REQUIRE(ravel_index(multi_index, grid) == 9);
    };

    SECTION("Dim 3 Grid 5")
    {
        uint8_t dim = 3;
        uint8_t grid = 5;
        auto multi_index = Eigen::VectorX<uint8_t>(dim);
        multi_index << 1, 4, 3;

        REQUIRE(ravel_index(multi_index, grid) == 48);
    };

    SECTION("Dim 6 Grid 5")
    {
        uint8_t dim = 6;
        uint8_t grid = 5;
        auto multi_index = Eigen::VectorX<uint8_t>(dim);
        multi_index << 1, 4, 3, 2, 1, 4;

        REQUIRE(ravel_index(multi_index, grid) == 6059);
    };
}

TEST_CASE("Unravel Index")
{
    SECTION("Index 4 Dim 1 Grid 5")
    {
        uint64_t index = 4;
        uint8_t dim = 1;
        uint8_t grid = 5;

        auto result = unravel_index(index, dim, grid);
        REQUIRE(result.size() == dim);
        REQUIRE(result(0) == 4);
    };

    SECTION("Index 9 Dim 2 Grid 5")
    {
        uint64_t index = 9;
        uint8_t dim = 2;
        uint8_t grid = 5;

        auto result = unravel_index(index, dim, grid);
        REQUIRE(result.size() == dim);
        REQUIRE(result(0) == 1);
        REQUIRE(result(1) == 4);
    };
    SECTION("Index 48 Dim 3 Grid 5")
    {
        uint64_t index = 48;
        uint8_t dim = 3;
        uint8_t grid = 5;

        auto result = unravel_index(index, dim, grid);
        REQUIRE(result.size() == dim);
        REQUIRE(result(0) == 1);
        REQUIRE(result(1) == 4);
        REQUIRE(result(2) == 3);
    };

    SECTION("Index 6059 Dim 6 Grid 5")
    {
        uint64_t index = 6059;
        uint8_t dim = 6;
        uint8_t grid = 5;

        auto result = unravel_index(index, dim, grid);
        REQUIRE(result.size() == dim);
        REQUIRE(result(0) == 1);
        REQUIRE(result(1) == 4);
        REQUIRE(result(2) == 3);
        REQUIRE(result(3) == 2);
        REQUIRE(result(4) == 1);
        REQUIRE(result(5) == 4);
    };
}

void func(const uint8_t dim)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> grid_dist(2, 40);

    for (int i = 0; i < 20; ++i)
    {
        // auto index = Eigen::VectorX<uint8_t>(dim);
        const auto grid = grid_dist(gen);
        std::uniform_int_distribution<uint8_t> ind_dist(0, grid - 1);

        auto index = vector<uint8_t>(dim);
        for (int j = 0; j < dim; ++j)
        {
            index(j) = ind_dist(gen);
        }

        auto result = unravel_index(ravel_index(index, grid), dim, grid);
        REQUIRE(result == index);
    }
}

TEST_CASE("Round-Trip consistency")
{
    SECTION("1D") { func(1); }
    SECTION("2D") { func(2); }
    SECTION("3D") { func(3); }
    SECTION("4D") { func(4); }
    SECTION("5D") { func(5); }
    SECTION("6D") { func(6); }
}

TEST_CASE("Shift Raveled Index")
{
    // std::uniform_int_distribution<uint8_t> grid_dist(2, 40);
    std::random_device rd;
    auto seed = rd();
    // seed = 1131213074;
    std::mt19937 gen(seed);
    constexpr uint8_t grid = 10;
    for (uint8_t dim = 1; dim < 8; ++dim)
    {
        std::uniform_int_distribution<uint8_t> ind_dist(1, grid - 2);
        auto index = vector<uint8_t>(dim);
        for (ssize_t i = 0; i < dim; ++i)
        {
            index(i) = ind_dist(gen);
        }

        for (ssize_t i = 0; i < dim; ++i)
        {
            auto new_index = index;
            new_index(i) += 1;

            const auto raveled_index = ravel_index(index, grid);
            const auto out1 = ravel_index(new_index, grid);
            const auto out2 = shift_ravelled_index(raveled_index, dim, grid, 1, static_cast<uint8_t>(i));
            auto b = out1 == out2;
            if (!b)
            {
                auto log_path{fs::path(__FILE__).parent_path() / "shift_ravel.log"};
                auto file = std::ofstream(log_path.string());
                file << std::format("seed: {}, dim {}, grid {}, shift_dim {}", seed, dim, grid, i) << std::endl;
                file << index << std::endl << std::endl;
                file << out1 << std::endl;
                file << out2 << std::endl;
            }
            REQUIRE(out1 == out2);
        }
    }
}
