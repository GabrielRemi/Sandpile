#define CATCH_CONFIG_MAIN
// #include <catch2/catch.hpp>
#include <Eigen/Dense>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cinttypes>
#include <iostream>
#include <print>
#include <random>

#include "../include/avalanche.hpp"

TEST_CASE("Ravel Index") {
    SECTION("Dim 1 Grid 5") {
        uint8_t dim = 1;
        uint8_t grid = 5;
        auto multi_index = Eigen::VectorX<uint8_t>(dim);
        multi_index << 4;

        REQUIRE(ravel_index(multi_index, grid) == 4);
    };

    SECTION("Dim 2 Grid 5") {
        uint8_t dim = 2;
        uint8_t grid = 5;
        auto multi_index = Eigen::VectorX<uint8_t>(dim);
        multi_index << 1, 4;

        REQUIRE(ravel_index(multi_index, grid) == 9);
    };

    SECTION("Dim 3 Grid 5") {
        uint8_t dim = 3;
        uint8_t grid = 5;
        auto multi_index = Eigen::VectorX<uint8_t>(dim);
        multi_index << 1, 4, 3;

        REQUIRE(ravel_index(multi_index, grid) == 48);
    };

    SECTION("Dim 6 Grid 5") {
        uint8_t dim = 6;
        uint8_t grid = 5;
        auto multi_index = Eigen::VectorX<uint8_t>(dim);
        multi_index << 1, 4, 3, 2, 1, 4;

        REQUIRE(ravel_index(multi_index, grid) == 6059);
    };
}

TEST_CASE("Unravel Index") {
    SECTION("Index 4 Dim 1 Grid 5") {
        uint64_t index = 4;
        uint8_t dim = 1;
        uint8_t grid = 5;

        auto result = unravel_index(index, dim, grid);
        REQUIRE(result.size() == dim);
        REQUIRE(result(0) == 4);
    };

    SECTION("Index 9 Dim 2 Grid 5") {
        uint64_t index = 9;
        uint8_t dim = 2;
        uint8_t grid = 5;

        auto result = unravel_index(index, dim, grid);
        REQUIRE(result.size() == dim);
        REQUIRE(result(0) == 1);
        REQUIRE(result(1) == 4);
    };

    SECTION("Index 48 Dim 3 Grid 5") {
        uint64_t index = 48;
        uint8_t dim = 3;
        uint8_t grid = 5;

        auto result = unravel_index(index, dim, grid);
        REQUIRE(result.size() == dim);
        REQUIRE(result(0) == 1);
        REQUIRE(result(1) == 4);
        REQUIRE(result(2) == 3);
    };

    SECTION("Index 6059 Dim 6 Grid 5") {
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

void func(const uint8_t dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> grid_dist(2, 40);

    for (int i = 0; i < 20; ++i) {
        // auto index = Eigen::VectorX<uint8_t>(dim);
        const auto grid = grid_dist(gen);
        std::uniform_int_distribution<uint8_t> ind_dist(0, grid - 1);

        auto index = vector<uint8_t>(dim);
        for (int j = 0; j < dim; ++j) {
            index(j) = ind_dist(gen);
        }

        auto result = unravel_index(ravel_index(index, grid), dim, grid);
        REQUIRE(result == index);
    }
}

TEST_CASE("Round-Trip consistency") {
    SECTION("1D") { func(1); }
    SECTION("2D") { func(2); }
    SECTION("3D") { func(3); }
    SECTION("4D") { func(4); }
    SECTION("5D") { func(5); }
    SECTION("6D") { func(6); }
}

// TEST_CASE("Shift Raveled Index") {
//     // std::uniform_int_distribution<uint8_t> grid_dist(2, 40);
//     constexpr uint8_t grid = 5, dim = 2;
//     std::uniform_int_distribution<uint8_t> ind_dist(0, grid - 1);
//     auto index = vector<uint8_t>(dim);
//     index << 1, 3;
//     REQUIRE(index.size() == dim);
//     REQUIRE(index(0) == 1);
//     REQUIRE(index(1) == 3);
//
//     auto new_index = index;
//     new_index(0) += 1;
//
//     const auto raveled_index = ravel_index(index, dim);
//     REQUIRE(raveled_index == 8);
//     const auto out1 = ravel_index(new_index, dim);
//     const auto out2 = shift_ravelled_index(raveled_index, dim, grid, 1, 0);
//     REQUIRE(out1 == out2);
//
// }