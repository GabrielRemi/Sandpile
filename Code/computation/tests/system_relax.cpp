#define CATCH_CONFIG_MAIN
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <avalanche.hpp>
#include <random>


TEST_CASE("OP_BOUND_SYSTEM_RELAX")
{
    SECTION("1D")
    {
        vector<int8_t> cfg(5);
        cfg << 0, 0, 4, 0, 0;
        vector<uint8_t> index(1);
        index << 2;

        // op_bound_system_relax(cfg, index, 5);
        op_bound_system_relax(cfg, ravel_index(index, 5), 5, 1);

        vector<int8_t> result(5);
        result << 0, 1, 2, 1, 0;
        REQUIRE(result == cfg);

        cfg << 0, 0, 0, 1, 0;
        result << 0, 0, 0, 2, -1;
        index << 4;
        // op_bound_system_relax(cfg, index, 5);
        op_bound_system_relax(cfg, ravel_index(index, 5), 5, 1);
        REQUIRE(result == cfg);
    }

    SECTION("2D")
    {
        // MIDDLE
        vector<int8_t> cfg(25);
        cfg.setZero();
        vector<uint8_t> index(2);
        index << 2, 2;
        cfg(ravel_index(index, 5)) = 4;
        // op_bound_system_relax(cfg, index, 5);
        op_bound_system_relax(cfg, ravel_index(index, 5), 5, 2);

        vector<int8_t> result(25);
        result.setZero();

        index(0) += 1;
        result(ravel_index(index, 5)) += 1;
        index(0) -= 2;
        result(ravel_index(index, 5)) += 1;
        index(0) += 1;
        index(1) += 1;
        result(ravel_index(index, 5)) += 1;
        index(1) -= 2;
        result(ravel_index(index, 5)) += 1;
        index(1) += 1;
        REQUIRE(result == cfg);

        // boundary
        cfg.setZero();
        result.setZero();
        index << 4, 4;
        cfg(ravel_index(index, 5)) = 4;
        // op_bound_system_relax(cfg, index, 5);
        op_bound_system_relax(cfg, ravel_index(index, 5), 5, 2);
        result(ravel_index(index, 5)) = 2;
        index(0) -= 1;
        result(ravel_index(index, 5)) = 1;
        index(0) += 1;
        index(1) -= 1;
        result(ravel_index(index, 5)) = 1;
        index(1) += 1;

        REQUIRE(result == cfg);
    }
    SECTION("Benchmarks")
    {
        BENCHMARK_ADVANCED("DIM 2 GRID 40")(Catch::Benchmark::Chronometer meter)
        {
            std::mt19937 gen(0);
            vector<int8_t> cfg(40 * 40);
            cfg.setConstant(10);
            vector<uint8_t> index(2);
            index << 20, 20;
            meter.measure(
                [&cfg, &index]()
                {
                    // return op_bound_system_relax(cfg, index, 40);
                    return op_bound_system_relax(cfg, ravel_index(index, 40), 40, 2);
                }
            );
        };

        BENCHMARK_ADVANCED("DIM 6 GRID 20")(Catch::Benchmark::Chronometer meter)
        {
            std::mt19937 gen(0);
            vector<int8_t> cfg(static_cast<int>(pow(20, 6)));
            cfg.setConstant(10);
            vector<uint8_t> index(6);
            index << 10, 10, 10, 10, 10, 10;
            meter.measure(
                [&cfg, &index]()
                {
                    // return op_bound_system_relax(cfg, index, 20);
                    return op_bound_system_relax(cfg, ravel_index(index, 20), 20, 6);
                }
            );
        };
    }
}

TEST_CASE("CL_BOUND_SYSTEM_RELAX")
{
    SECTION("1D")
    {
        vector<int8_t> cfg(5);
        cfg << 0, 0, 4, 0, 0;
        vector<uint8_t> index(1);
        index << 2;

        // cl_bound_system_relax(cfg, index, 5);
        cl_bound_system_relax(cfg, ravel_index(index, 5), 5, 1);

        vector<int8_t> result(5);
        result << 0, 1, 2, 1, 0;
        REQUIRE(result == cfg);

        cfg << 0, 0, 0, 1, 1;
        result << 0, 0, 0, 1, 0;
        index << 4;
        // cl_bound_system_relax(cfg, index, 5);
        cl_bound_system_relax(cfg, ravel_index(index, 5), 5, 1);
        REQUIRE(result == cfg);
    }

    SECTION("2D")
    {
        // MIDDLE
        vector<int8_t> cfg(25);
        cfg.setZero();
        vector<uint8_t> index(2);
        index << 2, 2;
        cfg(ravel_index(index, 5)) = 4;
        cl_bound_system_relax(cfg, ravel_index(index, 5), 5, 2);

        vector<int8_t> result(25);
        result.setZero();

        index(0) += 1;
        result(ravel_index(index, 5)) += 1;
        index(0) -= 2;
        result(ravel_index(index, 5)) += 1;
        index(0) += 1;
        index(1) += 1;
        result(ravel_index(index, 5)) += 1;
        index(1) -= 2;
        result(ravel_index(index, 5)) += 1;
        index(1) += 1;
        REQUIRE(result == cfg);

        // boundary
        cfg.setZero();
        result.setZero();
        index << 4, 4;
        cfg(ravel_index(index, 5)) = 4;
        cl_bound_system_relax(cfg, ravel_index(index, 5), 5, 2);

        REQUIRE(result == cfg);
    }

    SECTION("Benchmarks")
    {
        BENCHMARK_ADVANCED("DIM 2 GRID 40")(Catch::Benchmark::Chronometer meter)
        {
            std::mt19937 gen(0);
            vector<int8_t> cfg(40 * 40);
            cfg.setConstant(10);
            vector<uint8_t> index(2);
            index << 20, 20;
            meter.measure(
                [&cfg, &index]()
                {
                    return cl_bound_system_relax(cfg, ravel_index(index, 40), 40, 2);
                }
            );
        };
        BENCHMARK_ADVANCED("DIM 6 GRID 20")(Catch::Benchmark::Chronometer meter)
        {
            std::mt19937 gen(0);
            vector<int8_t> cfg(static_cast<int>(pow(20, 6)));
            cfg.setConstant(10);
            vector<uint8_t> index(6);
            index << 10, 10, 10, 10, 10, 10;
            meter.measure(
                [&cfg, &index]()
                {
                    return cl_bound_system_relax(cfg, ravel_index(index, 20), 20, 6);
                }
            );
        };
    };
}
