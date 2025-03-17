#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_chronometer.hpp>
#include <Eigen/Core>
#include <avalanche.hpp>
#include <vector>

// #include <sandpile.hpp>

#define N 100'000
TEST_CASE("benching")
{
    SECTION("move")
    {
        std::vector<int> v1(N);
        v1.assign(N, 20);
        REQUIRE(v1[0] == 20);

        Eigen::VectorX<int> v2(N);
        std::ranges::move(v1.begin(), v1.end(), v2.data());
        REQUIRE(v2(0) == 20);
    }
    BENCHMARK_ADVANCED("move")(Catch::Benchmark::Chronometer meter)
    {
        std::vector<int> v1(N);
        v1.assign(20, N);

        Eigen::VectorX<int> v2(N);
        meter.measure([&v1, &v2]()
        {
            std::ranges::move(v1.begin(), v1.end(), v2.data());
        });
    };

    BENCHMARK_ADVANCED("map")(Catch::Benchmark::Chronometer meter)
    {
        std::vector<int> v1(N);
        v1.assign(20, N);

        Eigen::VectorX<int> v2(N);
        meter.measure([&v1, &v2]()
        {
            vector<int> v3 = Eigen::Map<vector<int>>(&v1.front(), N);
        });
    };
#define N 10
    BENCHMARK("resizing"){
        Eigen::VectorXi v(50'000);
        int i = 0;
        for (i = 0; i < N; ++i)
        {
            v(i) = i;
        }
        v.conservativeResize(i);
    };

    BENCHMARK("push_back") {
        std::vector<int> v;
        v.reserve(50'000);
        for (int i = 0; i < N; ++i)
        {
            v.push_back(i);
        }
        Eigen::VectorXi v2(N);
        std::ranges::move(v.begin(), v.end(), v2.data());
        REQUIRE(v2(N-1) == (N-1));
    };
}
