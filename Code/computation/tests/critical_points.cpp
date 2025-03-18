#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <avalanche.hpp>
#include <random>
#include <fstream>
#include <filesystem>
#include <unordered_set>

template <typename T>
std::vector<T> remove_duplicates(const std::vector<T>& vec)
{
    std::unordered_set<T> seen;
    std::vector<T> result;

    for (int num : vec)
    {
        if (seen.insert(num).second)
        {
            // ✅ Inserts only if not already in set
            result.push_back(num);
        }
    }
    return result;
}

void ndim_test(const uint8_t dim, const uint8_t grid)
{
    std::random_device rd;
    auto seed = rd();
    std::mt19937 gen(seed);
    constexpr uint8_t crit_slope = 7;
    std::uniform_int_distribution<uint8_t> index_dist(0, grid - 1);
    std::uniform_int_distribution<int8_t> slope_dist(-crit_slope, crit_slope);
    const auto shape = static_cast<unsigned int>(pow(grid, dim));
    vector<int8_t> cfg(shape);

    for (int _i = 0; _i < 10; ++_i)
    {
        for (ssize_t i = 0; i < shape; i++)
        {
            cfg[i] = slope_dist(gen);
        }

        int crit_count = 5;
        std::vector<vector<uint8_t>> indices(crit_count);
        for (uint8_t i = 0; i < crit_count; ++i)
        {
            vector<uint8_t> index(dim);
            for (uint8_t j = 0; j < dim; ++j)
            {
                index[j] = index_dist(gen);
            }
            cfg[static_cast<Eigen::Index>(ravel_index(index, grid))] = crit_slope + 1;
            indices[i] = index;
        }
        std::ranges::sort(indices.begin(), indices.end(), [grid](const auto& a, const auto& b)
        {
            return ravel_index(a, grid) < ravel_index(b, grid);
        });
        auto last = std::unique(indices.begin(), indices.end());
        indices.erase(last, indices.end());
        crit_count = static_cast<int>(indices.size());

        const auto result = get_critical_points(cfg, dim, grid, crit_slope);
        if (_i == 0)
        {
            BENCHMARK(std::format("DIM {} GRID {}", dim, grid))
            {
                get_critical_points(cfg, dim, grid, crit_slope);
            };
        }
        // REQUIRE(result.size() == crit_count);
        if ((static_cast<int>(result.size()) != crit_count) || (result != indices))
        {
            auto log_file_path(std::filesystem::path(__FILE__).parent_path() / "critical_points_error.log");
            auto log_file{std::ofstream(log_file_path.string())};
            log_file << std::format("failed seed: {}", seed) << std::endl;
            for (auto& r : result)
            {
                log_file << r << std::endl;
            }
            log_file << std::endl;
            for (auto& ind : indices)
            {
                log_file << ind << std::endl;
            }
        }
        REQUIRE(result == indices);
    }
}

TEST_CASE("Test Critical Points")
{
    SECTION("1D")
    {
        ndim_test(1, 40);
    }
    SECTION("2D")
    {
        ndim_test(2, 40);
    }
    SECTION("3D")
    {
        ndim_test(3, 40);
    }
    SECTION("7D")
    {
        ndim_test(7, 10);
    }
}
