#include "gpu_batch_retry.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void test_partition_and_order() {
    const std::vector<std::size_t> sizes{1, 2, 1, 3, 2, 1, 1, 3, 1, 2, 1};
    std::vector<std::size_t> processed;
    std::size_t releases = 0, rejected = 0;
    run_gpu_batch_adaptive(0, sizes.size(), [&](auto begin, auto end, bool retry) {
        std::size_t bytes = 0;
        for (auto i = begin; i < end; ++i) bytes += sizes[i];
        if (bytes > 4) {
            ++rejected;
            throw GpuBatchMemoryError(bytes, 4, "test");
        }
        require(!retry, "A fitting singleton was unexpectedly retried");
        for (auto i = begin; i < end; ++i) processed.push_back(i);
    }, [&] { ++releases; });
    require(rejected && releases == rejected, "Rejected groups did not release scratch");
    require(processed.size() == sizes.size(), "Adaptive partition lost or duplicated records");
    for (std::size_t i = 0; i < sizes.size(); ++i)
        require(processed[i] == i, "Adaptive partition reordered records");
}
void test_singletons_and_failure_categories() {
    std::size_t attempts = 0, releases = 0;
    run_gpu_batch_adaptive(0, 1, [&](auto, auto, bool retry) {
        ++attempts;
        if (!retry) throw GpuBatchMemoryError(8, 4, "retained scratch");
    }, [&] { ++releases; });
    require(attempts == 2 && releases == 1, "Singleton did not retry once with released scratch");
    attempts = releases = 0;
    bool failed = false;
    try {
        run_gpu_batch_adaptive(0, 1, [&](auto, auto, bool) {
            ++attempts;
            throw GpuBatchMemoryError(8, 4, "oversized ligand");
        }, [&] { ++releases; });
    } catch (const GpuBatchMemoryError& error) {
        failed = error.required_bytes == 8 && error.limit_bytes == 4;
    }
    require(failed && attempts == 2 && releases == 1, "Oversized singleton did not fail explicitly");
    attempts = releases = 0;
    try {
        run_gpu_batch_adaptive(0, 8, [&](auto, auto, bool) {
            ++attempts;
            throw std::runtime_error("CUDA failure");
        }, [&] { ++releases; });
        throw std::logic_error("Generic CUDA failure was swallowed");
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()) == "CUDA failure", "Generic failure changed category");
    }
    require(attempts == 1 && releases == 0, "Generic failures triggered automatic retries");
    run_gpu_batch_adaptive(3, 3, [&](auto, auto, bool) { ++attempts; }, [&] { ++releases; });
    require(attempts == 1 && releases == 0, "Empty group performed work");
}
}

int main() {
    try {
        test_partition_and_order();
        test_singletons_and_failure_categories();
        std::cout << "Adaptive GPU batch partition preserves order and failure categories\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
