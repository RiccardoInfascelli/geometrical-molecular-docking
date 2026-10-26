#include "cuda_backend.hpp"

#include <iostream>

int main()
{
    if (cuda_smoke_test()) {
        std::cout << "CUDA smoke test passed\n";
        return 0;
    }

    std::cerr << "CUDA smoke test failed\n";
    return 1;
}