////////////////////////////////////////////////////////////////////////////////
// benchmark_sorts.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Compare the static sorting routine to std::sort on arrays of varying sizes
//  to determine the cross-over point and check if, e.g., a static sort of 9
//  elements is faster than sorting 6 elements with std::sort (in which case we
//  should pad and use static sorting).
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/16/2024 11:53:11
*///////////////////////////////////////////////////////////////////////////////
#include <MeshFEMSparse/Utilities/static_sort.hh>
#include <MeshFEMCore/Types.hh>
#include <algorithm>
#include <array>
#include <MeshFEMCore/GlobalBenchmark.hh>

using namespace MeshFEM;

// Prevent the compiler from optimizing away the benchmark code.
template<typename T>
void use(T &&t) {
  __asm__ __volatile__ ("" :: "g" (t));
}

template<class Derived>
void customSort(Eigen::MatrixBase<Derived> &x) {
    auto *data = x.derived().data();
    switch (x.size()) {
        case 6: return StaticTimSort<6>()(data);
        case 7: return StaticTimSort<7>()(data);
        case 8: return StaticTimSort<8>()(data);
        case 9: return StaticTimSort<9>()(data);
        default: std::sort(data, data + x.size());
    }
}

template<size_t N>
void benchmarkCustomSort(size_t n, const size_t ntests = 1000000) {
    if (n > N) return;

    VecMaxN_T<size_t, N> a(n);
    for (size_t i = 0; i < n; ++i)
        a[i] = std::rand() % 100;

    BENCHMARK_START_TIMER_SECTION(std::to_string(n) + "_custom_sort");
    for (size_t i = 0; i < ntests; ++i) {
        auto b = a;
        customSort(b);
        use(b);
    }
    BENCHMARK_STOP_TIMER_SECTION(std::to_string(n) + "_custom_sort");
}

template<size_t MinSize, size_t MaxSize>
void benchmarkDispatched(const size_t ntests = 1000000) {
    for (size_t n = MinSize; n <= MaxSize; ++n) {
        VecMaxN_T<size_t, MaxSize> a(n);
        for (size_t i = 0; i < n; ++i)
            a[i] = std::rand() % 100;

        BENCHMARK_START_TIMER_SECTION(std::to_string(n) + "_dispatched_sort_" + std::to_string(MinSize) + "_" + std::to_string(MaxSize));
        for (size_t i = 0; i < ntests; ++i) {
            auto b = a;
            dispatchedStaticSort<MinSize, MaxSize>(b.data(), b.size());
            use(b);
        }

        BENCHMARK_STOP_TIMER_SECTION(std::to_string(n) + "_dispatched_sort_" + std::to_string(MinSize) + "_" + std::to_string(MaxSize));
    }
}

template<size_t N>
void benchmarkDimension(const size_t ntests = 1000000) {
    std::array<size_t, N> a;
    for (size_t i = 0; i < N; ++i)
        a[i] = std::rand() % 100;

    StaticTimSort<N> timBoseNelsonSort;

    BENCHMARK_START_TIMER_SECTION(std::to_string(N) + "_static_sort");
    for (size_t i = 0; i < ntests; ++i) {
        std::array<size_t, N> b = a;
        timBoseNelsonSort(b);
        use(b);
    }
    BENCHMARK_STOP_TIMER_SECTION(std::to_string(N) + "_static_sort");

    BENCHMARK_START_TIMER_SECTION(std::to_string(N) + "_std_sort");
    for (size_t i = 0; i < ntests; ++i) {
        std::array<size_t, N> b = a;
        std::sort(b.begin(), b.end());
        use(b);
    }
    BENCHMARK_STOP_TIMER_SECTION(std::to_string(N) + "_std_sort");

}

int main(int argc, const char *argv[]) {
    benchmarkDimension<1>();
    benchmarkDimension<2>();
    benchmarkDimension<3>();
    benchmarkDimension<4>();
    benchmarkDimension<5>();
    benchmarkDimension<6>();
    benchmarkDimension<7>();
    benchmarkDimension<8>();
    benchmarkDimension<9>();
    benchmarkDimension<10>();
    benchmarkDimension<11>();
    benchmarkDimension<12>();
    benchmarkDimension<13>();
    benchmarkDimension<14>();
    benchmarkDimension<15>();
    benchmarkDimension<16>();

    benchmarkCustomSort<9>(6);
    benchmarkCustomSort<9>(7);
    benchmarkCustomSort<9>(8);
    benchmarkCustomSort<9>(9);

    benchmarkDispatched<6, 9>();
    benchmarkDispatched<6, 6>();

    BENCHMARK_REPORT();
    return 0;
}
