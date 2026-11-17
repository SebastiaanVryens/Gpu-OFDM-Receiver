// gpu_check: smoke test for the CUDA toolchain.
// Adds two vectors on the GPU, verifies the result on the CPU and prints the
// device and CUDA versions in use. Exit code 0 means the toolchain works.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Minimal error check for Phase 0; replaced by a proper CUDA_CHECK in step 1.7.
#define CHECK(call)                                                        \
  do {                                                                     \
    const cudaError_t err_ = (call);                                       \
    if (err_ != cudaSuccess) {                                             \
      std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__,   \
                   #call, cudaGetErrorString(err_));                       \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  // Grid-stride loop: each thread handles elements i, i + stride, ...
  // so the result is correct for any n, whatever grid size is launched.
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  int device_count = 0;
  CHECK(cudaGetDeviceCount(&device_count));
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, 0));
  int driver = 0;
  int runtime = 0;
  CHECK(cudaDriverGetVersion(&driver));
  CHECK(cudaRuntimeGetVersion(&runtime));

  std::printf("Device : %s (1 of %d; compute capability %d.%d, %d SMs, %.1f GiB)\n",
              prop.name, device_count, prop.major, prop.minor,
              prop.multiProcessorCount,
              static_cast<double>(prop.totalGlobalMem) / (1 << 30));
  std::printf("CUDA   : driver %d.%d, runtime %d.%d\n", driver / 1000,
              (driver % 1000) / 10, runtime / 1000, (runtime % 1000) / 10);

  constexpr int n = 1 << 24;  // 16M floats = 64 MiB per vector
  constexpr std::size_t bytes = n * sizeof(float);
  std::vector<float> a(n);
  std::vector<float> b(n);
  std::vector<float> c(n);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i % 1000);
    b[i] = 0.5f * static_cast<float>(i % 7);
  }

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  CHECK(cudaMalloc(&d_a, bytes));
  CHECK(cudaMalloc(&d_b, bytes));
  CHECK(cudaMalloc(&d_c, bytes));
  CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

  constexpr int block = 256;
  const int grid = 32 * prop.multiProcessorCount;  // enough blocks to fill every SM
  vector_add<<<grid, block>>>(d_a, d_b, d_c, n);
  CHECK(cudaGetLastError());       // errors in the launch itself
  CHECK(cudaDeviceSynchronize());  // errors while the kernel ran

  CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(d_a));
  CHECK(cudaFree(d_b));
  CHECK(cudaFree(d_c));

  // Float addition is correctly rounded (IEEE 754) on both CPU and GPU, so
  // the results must be bit-identical: exact comparison is intended here.
  int errors = 0;
  for (int i = 0; i < n; ++i) {
    const float expected = a[i] + b[i];
    if (c[i] != expected) {
      if (errors < 5) {
        std::fprintf(stderr, "mismatch at %d: got %f, expected %f\n", i,
                     static_cast<double>(c[i]), static_cast<double>(expected));
      }
      ++errors;
    }
  }
  if (errors != 0) {
    std::printf("FAIL: %d of %d elements wrong\n", errors, n);
    return EXIT_FAILURE;
  }
  std::printf("Result : %d-element vector add matches the CPU\nPASS\n", n);
  return EXIT_SUCCESS;
}
