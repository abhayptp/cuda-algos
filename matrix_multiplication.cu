#include <cuda_runtime.h>

#include <array>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#define N 10
#define M 10
#define K 10
#define MAX_ERR 1e-6

// Size(a) = n*k
// Size(b) = k*m
template <typename T>
__global__ void multiplyMatrixCuda(T *a, T *b, T *out, int n, int m, int k) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n && c < m) {
        T sum = 0;
        for (int i = 0; i < k; i++) {
            sum += a[r * k + i] * b[m * i + c];
        }
        out[r * m + c] = sum;
    }
}

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "Error: " << msg << " : " << cudaGetErrorString(err)
                  << std::endl;
        exit(err);
    }
}

template <typename T>
void initializeHostArrays(std::array<std::array<T>> &a,
                          std::array<std::array<T>> &b,
                          std::array<std::array<T>> &out, int n, int m, int k) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            a[i][j] = static_cast<T>(rand());
        }
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m; j++) {
            b[i][j] = static_cast<T>(rand());
        }
    }
}

template <typename T>
void multiplyMatrix(const std::array<std::array<T>> &a,
                    const std::array<std::array<T>> &b,
                    std::array<std::array<T>> &out, int n, int m, int k) {
    // Device pointers
    float *d_a, *d_b, *d_out;

    // Allocate device memory and check for errors
    cudaError_t err = cudaMalloc(&d_a, n * k * sizeof(T));
    checkCudaError(err, "Failed to allocate memory for d_a");

    err = cudaMalloc(&d_b, k * m * sizeof(T));
    checkCudaError(err, "Failed to allocate memory for d_b");

    err = cudaMalloc(&d_out, n * m * sizeof(T));
    checkCudaError(err, "Failed to allocate memory for d_out");

    // Copy data to device.
    // Using `.data()` works here because 2d vector data is
    // stored contigously in memory (in row major format).
    err = cudaMemcpy(d_a, a.data(), n * k * sizeof(T), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy a from host to device");

    err = cudaMemcpy(d_b, b.data(), k * m * sizeof(T), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy b from host to device");

    // Launch kernel
    dim3 threadBlock(32, 32);
    dim3 blockGrid((n + 31) / 32, (m + 31) / 32);
    multiplyMatrixCuda<<<blockGrid, threadBlock>>>(d_a, d_b, d_out, n, m, k);

    // Check for kernel launch errors
    err = cudaGetLastError();
    checkCudaError(err, "Kernel launch failed");

    // Copy result back to host
    err = cudaMemcpy(out.data(), d_out, n * m * sizeof(float),
                     cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy result from device to host");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

int main() {
    srand(static_cast<unsigned int>(time(nullptr)));

    std::array<std::array<float, k>, n> a;
    std::array<std::array<float, m>, k> b;
    std::array<std::array<float, m>, n> out;

    initializeHostArrays(a, b, out, N, M, K);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    multiplyMatrix(a, b, out, N, M, K);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time taken: " << milliseconds << " ms" << std::endl;

    return 0;
}
