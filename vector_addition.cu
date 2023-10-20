#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <cuda_runtime.h>

#define N 100000000
#define MAX_ERR 1e-6

template<typename T>
__global__ void addVectorCuda(T *a, T *b, T *out, int n) {
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "Error: " << msg << " : " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
}

template<typename T>
void initializeHostArrays(std::vector<T> &a, std::vector<T> &b, std::vector<T> &out, int n) {
    for (int i = 0; i < n; i++) {
        a.push_back(static_cast<T>(rand()));
        b.push_back(static_cast<T>(rand()));
        out.push_back(0.0f);
    }
}

template<typename T>
void addVector(const std::vector<T> &a, const std::vector<T> &b, std::vector<T> &out, int n) {
    // Device pointers
    T *d_a, *d_b, *d_out;

    // Allocate device memory and check for errors
    cudaError_t err = cudaMalloc(&d_a, n * sizeof(T));
    checkCudaError(err, "Failed to allocate memory for d_a");

    err = cudaMalloc(&d_b, n * sizeof(T));
    checkCudaError(err, "Failed to allocate memory for d_b");

    err = cudaMalloc(&d_out, n * sizeof(T));
    checkCudaError(err, "Failed to allocate memory for d_out");

    // Copy data to device
    err = cudaMemcpy(d_a, a.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy a from host to device");

    err = cudaMemcpy(d_b, b.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy b from host to device");

    // Launch kernel
    int threadsPerBlock = 1024;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    addVectorCuda<<<blocks, threadsPerBlock>>>(d_a, d_b, d_out, n);

    // Check for kernel launch errors
    err = cudaGetLastError();
    checkCudaError(err, "Kernel launch failed");

    // Copy result back to host
    err = cudaMemcpy(out.data(), d_out, n * sizeof(T), cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy result from device to host");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

int main() {
    srand(static_cast<unsigned int>(time(nullptr)));

    std::vector<float> a, b, out;

    initializeHostArrays(a, b, out, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    addVector(a, b, out, N);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time taken: " << milliseconds << " ms" << std::endl;

    for (int i = 0; i < N; i++) {
        assert(std::abs(out[i] - (a[i] + b[i])) < MAX_ERR);
    }

    return 0;
}