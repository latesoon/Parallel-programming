#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>
#include <cstdlib>

#define N 1024
#define BLOCK_SIZE 32

__global__ void division_kernel(float* data, int k, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    if (tid < n) {
        float element = data[k * n + k];
        float temp = data[k * n + tid];
        data[k * n + tid] = temp / element;
    }
}

__global__ void eliminate_kernel(float* data, int k, int n) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tx == 0) 
        data[k * n + k] = 1.0f;
    int row = k + 1 + blockIdx.x;
    while (row < n) {
        int tid = threadIdx.x;
        while (k + 1 + tid < n) {
            int col = k + 1 + tid;
            float temp_1 = data[row * n + col];
            float temp_2 = data[row * n + k];
            float temp_3 = data[k * n + col];
            data[row * n + col] = temp_1 - temp_2 * temp_3;
            tid = tid + blockDim.x;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            data[row * n + k] = 0;
        }
        row += gridDim.x;
    }
}
void cud(float* data_D, int n) {
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    cudaError_t ret;
    for (int k = 0; k < n; k++) {
        division_kernel<<<grid, block>>>(data_D, k, n);
        cudaDeviceSynchronize(); 
        ret = cudaGetLastError();
        if (ret != cudaSuccess) {
            printf("division_kernel failed, %s\n", cudaGetErrorString(ret));
        }

        eliminate_kernel<<<grid, block>>>(data_D, k, n);
        cudaDeviceSynchronize();
        ret = cudaGetLastError();
        if (ret != cudaSuccess) {
            printf("eliminate_kernel failed, %s\n", cudaGetErrorString(ret));
        }
    }
}

void common(float* data, int n) {
    for (int k = 0; k < n; k++) {
        float element = data[k * n + k];
        for (int j = k; j < n; j++) {
            data[k * n + j] /= element;
        }
        for (int i = k + 1; i < n; i++) {
            float factor = data[i * n + k];
            for (int j = k; j < n; j++) {
                data[i * n + j] -= factor * data[k * n + j];
            }
            data[i * n + k] = 0;
        }
    }
}

void op(float* data_H, int n, int iterations, void(*elimination_func)(float*, int)) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        float* data_copy = new float[n * n];
        std::copy(data_H, data_H + n * n, data_copy);
        elimination_func(data_copy, n);
        delete[] data_copy;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time taken for " << iterations << " iterations: " << diff.count() << " seconds" << std::endl;
}

int main() {
    float* data_H = new float[N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            data_H[i * N + j] = static_cast<float>(rand() % 100);
        }
    }

    int iterations = 10;

    std::cout << "CUDA:" << std::endl;
    op(data_H, N, iterations, [](float* data, int n) {
        float* data_D;
        cudaMalloc((void**)&data_D, n * n * sizeof(float));
        cudaMemcpy(data_D, data, n * n * sizeof(float), cudaMemcpyHostToDevice);
        cud(data_D, n);
        cudaMemcpy(data, data_D, n * n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(data_D);
    });

    std::cout << "common:" << std::endl;
    op(data_H, N, iterations, common);

    delete[] data_H;
    return 0;
}