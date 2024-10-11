#include <cuda_runtime.h>

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>

#include "gpu_test.hpp"

constexpr size_t VECTOR_BLOCK_SIZE = 256;
constexpr size_t VECTOR_SIZE = 8'000'000;

constexpr size_t MATRIX_BLOCK_SIZE = 32;
constexpr size_t MATRIX_SIZE = 8192;

constexpr size_t SIMPLE_MATRIX_BLOCK_SIZE = 3;
constexpr size_t SIMPLE_MATRIX_SIZE = 3;

__global__ void _vectorAddKernel(double const *__restrict__ a, double const *__restrict__ b, double *c, size_t n)
{
    if (!a || !b)
        return;

    size_t idx{blockIdx.x * blockDim.x + threadIdx.x};
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

__global__ void _matrixMultKernel(double const *__restrict__ a, double const *__restrict__ b, double *c, size_t n)
{
    if (!a || !b)
        return;

    size_t row{blockIdx.y * blockDim.y + threadIdx.y};
    size_t col{blockIdx.x * blockDim.x + threadIdx.x};

    if (row < n && col < n)
    {
        double sum{};
        for (size_t i{}; i < n; ++i)
            sum += a[row * n + i] * b[i * n + col];
        c[row * n + col] = sum;
    }
}

void _vectorAdd(double const *__restrict__ a, double const *__restrict__ b, double *c, size_t n)
{
    if (a == nullptr || b == nullptr)
        throw std::runtime_error("Input ptrs can't be 'nullptr'");

    double *d_a, *d_b, *d_c;
    cudaMalloc(std::addressof(d_a), n * sizeof(double));
    cudaMalloc(std::addressof(d_b), n * sizeof(double));
    cudaMalloc(std::addressof(d_c), n * sizeof(double));

    cudaMemcpy(d_a, a, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice);

    size_t numBlocks{(n + VECTOR_BLOCK_SIZE - 1) / VECTOR_BLOCK_SIZE};
    _vectorAddKernel<<<numBlocks, VECTOR_BLOCK_SIZE>>>(d_a, d_b, d_c, n);

    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void _matrixMultiply(double const *__restrict__ a, double const *__restrict__ b, double *c, size_t n)
{
    double *d_a, *d_b, *d_c;
    size_t size{n * n * sizeof(double)};

    // Allocate memory on the device
    cudaMalloc(std::addressof(d_a), size);
    cudaMalloc(std::addressof(d_b), size);
    cudaMalloc(std::addressof(d_c), size);

    // Copy matrices from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 block{MATRIX_BLOCK_SIZE, MATRIX_BLOCK_SIZE};
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    _matrixMultKernel<<<grid, block>>>(d_a, d_b, d_c, n);

    cudaDeviceSynchronize();

    // Copy result matrix back from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void _matrixMultiplySimple(double const *__restrict__ a, double const *__restrict__ b, double *c, size_t n)
{
    double *d_a, *d_b, *d_c;
    size_t size{n * n * sizeof(double)};

    // Allocate memory on the device
    cudaMalloc(std::addressof(d_a), size);
    cudaMalloc(std::addressof(d_b), size);
    cudaMalloc(std::addressof(d_c), size);

    // Copy matrices from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 block{SIMPLE_MATRIX_BLOCK_SIZE, SIMPLE_MATRIX_BLOCK_SIZE};
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    _matrixMultKernel<<<grid, block>>>(d_a, d_b, d_c, n);

    cudaDeviceSynchronize();

    // Copy result matrix back from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void _initializeVector(double *vec, size_t n)
{
    for (size_t i{}; i < n; ++i)
        vec[i] = static_cast<double>(rand()) / RAND_MAX;
}

void _initializeMatrix(double *matrix, size_t n)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i{}; i < n * n; ++i)
        matrix[i] = dis(gen);
}

void _printVectorAddition(double const *__restrict__ a, double const *__restrict__ b, double const *__restrict__ c, size_t n)
{
    std::cout << "Vector Addition:\n\n";
    for (size_t i{}; i < 10; ++i)
        std::cout << "| " << a[i] << " |\t+\t| " << b[i] << " |\t=\t| " << c[i] << " |\n";
    std::cout << "...\n\n";
}

void _printMatrixMultiplication(double const *__restrict__ a, double const *__restrict__ b, double const *__restrict__ c, size_t n)
{
    std::cout << "Matrix Multiplication:\n\n";
    for (size_t i{}; i < 5; ++i)
    {
        for (size_t j{}; j < 5; ++j)
            std::cout << "| " << a[i * n + j] << " ";
        std::cout << "|\t*\t";

        for (size_t j{}; j < 5; ++j)
            std::cout << "| " << b[i * n + j] << " ";
        std::cout << "|\t=\t";

        for (size_t j{}; j < 5; ++j)
            std::cout << "| " << c[i * n + j] << " ";
        std::cout << "|\n";
    }
    std::cout << "...\n\n";
}

void run_vector_add()
{
    srand(time(0));

    double *h_a{new double[VECTOR_SIZE]},
        *h_b{new double[VECTOR_SIZE]},
        *h_c{new double[VECTOR_SIZE]};

    _initializeVector(h_a, VECTOR_SIZE);
    _initializeVector(h_b, VECTOR_SIZE);

    _vectorAdd(h_a, h_b, h_c, VECTOR_SIZE);
    _printVectorAddition(h_a, h_b, h_c, VECTOR_SIZE);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}

void run_matrix_mult()
{
    srand(time(0));

    double *h_a{new double[MATRIX_SIZE * MATRIX_SIZE]},
        *h_b{new double[MATRIX_SIZE * MATRIX_SIZE]},
        *h_c{new double[MATRIX_SIZE * MATRIX_SIZE]};

    _initializeMatrix(h_a, MATRIX_SIZE);
    _initializeMatrix(h_b, MATRIX_SIZE);

    _matrixMultiply(h_a, h_b, h_c, MATRIX_SIZE);
    _printMatrixMultiplication(h_a, h_b, h_c, MATRIX_SIZE);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}

void _printMatrix(const double *matrix, size_t n)
{
    for (size_t i{}; i < n; ++i)
    {
        for (size_t j{}; j < n; ++j)
            std::cout << matrix[i * n + j] << " ";
        std::cout << "\n";
    }
}

void run_matrix_mult_simple()
{
    double a[SIMPLE_MATRIX_SIZE * SIMPLE_MATRIX_SIZE] = {5, 2, 4, 1, 7, 1, 5, 6, 3};
    double b[SIMPLE_MATRIX_SIZE * SIMPLE_MATRIX_SIZE] = {5, 8, 0, 5, 5, 2, 7, 5, 1};
    double c[SIMPLE_MATRIX_SIZE * SIMPLE_MATRIX_SIZE] = {0};

    _matrixMultiplySimple(a, b, c, SIMPLE_MATRIX_SIZE);

    std::cout << "Matrix A:\n";
    _printMatrix(a, SIMPLE_MATRIX_SIZE);

    std::cout << "\nMatrix B:\n";
    _printMatrix(b, SIMPLE_MATRIX_SIZE);

    std::cout << "\nMatrix C (Result):\n";
    _printMatrix(c, SIMPLE_MATRIX_SIZE);
}
