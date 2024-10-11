#ifndef GPU_TEST_HPP
#define GPU_TEST_HPP

#include <cstddef>

void vectorAdd(double const *__restrict__ a, double const *__restrict__ b, double *c, size_t n);

void matrixMultiply(double const *__restrict__ a, double const *__restrict__ b, double *c, size_t n);

void run_vector_add();

void run_matrix_mult();

void run_matrix_mult_simple();

#endif // !GPU_TEST_HPP
