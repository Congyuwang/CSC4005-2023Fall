//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// SIMD + Reordering Matrix Multiplication
//

#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

inline void
simd_add_to_result_i(int* result_i,
                     const int m1_ik,
                     const int* m2_k,
                     const size_t N)
{
  const size_t N_BY_8 = N / 8;
  __m256i m1_ik8 = _mm256_set1_epi32(m1_ik);
  __m256i_u* result_i_j = (__m256i_u*) result_i;
  __m256i_u* m2_k_j = (__m256i_u*) m2_k;
  for (size_t j = 0; j < N_BY_8; ++j) {

    __m256i result_i8 = _mm256_loadu_si256(result_i_j);
    __m256i m2_k8 = _mm256_loadu_si256(m2_k_j);

    __m256i out = _mm256_add_epi32(result_i8, _mm256_mul_epu32(m1_ik8, m2_k8));
    _mm256_storeu_si256(result_i_j, out);

    result_i_j++;
    m2_k_j++;
  }
}

Matrix
matrix_multiply_simd(const Matrix& matrix1, const Matrix& matrix2)
{
  if (matrix1.getCols() != matrix2.getRows()) {
    throw std::invalid_argument(
      "Matrix dimensions are not compatible for multiplication.");
  }

  size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

  Matrix result(M, N);

  for (size_t i = 0; i < M; ++i) {
    const int* m1_i = matrix1[i];
    for (size_t k = 0; k < K; ++k) {
      simd_add_to_result_i(result[i], m1_i[k], matrix2[k], N);
    }
  }

  return result;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    const std::string matrix1_path = argv[1];

    const std::string matrix2_path = argv[2];

    const std::string result_path = argv[3];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_simd(matrix1, matrix2);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    return 0;
}
