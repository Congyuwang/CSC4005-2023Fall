#include "matrix.hpp"
#include <chrono>
#include <immintrin.h>
#include <stdexcept>

#define TILE_SIZE 16

inline void
store_to_result_8(__m512i_u* result_i_j, __m512i m2_k8, __m512i m1_ik8)
{
  __m512i result_i8 = _mm512_loadu_si512(result_i_j);
  __m512i mult = _mm512_mullo_epi32(m1_ik8, m2_k8);
  __m512i add = _mm512_add_epi32(result_i8, mult);
  _mm512_storeu_si512(result_i_j, add);
}

#define M1_II(m1_ii)                                                           \
  {                                                                            \
    _mm512_set1_epi32(m1_ii[kk + 0]), _mm512_set1_epi32(m1_ii[kk + 1]),        \
      _mm512_set1_epi32(m1_ii[kk + 2]), _mm512_set1_epi32(m1_ii[kk + 3]),      \
      _mm512_set1_epi32(m1_ii[kk + 4]), _mm512_set1_epi32(m1_ii[kk + 5]),      \
      _mm512_set1_epi32(m1_ii[kk + 6]), _mm512_set1_epi32(m1_ii[kk + 7]),      \
      _mm512_set1_epi32(m1_ii[kk + 8]), _mm512_set1_epi32(m1_ii[kk + 9]),      \
      _mm512_set1_epi32(m1_ii[kk + 10]), _mm512_set1_epi32(m1_ii[kk + 11]),    \
      _mm512_set1_epi32(m1_ii[kk + 12]), _mm512_set1_epi32(m1_ii[kk + 13]),    \
      _mm512_set1_epi32(m1_ii[kk + 14]), _mm512_set1_epi32(m1_ii[kk + 15]),    \
  }

Matrix
matrix_multiply(const Matrix& matrix1, const Matrix& matrix2)
{
  if (matrix1.getCols() != matrix2.getRows()) {
    throw std::invalid_argument(
      "Matrix dimensions are not compatible for multiplication.");
  }

  size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

  Matrix result(M, N);

  // when the matrix is smaller than TILE_SIZE
  if (M < TILE_SIZE || N < TILE_SIZE || K < TILE_SIZE) {
    for (size_t i = 0; i < M; ++i) {
      const int* m1_i = matrix1[i];
      int* result_i = result[i];

      for (size_t k = 0; k < K; ++k) {
        const int m1_ik = m1_i[k];
        const int* m2_k = matrix2[k];

        for (size_t j = 0; j < N; ++j) {
          result_i[j] += m1_ik * m2_k[j];
        }
      }
    }

    return result;
  }

  for (size_t ii = 0; ii < M; ii += TILE_SIZE) {

    for (size_t kk = 0; kk < K; kk += TILE_SIZE) {

      // the complete m1_ii_kk block, splatted
      __m512i m1_ii_kk[TILE_SIZE][TILE_SIZE] = {
        M1_II(matrix1[ii + 0]),  M1_II(matrix1[ii + 1]),
        M1_II(matrix1[ii + 2]),  M1_II(matrix1[ii + 3]),
        M1_II(matrix1[ii + 4]),  M1_II(matrix1[ii + 5]),
        M1_II(matrix1[ii + 6]),  M1_II(matrix1[ii + 7]),
        M1_II(matrix1[ii + 8]),  M1_II(matrix1[ii + 9]),
        M1_II(matrix1[ii + 10]), M1_II(matrix1[ii + 11]),
        M1_II(matrix1[ii + 12]), M1_II(matrix1[ii + 13]),
        M1_II(matrix1[ii + 14]), M1_II(matrix1[ii + 15]),
      };

      for (size_t jj = 0; jj < N; jj += TILE_SIZE) {

        // the complete m2_kk_jj block
        __m512i m2_kk_jj[TILE_SIZE] = {
          _mm512_loadu_si512((__m512i_u*)(matrix2[kk + 0] + jj)),
          _mm512_loadu_si512((__m512i_u*)(matrix2[kk + 1] + jj)),
          _mm512_loadu_si512((__m512i_u*)(matrix2[kk + 2] + jj)),
          _mm512_loadu_si512((__m512i_u*)(matrix2[kk + 3] + jj)),
          _mm512_loadu_si512((__m512i_u*)(matrix2[kk + 4] + jj)),
          _mm512_loadu_si512((__m512i_u*)(matrix2[kk + 5] + jj)),
          _mm512_loadu_si512((__m512i_u*)(matrix2[kk + 6] + jj)),
          _mm512_loadu_si512((__m512i_u*)(matrix2[kk + 7] + jj)),
          _mm512_loadu_si512((__m512i_u*)(matrix2[kk + 8] + jj)),
          _mm512_loadu_si512((__m512i_u*)(matrix2[kk + 9] + jj)),
          _mm512_loadu_si512((__m512i_u*)(matrix2[kk + 10] + jj)),
          _mm512_loadu_si512((__m512i_u*)(matrix2[kk + 11] + jj)),
          _mm512_loadu_si512((__m512i_u*)(matrix2[kk + 12] + jj)),
          _mm512_loadu_si512((__m512i_u*)(matrix2[kk + 13] + jj)),
          _mm512_loadu_si512((__m512i_u*)(matrix2[kk + 14] + jj)),
          _mm512_loadu_si512((__m512i_u*)(matrix2[kk + 15] + jj)),
        };

        for (size_t i = 0; i < TILE_SIZE; i++) {
          __m512i_u* result_i_j = (__m512i_u*)(result[ii + i] + jj);
          for (size_t k = 0; k < TILE_SIZE; k++) {
            store_to_result_8(result_i_j, m2_kk_jj[k], m1_ii_kk[i][k]);
          }
        }
      }
    }
  }

  return result;
}

int
main(int argc, char** argv)
{
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

  Matrix result = matrix_multiply(matrix1, matrix2);

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
