#include "../matrix.hpp"
#include <chrono>
#include <stdexcept>

#define TILE 32

Matrix
matrix_multiply(const Matrix& matrix1, const Matrix& matrix2)
{
  if (matrix1.getCols() != matrix2.getRows()) {
    throw std::invalid_argument(
      "Matrix dimensions are not compatible for multiplication.");
  }

  size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

  Matrix result(M, N);

  int** mat1 = matrix1.raw();
  int** mat2 = matrix2.raw();
  int** matr = result.raw();

  std::chrono::milliseconds elapsed_time;

#pragma acc data copyin(mat1[0 : M][0 : K], mat2[0 : K][0 : N])                \
  copyout(matr[0 : M][0 : N])
  {
    auto start_time = std::chrono::high_resolution_clock::now();
#pragma acc parallel loop tile(TILE, TILE)
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        int sum = 0;
        for (size_t k = 0; k < K; ++k) {
          sum += mat1[i][k] * mat2[k][j];
        }
        matr[i][j] = sum;
      }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  }

  std::cout << "Multiplication Complete!" << std::endl;
  std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
            << std::endl;

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

  Matrix result = matrix_multiply(matrix1, matrix2);

  result.saveToFile(result_path);

  std::cout << "Output file to: " << result_path << std::endl;

  return 0;
}
