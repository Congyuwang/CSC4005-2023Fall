#include "../matrix_cuda.hpp"
#include <cuda_runtime.h> // CUDA Header
#include <iostream>

#define TILE 8

#define CUDA_ERR(val) check((val), #val, __FILE__, __LINE__)

void
check(cudaError_t err, const char* const func, const char* const file, int const line)
{
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    throw std::exception();
  }
}

// CUDA kernel functon.
//
// !!! ASSUME ALL DIMS PADDED !!!
//
// Each block compute a tile of the result matrix.
//
// Requires 2D `TILE * TILE` THREADS_PER_BLOCK.
//
__global__ void
mat_mul(int* mat1, int* mat2, int* matr, size_t m, size_t k, size_t n)
{

  // CUDA L1 Cache
  __shared__ int mat1_tile[TILE][TILE];
  __shared__ int mat2_tile[TILE][TILE];

  int sum = 0;

  // 2D block and 2D thread
  // Each thread computes one cell in mat_3.
  size_t blkx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t blky = blockIdx.y * blockDim.y + threadIdx.y;

  for (size_t kk = 0; kk < k; kk += TILE) {

    // load M1 (blkx.., kk..)
    size_t j = kk + threadIdx.y;
    mat1_tile[threadIdx.x][threadIdx.y] = mat1[blkx * k + j];

    // load M2 (kk.., blky..)
    size_t i = kk + threadIdx.x;
    mat2_tile[threadIdx.x][threadIdx.y] = mat2[i * n + blky];

    __syncthreads();

    // compute result M1 (blkx.., kk..) @ M2 (kk.., blky..)
    for (size_t k = 0; k < TILE; ++k) {
      sum += mat1_tile[threadIdx.x][k] * mat2_tile[k][threadIdx.y];
    }

    __syncthreads();
  }

  matr[blkx * n + blky] = sum;
}

/// @brief matrix multiplication with CUDA core.
/// @param matrix1 !!PADDED
/// @param matrix2 !!PADDED
/// @return output matrix
Matrix
matrix_multiply(const Matrix& matrix1, const Matrix& matrix2)
{
  if (matrix1.getCols() != matrix2.getRows()) {
    std::cerr << "Matrix dimensions are not compatible for multiplication."
              << std::endl;
    throw std::exception();
  }

  const size_t M = matrix1.getRows(), K = matrix1.getCols(),
               N = matrix2.getCols();
  const size_t MT = matrix1.getRowsPadded(), KT = matrix1.getColsPadded(),
               NT = matrix2.getColsPadded();
  const size_t M1_SIZE = MT * KT, M2_SIZE = KT * NT, MR_SIZE = MT * NT;

  const dim3 THREADS_PER_BLOCK(TILE, TILE);
  const dim3 BLOCKS_PER_GRID(MT / TILE, NT / TILE);

  Matrix result(M, N, TILE, TILE);

  int* mat1;
  int* mat2;
  int* matr;
  CUDA_ERR(cudaMalloc((void**)&mat1, M1_SIZE * sizeof(int)));
  CUDA_ERR(cudaMalloc((void**)&mat2, M2_SIZE * sizeof(int)));
  CUDA_ERR(cudaMalloc((void**)&matr, MR_SIZE * sizeof(int)));
  // Initilize result matrix
  CUDA_ERR(cudaMemset(matr, 0, MR_SIZE * sizeof(int)));
  // Copy input matrix
  CUDA_ERR(cudaMemcpy(
    mat1, matrix1.raw(), M1_SIZE * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_ERR(cudaMemcpy(
    mat2, matrix2.raw(), M2_SIZE * sizeof(int), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  float gpuDuration;
  CUDA_ERR(cudaEventCreate(&start));
  CUDA_ERR(cudaEventCreate(&stop));

  CUDA_ERR(cudaEventRecord(start, 0)); // GPU start time
  mat_mul<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(mat1, mat2, matr, MT, KT, NT);
  CUDA_ERR(cudaEventRecord(stop, 0)); // GPU end time

  // Print time of the GPU computation
  CUDA_ERR(cudaEventSynchronize(stop));
  CUDA_ERR(cudaEventElapsedTime(&gpuDuration, start, stop));
  // Copy output data from device to host
  CUDA_ERR(cudaMemcpy(
    result.raw(), matr, MR_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

  std::cout << "Multiplication Complete!" << std::endl;
  std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds"
            << std::endl;

  return result;
}

int
main(int argc, char** argv)
{
  // Verify input argument format
  if (argc != 4) {
    std::cerr << "Invalid argument, should be: ./executable "
                 "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result"
              << std::endl;
    throw std::exception();
  }

  const std::string matrix1_path = argv[1];

  const std::string matrix2_path = argv[2];

  const std::string result_path = argv[3];

  Matrix matrix1 = Matrix::loadFromFile(matrix1_path, TILE, TILE);

  Matrix matrix2 = Matrix::loadFromFile(matrix2_path, TILE, TILE);

  Matrix result = matrix_multiply(matrix1, matrix2);

  result.saveToFile(result_path);

  std::cout << "Output file to: " << result_path << std::endl;

  return 0;
}
