#include <iostream>
#include <cuda_runtime.h> // CUDA Header
#include "utils.hpp"

const int FILTER_SIZE = 3;
__constant__ float filter[FILTER_SIZE][FILTER_SIZE] = {
  { 1.0 / 9, 1.0 / 9, 1.0 / 9 },
  { 1.0 / 9, 1.0 / 9, 1.0 / 9 },
  { 1.0 / 9, 1.0 / 9, 1.0 / 9 }
};

/**
 * Sequentailly compute a single px.
 */
__global__ void
smooth_single_px(const unsigned char* input_buf,
                 unsigned char* output_buf,
                 const int row_length,
                 const int width,
                 const int height,
                 const int channel)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // keep the border
int h = idx / width;
  int w = idx % width;
  if (w >= width - 2 || h >= height - 2) {
    return;
  }

  idx *= channel;
  int mid_idx = idx + row_length;
  int out_idx = mid_idx + channel;
  int bot_idx = mid_idx + row_length;

  unsigned char cv_l_t_r = input_buf[idx + 0];
  unsigned char cv_l_t_g = input_buf[idx + 1];
  unsigned char cv_l_t_b = input_buf[idx + 2];
  unsigned char cv_m_t_r = input_buf[idx + 3];
  unsigned char cv_m_t_g = input_buf[idx + 4];
  unsigned char cv_m_t_b = input_buf[idx + 5];
  unsigned char cv_r_t_r = input_buf[idx + 6];
  unsigned char cv_r_t_g = input_buf[idx + 7];
  unsigned char cv_r_t_b = input_buf[idx + 8];

  unsigned char cv_l_m_r = input_buf[mid_idx + 0];
  unsigned char cv_l_m_g = input_buf[mid_idx + 1];
  unsigned char cv_l_m_b = input_buf[mid_idx + 2];
  unsigned char cv_m_m_r = input_buf[mid_idx + 3];
  unsigned char cv_m_m_g = input_buf[mid_idx + 4];
  unsigned char cv_m_m_b = input_buf[mid_idx + 5];
  unsigned char cv_r_m_r = input_buf[mid_idx + 6];
  unsigned char cv_r_m_g = input_buf[mid_idx + 7];
  unsigned char cv_r_m_b = input_buf[mid_idx + 8];

  unsigned char cv_l_b_r = input_buf[bot_idx + 0];
  unsigned char cv_l_b_g = input_buf[bot_idx + 1];
  unsigned char cv_l_b_b = input_buf[bot_idx + 2];
  unsigned char cv_m_b_r = input_buf[bot_idx + 3];
  unsigned char cv_m_b_g = input_buf[bot_idx + 4];
  unsigned char cv_m_b_b = input_buf[bot_idx + 5];
  unsigned char cv_r_b_r = input_buf[bot_idx + 6];
  unsigned char cv_r_b_g = input_buf[bot_idx + 7];
  unsigned char cv_r_b_b = input_buf[bot_idx + 8];

  int sum_r = cv_l_t_r * filter[0][0] + cv_m_t_r * filter[0][1] +
              cv_r_t_r * filter[0][2] + cv_l_m_r * filter[1][0] +
              cv_m_m_r * filter[1][1] + cv_r_m_r * filter[1][2] +
              cv_l_b_r * filter[2][0] + cv_m_b_r * filter[2][1] +
              cv_r_b_r * filter[2][2];

  int sum_g = cv_l_t_g * filter[0][0] + cv_m_t_g * filter[0][1] +
              cv_r_t_g * filter[0][2] + cv_l_m_g * filter[1][0] +
              cv_m_m_g * filter[1][1] + cv_r_m_g * filter[1][2] +
              cv_l_b_g * filter[2][0] + cv_m_b_g * filter[2][1] +
              cv_r_b_g * filter[2][2];

  int sum_b = cv_l_t_b * filter[0][0] + cv_m_t_b * filter[0][1] +
              cv_r_t_b * filter[0][2] + cv_l_m_b * filter[1][0] +
              cv_m_m_b * filter[1][1] + cv_r_m_b * filter[1][2] +
              cv_l_b_b * filter[2][0] + cv_m_b_b * filter[2][1] +
              cv_r_b_b * filter[2][2];

  output_buf[out_idx + 0] = static_cast<unsigned char>(sum_r);
  output_buf[out_idx + 1] = static_cast<unsigned char>(sum_g);
  output_buf[out_idx + 2] = static_cast<unsigned char>(sum_b);
}

int main(int argc, char** argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read from input JPEG
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    // Allocate memory on host (CPU)
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height *
                                           input_jpeg.num_channels];
    // Allocate memory on device (GPU)
    unsigned char* d_input;
    unsigned char* d_output;
    cudaMalloc((void**)&d_input, input_jpeg.width * input_jpeg.height *
                                 input_jpeg.num_channels * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, input_jpeg.width * input_jpeg.height *
                                  input_jpeg.num_channels * sizeof(unsigned char));
    // Initilize output image
    cudaMemset((void**)&d_output, 0, input_jpeg.width * input_jpeg.height *
                                     input_jpeg.num_channels * sizeof(unsigned char));
    // Copy input data from host to device
    cudaMemcpy(d_input, input_jpeg.buffer, input_jpeg.width * input_jpeg.height *
                                           input_jpeg.num_channels * sizeof(unsigned char),
               cudaMemcpyHostToDevice);
    // Computation: Smoothing
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int blockSize = 512;
    int numBlocks = (input_jpeg.width * (input_jpeg.height - 2)) / blockSize + 1;
    int row_length = input_jpeg.width * input_jpeg.num_channels;
    cudaEventRecord(start, 0); // GPU start time
    smooth_single_px<<<numBlocks, blockSize>>>(
        d_input,
        d_output,
        row_length,
        input_jpeg.width,
        input_jpeg.height,
        input_jpeg.num_channels
    );
    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);
    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);
    // Copy output data from device to host
    cudaMemcpy(filteredImage, d_output, input_jpeg.width * input_jpeg.height *
                                        input_jpeg.num_channels * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);
    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{ filteredImage,
                          input_jpeg.width,
                          input_jpeg.height,
                          input_jpeg.num_channels,
                          input_jpeg.color_space };
    if (write_to_jpeg(output_jpeg, output_filepath)) {
      std::cerr << "Failed to write output JPEG\n";
      return -1;
    }
    // Release allocated memory on device and host
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
