#include "utils.hpp"
#include <chrono>
#include <iostream>

const int FILTER_SIZE = 3;
const float filter[FILTER_SIZE][FILTER_SIZE] = {
  { 1.0 / 9, 1.0 / 9, 1.0 / 9 },
  { 1.0 / 9, 1.0 / 9, 1.0 / 9 },
  { 1.0 / 9, 1.0 / 9, 1.0 / 9 }
};

int
main(int argc, char** argv)
{
  // Verify input argument format
  if (argc != 3) {
    std::cerr << "Invalid argument, should be: ./executable "
                 "/path/to/input/jpeg /path/to/output/jpeg\n";
    return -1;
  }
  // Read from input JPEG
  const char* input_filepath = argv[1];
  std::cout << "Input file from: " << input_filepath << "\n";
  JPEGMeta input_jpeg = read_from_jpeg(input_filepath);
  // Computation: Smoothing
  int width = input_jpeg.width;
  int height = input_jpeg.height;
  int num_channels = input_jpeg.num_channels;
  int row_length = width * num_channels;
  unsigned char* buffer = input_jpeg.buffer;
  unsigned char* filteredImage =
    new unsigned char[width * height * num_channels];
  for (int i = 0; i < width * height * num_channels; ++i)
    filteredImage[i] = 0;
  std::chrono::milliseconds elapsed_time;

#pragma acc data copyin(buffer[0 : width * height * num_channels])
#pragma acc data copy(filteredImage[0 : width * height * num_channels])
  {
    auto start_time = std::chrono::high_resolution_clock::now();
#pragma acc parallel loop independent num_gangs(1024)
    for (int i = 0; i < width * (height - 2); i++) {
      // keep the border
      int w = i % width;
      if (w >= width - 2) {
        continue;
      }

      int idx = i * 3;
      int mid_idx = idx + row_length;
      int out_idx = mid_idx + num_channels;
      int bot_idx = mid_idx + row_length;

      unsigned char cv_l_t_r = buffer[idx + 0];
      unsigned char cv_l_t_g = buffer[idx + 1];
      unsigned char cv_l_t_b = buffer[idx + 2];
      unsigned char cv_m_t_r = buffer[idx + 3];
      unsigned char cv_m_t_g = buffer[idx + 4];
      unsigned char cv_m_t_b = buffer[idx + 5];
      unsigned char cv_r_t_r = buffer[idx + 6];
      unsigned char cv_r_t_g = buffer[idx + 7];
      unsigned char cv_r_t_b = buffer[idx + 8];

      unsigned char cv_l_m_r = buffer[mid_idx + 0];
      unsigned char cv_l_m_g = buffer[mid_idx + 1];
      unsigned char cv_l_m_b = buffer[mid_idx + 2];
      unsigned char cv_m_m_r = buffer[mid_idx + 3];
      unsigned char cv_m_m_g = buffer[mid_idx + 4];
      unsigned char cv_m_m_b = buffer[mid_idx + 5];
      unsigned char cv_r_m_r = buffer[mid_idx + 6];
      unsigned char cv_r_m_g = buffer[mid_idx + 7];
      unsigned char cv_r_m_b = buffer[mid_idx + 8];

      unsigned char cv_l_b_r = buffer[bot_idx + 0];
      unsigned char cv_l_b_g = buffer[bot_idx + 1];
      unsigned char cv_l_b_b = buffer[bot_idx + 2];
      unsigned char cv_m_b_r = buffer[bot_idx + 3];
      unsigned char cv_m_b_g = buffer[bot_idx + 4];
      unsigned char cv_m_b_b = buffer[bot_idx + 5];
      unsigned char cv_r_b_r = buffer[bot_idx + 6];
      unsigned char cv_r_b_g = buffer[bot_idx + 7];
      unsigned char cv_r_b_b = buffer[bot_idx + 8];

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

      filteredImage[out_idx + 0] = static_cast<unsigned char>(sum_r);
      filteredImage[out_idx + 1] = static_cast<unsigned char>(sum_g);
      filteredImage[out_idx + 2] = static_cast<unsigned char>(sum_b);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  }

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
  // Release allocated memory
  delete[] input_jpeg.buffer;
  delete[] filteredImage;
  std::cout << "Transformation Complete!" << std::endl;
  std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
  return 0;
}
