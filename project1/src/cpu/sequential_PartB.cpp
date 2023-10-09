//
// Created by Liu Yuxuan on 2023/9/15.
// Email: yuxuanliu1@link.cuhk.edu.cm
//
// A naive sequential implementation of image filtering
//

#include <chrono>
#include <cmath>
#include <iostream>
#include "utils.hpp"

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
  { 1.0 / 9, 1.0 / 9, 1.0 / 9 },
  { 1.0 / 9, 1.0 / 9, 1.0 / 9 },
  { 1.0 / 9, 1.0 / 9, 1.0 / 9 }
};

int
main(int argc, char** argv)
{
  if (argc != 3) {
    std::cerr << "Invalid argument, should be: ./executable "
                 "/path/to/input/jpeg /path/to/output/jpeg\n";
    return -1;
  }

  // Read input JPEG image
  const char* input_filename = argv[1];
  std::cout << "Input file from: " << input_filename << "\n";
  auto input_jpeg = read_from_jpeg(input_filename);
  if (input_jpeg.buffer == NULL) {
    std::cerr << "Failed to read input JPEG image\n";
    return -1;
  }

  // Apply the filter to the image
  auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height *
                                         input_jpeg.num_channels];
  for (int i = 0;
       i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels;
       ++i)
    filteredImage[i] = 0;

  // Start timer
  auto start_time = std::chrono::high_resolution_clock::now();
  unsigned char* top_base = input_jpeg.buffer;
  unsigned char* out_base = filteredImage;
  int row_length = input_jpeg.width * input_jpeg.num_channels;
  for (int height = 1; height < input_jpeg.height - 1; height++) {
    for (int width = 1; width < input_jpeg.width - 1; width++) {
      unsigned char* mid_base = top_base + row_length;
      unsigned char* bot_base = mid_base + row_length;

      unsigned char cv_l_t_r = top_base[0];
      unsigned char cv_l_t_g = top_base[1];
      unsigned char cv_l_t_b = top_base[2];
      unsigned char cv_m_t_r = top_base[3];
      unsigned char cv_m_t_g = top_base[4];
      unsigned char cv_m_t_b = top_base[5];
      unsigned char cv_r_t_r = top_base[6];
      unsigned char cv_r_t_g = top_base[7];
      unsigned char cv_r_t_b = top_base[8];

      unsigned char cv_l_m_r = mid_base[0];
      unsigned char cv_l_m_g = mid_base[1];
      unsigned char cv_l_m_b = mid_base[2];
      unsigned char cv_m_m_r = mid_base[3];
      unsigned char cv_m_m_g = mid_base[4];
      unsigned char cv_m_m_b = mid_base[5];
      unsigned char cv_r_m_r = mid_base[6];
      unsigned char cv_r_m_g = mid_base[7];
      unsigned char cv_r_m_b = mid_base[8];

      unsigned char cv_l_b_r = bot_base[0];
      unsigned char cv_l_b_g = bot_base[1];
      unsigned char cv_l_b_b = bot_base[2];
      unsigned char cv_m_b_r = bot_base[3];
      unsigned char cv_m_b_g = bot_base[4];
      unsigned char cv_m_b_b = bot_base[5];
      unsigned char cv_r_b_r = bot_base[6];
      unsigned char cv_r_b_g = bot_base[7];
      unsigned char cv_r_b_b = bot_base[8];

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

      *(out_base + 0) = static_cast<unsigned char>(sum_r);
      *(out_base + 1) = static_cast<unsigned char>(sum_g);
      *(out_base + 2) = static_cast<unsigned char>(sum_b);

      top_base += input_jpeg.num_channels;
      out_base += input_jpeg.num_channels;
    }
    top_base += input_jpeg.num_channels * 2;
    out_base += input_jpeg.num_channels * 2;
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    end_time - start_time);

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
  // Post-processing
  delete[] input_jpeg.buffer;
  delete[] filteredImage;
  std::cout << "Transformation Complete!" << std::endl;
  std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
  return 0;
}
