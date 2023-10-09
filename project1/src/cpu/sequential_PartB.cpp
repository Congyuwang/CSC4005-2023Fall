//
// Created by Liu Yuxuan on 2023/9/15.
// Email: yuxuanliu1@link.cuhk.edu.cm
//
// A naive sequential implementation of image filtering
//

#include "PartB_Core.hpp"
#include "utils.hpp"
#include <chrono>
#include <cmath>
#include <iostream>

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
  int row_length = input_jpeg.width * input_jpeg.num_channels;
  unsigned char* top_base = input_jpeg.buffer;
  unsigned char* out_base = filteredImage + row_length;
  for (int height = 1; height < input_jpeg.height - 1; height++) {
    for (int width = 1; width < input_jpeg.width - 1; width++) {
      top_base += input_jpeg.num_channels;
      out_base += input_jpeg.num_channels;
      smooth_single_px(top_base, out_base, row_length);
    }
    top_base += input_jpeg.num_channels * 2;
    out_base += input_jpeg.num_channels * 2;
  }
  // End timer
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
