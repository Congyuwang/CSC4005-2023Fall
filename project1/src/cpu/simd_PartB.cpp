#include "PartB_Core.hpp"
#include "utils.hpp"
#include <chrono>
#include <cmath>
#include <iostream>

__m256
row_filter(int row)
{
  return _mm256_setr_ps(filter[row][0], // 0, 0
                        filter[row][0], // 0, 1
                        filter[row][0], // 0, 2
                        filter[row][1], // 0, 3
                        filter[row][1], // 1, 0
                        filter[row][1], // 1, 1
                        filter[row][2], // 1, 2
                        filter[row][2]  // 1, 3
  );
}

int
main(int argc, char** argv)
{
  // Verify input argument format
  if (argc != 3) {
    std::cerr << "Invalid argument, should be: ./executable "
                 "/path/to/input/jpeg /path/to/output/jpeg\n";
    return -1;
  }
  // Read JPEG File
  const char* input_filepath = argv[1];
  std::cout << "Input file from: " << input_filepath << "\n";
  auto input_jpeg = read_from_jpeg(input_filepath);
  if (input_jpeg.buffer == NULL) {
    std::cerr << "Failed to read input JPEG image\n";
    return -1;
  }

  // Prepare Consts
  __m256 filter_t = row_filter(0);
  __m256 filter_m = row_filter(1);
  __m256 filter_b = row_filter(2);

  // Apply the filter to the image
  auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height *
                                         input_jpeg.num_channels];
  for (int i = 0;
       i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels;
       ++i)
    filteredImage[i] = 0;

  // Using SIMD to accelerate the transformation
  auto start_time =
    std::chrono::high_resolution_clock::now(); // Start recording time

  int row_length = input_jpeg.width * input_jpeg.num_channels;
  unsigned char* top_base = input_jpeg.buffer;
  unsigned char* out_base = filteredImage + row_length;
  for (int height = 1; height < input_jpeg.height - 1; height++) {
    for (int width = 1; width < input_jpeg.width - 1; width++) {
      top_base += input_jpeg.num_channels;
      out_base += input_jpeg.num_channels;
      smooth_single_px_simd(top_base, out_base, row_length, filter_t, filter_m, filter_b);
    }
    top_base += input_jpeg.num_channels * 2;
    out_base += input_jpeg.num_channels * 2;
  }
  auto end_time =
    std::chrono::high_resolution_clock::now(); // Stop recording time
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

  // Release allocated memory
  delete[] input_jpeg.buffer;
  delete[] filteredImage;
  std::cout << "Transformation Complete!" << std::endl;
  std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
  return 0;
}
