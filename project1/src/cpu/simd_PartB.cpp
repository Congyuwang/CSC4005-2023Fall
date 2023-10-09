#include "utils.hpp"
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iostream>

const int FILTER_SIZE = 3;
const float filter[FILTER_SIZE][FILTER_SIZE] = {
  { 1.0 / 9, 1.0 / 9, 1.0 / 9 },
  { 1.0 / 9, 1.0 / 9, 1.0 / 9 },
  { 1.0 / 9, 1.0 / 9, 1.0 / 9 }
};

const unsigned short LOAD_MASK = 0b111111111;

inline __m512
load_row(unsigned char* base)
{
  auto chars = _mm_maskz_loadu_epi8(LOAD_MASK, base);
  auto ints = _mm512_maskz_cvtepu8_epi32(LOAD_MASK, chars);
  return _mm512_maskz_cvtepi32_ps(LOAD_MASK, ints);
}

__m512
row_filter(int row)
{
  return _mm512_setr_ps(filter[row][0], // 0, 0
                        filter[row][0], // 0, 1
                        filter[row][0], // 0, 2
                        filter[row][1], // 0, 3
                        filter[row][1], // 1, 0
                        filter[row][1], // 1, 1
                        filter[row][2], // 1, 2
                        filter[row][2], // 1, 3
                        filter[row][2], // 2, 0
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0);
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
  __m512 filter_t = row_filter(0);
  __m512 filter_m = row_filter(1);
  __m512 filter_b = row_filter(2);

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
      unsigned char* mid_base = top_base + row_length;
      unsigned char* bot_base = mid_base + row_length;

      // load value
      __m512 row_t = load_row(top_base);
      __m512 row_m = load_row(mid_base);
      __m512 row_b = load_row(bot_base);

      // apply filter
      row_t = _mm512_maskz_mul_ps(LOAD_MASK, row_t, filter_t);
      row_m = _mm512_maskz_mul_ps(LOAD_MASK, row_m, filter_m);
      row_b = _mm512_maskz_mul_ps(LOAD_MASK, row_b, filter_b);

      // aggregate rows
      __m512 result = _mm512_maskz_add_ps(LOAD_MASK, row_t, row_m);
      result = _mm512_maskz_add_ps(LOAD_MASK, result, row_b);

      // extract rgb
      __m512i result_int = _mm512_maskz_cvtps_epi32(LOAD_MASK, result);
      __m128i result_int0 = _mm512_extracti32x4_epi32(result_int, 0);
      __m128i result_int1 = _mm512_extracti32x4_epi32(result_int, 1);
      __m128i result_int2 = _mm512_extracti32x4_epi32(result_int, 2);
      int sum_r = _mm_extract_epi32(result_int0, 0) +
                  _mm_extract_epi32(result_int0, 3) +
                  _mm_extract_epi32(result_int1, 2);
      int sum_g = _mm_extract_epi32(result_int0, 1) +
                  _mm_extract_epi32(result_int1, 0) +
                  _mm_extract_epi32(result_int1, 3);
      int sum_b = _mm_extract_epi32(result_int0, 2) +
                  _mm_extract_epi32(result_int1, 1) +
                  _mm_extract_epi32(result_int2, 0);

      *(out_base + 0) = static_cast<unsigned char>(sum_r);
      *(out_base + 1) = static_cast<unsigned char>(sum_g);
      *(out_base + 2) = static_cast<unsigned char>(sum_b);
    }
    top_base += input_jpeg.num_channels * 2;
    out_base += input_jpeg.num_channels * 2;
  }
  auto end_time =
    std::chrono::high_resolution_clock::now(); // Stop recording time
  auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    end_time - start_time);

  // Save output Gray JPEG Image
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
