#include <iostream>
#include <chrono>
#include <cmath>
#include <immintrin.h>

#include "utils.hpp"

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
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

    // Apply the filter to the image
    auto filteredImage =
        new unsigned char[input_jpeg.width * input_jpeg.height *
                          input_jpeg.num_channels];
    for (int i = 0;
         i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels;
         ++i)
        filteredImage[i] = 0;

    // Using SIMD to accelerate the transformation
    auto start_time = std::chrono::high_resolution_clock::now();    // Start recording time
    // Nested for loop, please optimize it
    for (int height = 1; height < input_jpeg.height - 1; height++)
    {
        for (int width = 1; width < input_jpeg.width - 1; width++)
        {
            int top_base = ((height - 1) * input_jpeg.width + (width - 1)) *
                           input_jpeg.num_channels;
            int mid_base = (height * input_jpeg.width + (width - 1)) *
                           input_jpeg.num_channels;
            int bot_base = ((height + 1) * input_jpeg.width + (width - 1)) *
                           input_jpeg.num_channels;
            unsigned char cv_l_t_r = input_jpeg.buffer[top_base];
            unsigned char cv_l_t_g = input_jpeg.buffer[top_base + 1];
            unsigned char cv_l_t_b = input_jpeg.buffer[top_base + 2];
            unsigned char cv_m_t_r = input_jpeg.buffer[top_base + 3];
            unsigned char cv_m_t_g = input_jpeg.buffer[top_base + 4];
            unsigned char cv_m_t_b = input_jpeg.buffer[top_base + 5];
            unsigned char cv_r_t_r = input_jpeg.buffer[top_base + 6];
            unsigned char cv_r_t_g = input_jpeg.buffer[top_base + 7];
            unsigned char cv_r_t_b = input_jpeg.buffer[top_base + 8];
            unsigned char cv_l_m_r = input_jpeg.buffer[mid_base];
            unsigned char cv_l_m_g = input_jpeg.buffer[mid_base + 1];
            unsigned char cv_l_m_b = input_jpeg.buffer[mid_base + 2];
            unsigned char cv_m_m_r = input_jpeg.buffer[mid_base + 3];
            unsigned char cv_m_m_g = input_jpeg.buffer[mid_base + 4];
            unsigned char cv_m_m_b = input_jpeg.buffer[mid_base + 5];
            unsigned char cv_r_m_r = input_jpeg.buffer[mid_base + 6];
            unsigned char cv_r_m_g = input_jpeg.buffer[mid_base + 7];
            unsigned char cv_r_m_b = input_jpeg.buffer[mid_base + 8];
            unsigned char cv_l_b_r = input_jpeg.buffer[bot_base];
            unsigned char cv_l_b_g = input_jpeg.buffer[bot_base + 1];
            unsigned char cv_l_b_b = input_jpeg.buffer[bot_base + 2];
            unsigned char cv_m_b_r = input_jpeg.buffer[bot_base + 3];
            unsigned char cv_m_b_g = input_jpeg.buffer[bot_base + 4];
            unsigned char cv_m_b_b = input_jpeg.buffer[bot_base + 5];
            unsigned char cv_r_b_r = input_jpeg.buffer[bot_base + 6];
            unsigned char cv_r_b_g = input_jpeg.buffer[bot_base + 7];
            unsigned char cv_r_b_b = input_jpeg.buffer[bot_base + 8];

            __m128i sum = _mm_setzero_si128();
            _mm_add_epi32(sum, _mm_setr_epi32(cv_l_t_r, cv_l_t_g, cv_l_t_b, 0));
            _mm_add_epi32(sum, _mm_setr_epi32(cv_m_t_r, cv_m_t_g, cv_m_t_b, 0));
            _mm_add_epi32(sum, _mm_setr_epi32(cv_r_t_r, cv_r_t_g, cv_r_t_b, 0));
            _mm_add_epi32(sum, _mm_setr_epi32(cv_l_m_r, cv_l_m_g, cv_l_m_b, 0));
            _mm_add_epi32(sum, _mm_setr_epi32(cv_m_m_r, cv_m_m_g, cv_m_m_b, 0));
            _mm_add_epi32(sum, _mm_setr_epi32(cv_r_m_r, cv_r_m_g, cv_r_m_b, 0));
            _mm_add_epi32(sum, _mm_setr_epi32(cv_l_b_r, cv_l_b_g, cv_l_b_b, 0));
            _mm_add_epi32(sum, _mm_setr_epi32(cv_m_b_r, cv_m_b_g, cv_m_b_b, 0));
            _mm_add_epi32(sum, _mm_setr_epi32(cv_r_b_r, cv_r_b_g, cv_r_b_b, 0));

            filteredImage[(height * input_jpeg.width + width) *
                          input_jpeg.num_channels] =
                static_cast<unsigned char>(
                    std::round(_mm_extract_epi32(sum, 0) / 9.0));
            filteredImage[(height * input_jpeg.width + width) *
                              input_jpeg.num_channels +
                          1] =
                static_cast<unsigned char>(
                    std::round(_mm_extract_epi32(sum, 1) / 9.0));
            filteredImage[(height * input_jpeg.width + width) *
                              input_jpeg.num_channels +
                          2] =
                static_cast<unsigned char>(
                    std::round(_mm_extract_epi32(sum, 2) / 9.0));
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();  // Stop recording time
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save output Gray JPEG Image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height,
                         input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath))
    {
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
