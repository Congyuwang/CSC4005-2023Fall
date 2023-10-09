#include "PartB_Core.hpp"
#include "utils.hpp"
#include <chrono>
#include <iostream>
#include <pthread.h>

// Structure to pass data to each thread
struct ThreadData
{
  // buffers
  unsigned char* input_buf;
  unsigned char* output_buf;
  // image info
  int width;
  int num_channels;
  // task info
  int start_height;
  int end_height;
  // filters
  __m256 filter_t;
  __m256 filter_m;
  __m256 filter_b;
};

// Function to convert RGB to Grayscale for a portion of the image
void*
smooth_picture(void* arg)
{
  ThreadData* data = reinterpret_cast<ThreadData*>(arg);

  int width = data->width;
  int num_channels = data->num_channels;
  __m256 filter_t = data->filter_t;
  __m256 filter_m = data->filter_m;
  __m256 filter_b = data->filter_b;
  int row_length = width * num_channels;
  int task_offset = data->start_height * row_length;
  unsigned char* top_base = data->input_buf + task_offset;
  unsigned char* out_base = data->output_buf + task_offset + row_length;

  for (int h = data->start_height; h < data->end_height; h++) {
    for (int i = 1; i < width - 1; i++) {
      top_base += num_channels;
      out_base += num_channels;
      smooth_single_px_simd(
        top_base, out_base, row_length, filter_t, filter_m, filter_b);
    }
    top_base += num_channels * 2;
    out_base += num_channels * 2;
  }

  return nullptr;
}

int
main(int argc, char** argv)
{
  // Verify input argument format
  if (argc != 4) {
    std::cerr << "Invalid argument, should be: ./executable "
                 "/path/to/input/jpeg /path/to/output/jpeg num_threads\n";
    return -1;
  }

  int num_threads = std::stoi(argv[3]); // User-specified thread count

  // Read from input JPEG
  const char* input_filepath = argv[1];
  std::cout << "Input file from: " << input_filepath << "\n";
  auto input_jpeg = read_from_jpeg(input_filepath);

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

  pthread_t threads[num_threads];
  ThreadData thread_data[num_threads];

  auto start_time = std::chrono::high_resolution_clock::now();

  int chunk_size = (input_jpeg.height - 2) / num_threads;
  if (chunk_size == 0) {
    std::cerr << "Too many threads for this image size\n";
    return -1;
  }
  for (int i = 0; i < num_threads; i++) {
    thread_data[i].input_buf = input_jpeg.buffer;
    thread_data[i].output_buf = filteredImage;
    thread_data[i].filter_t = filter_t;
    thread_data[i].filter_m = filter_m;
    thread_data[i].filter_b = filter_b;
    thread_data[i].width = input_jpeg.width;
    thread_data[i].num_channels = input_jpeg.num_channels;

    thread_data[i].start_height = i * chunk_size;
    thread_data[i].end_height =
      (i == num_threads - 1) ? input_jpeg.height - 2 : (i + 1) * chunk_size;

    pthread_create(&threads[i], nullptr, smooth_picture, &thread_data[i]);
  }

  // Wait for all threads to finish
  for (int i = 0; i < num_threads; i++) {
    pthread_join(threads[i], nullptr);
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

  // Release allocated memory
  delete[] input_jpeg.buffer;
  delete[] filteredImage;
  std::cout << "Transformation Complete!" << std::endl;
  std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
  return 0;
}
