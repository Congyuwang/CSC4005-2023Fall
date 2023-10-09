//
// Created by Yang Yufan on 2023/9/16.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI implementation of transforming a JPEG image from RGB to gray
//

#include "PartB_Core.hpp"
#include "utils.hpp"
#include <chrono>
#include <iostream>
#include <mpi.h> // MPI Header
#include <vector>

#define MASTER 0
#define TAG_GATHER 0

// Structure to pass data to each thread
struct Task
{
  int start_height;
  int end_height;
};

inline int
msg_length(Task task, int width, int channel)
{
  return (task.end_height - task.start_height) * width * channel;
}

inline void
process_task(Task task,
             const unsigned char* top_base,
             unsigned char* out_base,
             int width,
             int num_channels)
{
  int row_length = width * num_channels;
  for (int h = task.start_height; h < task.end_height; h++) {
    for (int i = 1; i < width - 1; i++) {
      smooth_single_px(top_base, out_base, row_length);
      top_base += num_channels;
      out_base += num_channels;
    }
    top_base += num_channels * 2;
    out_base += num_channels * 2;
  }
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
  // Start the MPI
  MPI_Init(&argc, &argv);
  // How many processes are running
  int numtasks;
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  // What's my rank?
  int taskid;
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  // Which node am I running on?
  int len;
  char hostname[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(hostname, &len);
  MPI_Status status;

  // Read JPEG File
  const char* input_filepath = argv[1];
  auto input_jpeg = read_from_jpeg(input_filepath);
  if (input_jpeg.buffer == NULL) {
    std::cerr << "Failed to read input JPEG image\n";
    return -1;
  }
  int width = input_jpeg.width;
  int num_channels = input_jpeg.num_channels;
  int row_length = width * num_channels;

  Task tasks[numtasks];

  auto start_time = std::chrono::high_resolution_clock::now();

  int chunk_size = (input_jpeg.height - 2) / numtasks;
  if (chunk_size == 0) {
    std::cerr << "Too many workers for this image size\n";
    return -1;
  }
  for (int i = 0; i < numtasks; i++) {
    tasks[i].start_height = i * chunk_size;
    tasks[i].end_height =
      (i == numtasks - 1) ? input_jpeg.height - 2 : (i + 1) * chunk_size;
  }

  // The tasks for the master executor
  // 1. Apply the filter to the first part of the image
  // 2. Receive the transformed Gray contents from slave executors
  // 3. Write the content to the JPEG File
  if (taskid == MASTER) {
    std::cout << "Num task: " << numtasks << "\n"
              << "hostname: " << hostname << std::endl;
    std::cout << "Input file from: " << input_filepath << "\n";

    // Apply the filter to the image
    auto filteredImage =
      new unsigned char[input_jpeg.width * input_jpeg.height *
                        input_jpeg.num_channels];
    for (int i = 0;
         i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels;
         ++i)
      filteredImage[i] = 0;

    unsigned char* top_base = input_jpeg.buffer;
    // one pixel right and one pixel bottom to top_base
    unsigned char* out_base = filteredImage + row_length + num_channels;

    auto task = tasks[MASTER];
    process_task(task, top_base, out_base, width, num_channels);

    // Receive the transformed Gray contents from each slave executors
    for (int i = MASTER + 1; i < numtasks; i++) {
      int start_height = tasks[i].start_height;
      int msg_len = msg_length(tasks[i], width, num_channels);
      unsigned char* start_pos =
        filteredImage + (start_height + 1) * row_length;
      MPI_Recv(
        start_pos, msg_len, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
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

    // Release the memory
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
  }
  // The tasks for the slave executor
  // 1. Apply the filter to the image
  // 2. Send the transformed Gray contents back to the master executor
  else {
    // Apply the filter to the image
    Task task = tasks[taskid];
    int length = msg_length(task, width, num_channels);
    unsigned char* top_base =
      input_jpeg.buffer + task.start_height * row_length;
    auto filteredImage = new unsigned char[length];
    process_task(task, top_base, filteredImage + num_channels, width, num_channels);

    // Send the gray image back to the master
    MPI_Send(
      filteredImage, length, MPI_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);

    // Release the memory
    delete[] filteredImage;
  }

  MPI_Finalize();
  return 0;
}
