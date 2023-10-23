#include "matrix_aligned.hpp"
#include <fstream>
#include <iostream>
#include <memory.h>
#include <random>
#include <stdexcept>
#include <vector>

MatrixAligned::MatrixAligned(size_t rows, size_t cols)
  : rows(rows)
  , cols(cols)
{
  // Allocate memory for the matrix
  mem = new int*[rows];
  data = new int*[rows];
  for (size_t i = 0; i < rows; ++i) {
    // +8 for SIMD alignment
    mem[i] = new int[cols + 32];
    memset(mem[i], 0, (cols + 32) * sizeof(int));
    data[i] = (int*)(((uintptr_t)mem[i] + 63) & ALIGN_MASK_32);
  }
}

MatrixAligned::~MatrixAligned()
{
  // Destructor to free memory
  if (mem != nullptr) {
    for (size_t i = 0; i < rows; ++i) {
      delete[] mem[i];
    }
    delete[] mem;
  }
}

void
MatrixAligned::display() const
{
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      std::cout << data[i][j] << ' ';
    }
    std::cout << std::endl;
  }
}

size_t
MatrixAligned::getRows() const
{
  return rows;
}

size_t
MatrixAligned::getCols() const
{
  return cols;
}

MatrixAligned
MatrixAligned::loadFromFile(const std::string& filename)
{
  std::ifstream inputFile(filename);
  if (!inputFile.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  size_t rows, cols;
  inputFile >> rows >> cols;

  MatrixAligned loadedMatrix(rows, cols);

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      if (!(inputFile >> loadedMatrix[i][j])) {
        throw std::runtime_error("Error reading data from file: " + filename);
      }
    }
  }

  inputFile.close();

  return loadedMatrix;
}

void
MatrixAligned::saveToFile(const std::string& filename) const
{
  std::ofstream outputFile(filename);
  if (!outputFile.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }

  outputFile << rows << ' ' << cols << std::endl;

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      outputFile << data[i][j] << ' ';
    }
    outputFile << std::endl;
  }

  outputFile.close();
}

MatrixAligned::MatrixAligned(MatrixAligned&& other) noexcept
{
  mem = other.mem;
  data = other.data;
  rows = other.rows;
  cols = other.cols;
  other.mem = nullptr;
  other.data = nullptr;
  other.rows = 0;
  other.cols = 0;
}

MatrixAligned&
MatrixAligned::operator=(MatrixAligned&& other) noexcept
{
  // prevent self-assignment
  if (this == &other) {
    return *this;
  }
  // Free the memory of the current object
  if (mem != nullptr) {
    for (size_t i = 0; i < rows; ++i) {
      delete[] mem[i];
    }
    delete[] mem;
  }
  // Move the data from the other object
  mem = other.mem;
  data = other.data;
  rows = other.rows;
  cols = other.cols;
  // Reset the other object
  other.mem = nullptr;
  other.data = nullptr;
  other.rows = 0;
  other.cols = 0;
  return *this;
}
