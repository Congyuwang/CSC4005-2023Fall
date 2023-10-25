#include "matrix_cuda.hpp"
#include <fstream>
#include <iostream>
#include <memory.h>
#include <random>
#include <stdexcept>
#include <vector>

Matrix::Matrix(size_t rows,
               size_t cols,
               size_t rows_tile,
               size_t cols_tile)
  : rows(rows)
  , cols(cols)
{
  if (rows == 0 || cols == 0) {
    rows_padded = 0;
    cols_padded = 0;
    data = nullptr;
    return;
  }
  rows_padded = (1 + ((rows - 1) / rows_tile)) * rows_tile;
  cols_padded = (1 + ((cols - 1) / cols_tile)) * cols_tile;
  // Allocate memory for the matrix
  data = new int[rows_padded * cols_padded];
}

Matrix::~Matrix()
{
  // Destructor to free memory
  if (data != nullptr) {
    delete[] data;
  }
}

void
Matrix::display() const
{
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      std::cout << data[i * cols_padded + j] << ' ';
    }
    std::cout << std::endl;
  }
}

size_t
Matrix::getRows() const
{
  return rows;
}

size_t
Matrix::getCols() const
{
  return cols;
}

size_t
Matrix::getRowsPadded() const
{
  return rows_padded;
}

size_t
Matrix::getColsPadded() const
{
  return cols_padded;
}

Matrix
Matrix::loadFromFile(const std::string& filename, size_t rows_tile, size_t cols_tile)
{
  std::ifstream inputFile(filename);
  if (!inputFile.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    throw std::exception();
  }

  size_t rows, cols;
  inputFile >> rows >> cols;

  Matrix loadedMatrix(rows, cols, rows_tile, cols_tile);

  size_t cols_padded = loadedMatrix.getColsPadded();

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      if (!(inputFile >> loadedMatrix.raw()[i * cols_padded + j])) {
        std::cerr << "Error reading data from file: " << filename << std::endl;
        throw std::exception();
      }
    }
  }

  inputFile.close();

  return loadedMatrix;
}

void
Matrix::saveToFile(const std::string& filename) const
{
  std::ofstream outputFile(filename);
  if (!outputFile.is_open()) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    throw std::exception();
  }

  outputFile << rows << ' ' << cols << std::endl;

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      outputFile << data[i * cols_padded + j] << ' ';
    }
    outputFile << std::endl;
  }

  outputFile.close();
}

Matrix::Matrix(Matrix&& other) noexcept
{
  data = other.data;
  rows = other.rows;
  cols = other.cols;
  rows_padded = other.rows_padded;
  cols_padded = other.cols_padded;
  other.data = nullptr;
  other.rows = 0;
  other.cols = 0;
  other.rows_padded = 0;
  other.cols_padded = 0;
}

Matrix&
Matrix::operator=(Matrix&& other) noexcept
{
  // prevent self-assignment
  if (this == &other) {
    return *this;
  }
  // Free the memory of the current object
  if (data != nullptr) {
    delete[] data;
  }
  // Move the data from the other object
  data = other.data;
  rows = other.rows;
  cols = other.cols;
  rows_padded = other.rows_padded;
  cols_padded = other.cols_padded;
  // Reset the other object
  other.data = nullptr;
  other.rows = 0;
  other.cols = 0;
  other.rows_padded = 0;
  other.cols_padded = 0;
  return *this;
}
