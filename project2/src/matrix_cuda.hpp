#ifndef CSC4005_PROJECT_2_MATRIX_GPU_HPP
#define CSC4005_PROJECT_2_MATRIX_GPU_HPP

#include <iostream>
#include <vector>

/**
 * 1D array with padding.
*/
class Matrix
{
private:
  int* data;
  size_t rows;
  size_t cols;
  size_t rows_padded;
  size_t cols_padded;

public:
  // Constructor
  Matrix(size_t rows, size_t cols, size_t rows_padding, size_t cols_padding);

  // Destructor
  ~Matrix();

  // raw pointer
  inline int* raw() { return data; }
  inline int* raw() const { return data; }

  // Function to display the matrix
  void display() const;

  // Get the row numbers of a matrix
  size_t getRows() const;

  // Get the column numbers of a matrix
  size_t getCols() const;

  size_t getRowsPadded() const;

  size_t getColsPadded() const;

  // Load a matrix from a file
  static Matrix loadFromFile(const std::string& filename,
                             size_t rows_tile,
                             size_t cols_tile);

  // Save a matrix to a file
  void saveToFile(const std::string& filename) const;

  // disable copy
  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;

  // enable move
  Matrix(Matrix&&) noexcept;
  Matrix& operator=(Matrix&&) noexcept;
};

#endif
