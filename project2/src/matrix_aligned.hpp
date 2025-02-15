#ifndef CSC4005_PROJECT_2_MATRIX_ALIGNED_HPP
#define CSC4005_PROJECT_2_MATRIX_ALIGNED_HPP

#include <iostream>
#include <vector>

const uintptr_t ALIGN_MASK_32 = ~0b111111;

class MatrixAligned
{
private:
  int** mem;
  int** data;
  size_t rows;
  size_t cols;

public:
  // Constructor
  MatrixAligned(size_t rows, size_t cols);

  // Destructor
  ~MatrixAligned();

  // Overload the [] operator for convenient element access
  inline int* operator[](size_t rowIndex) { return data[rowIndex]; }

  // Read only element access
  inline const int* operator[](size_t rowIndex) const { return data[rowIndex]; }

  // Function to display the matrix
  void display() const;

  // Get the row numbers of a matrix
  size_t getRows() const;

  // Get the column numbers of a matrix
  size_t getCols() const;

  // Load a matrix from a file
  static MatrixAligned loadFromFile(const std::string& filename);

  // Save a matrix to a file
  void saveToFile(const std::string& filename) const;

  // disable copy
  MatrixAligned(const MatrixAligned&) = delete;
  MatrixAligned& operator=(const MatrixAligned&) = delete;

  // enable move
  MatrixAligned(MatrixAligned&&) noexcept;
  MatrixAligned& operator=(MatrixAligned&&) noexcept;
};

#endif
