//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Simple Matrix Declaration
//

#ifndef CSC4005_PROJECT_2_MATRIX_HPP
#define CSC4005_PROJECT_2_MATRIX_HPP

#include <iostream>
#include <vector>

const uintptr_t ALIGN_MASK_32 = ~0b11111;

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
  int* operator[](size_t rowIndex);

  // Read only element access
  const int* operator[](size_t rowIndex) const;

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
