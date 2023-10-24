# Required packages and flags
## MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    message("MPI FOUND, Version: ${MPI_CXX_VERSION}")
endif ()
## OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    message("OPENMP FOUND, Version: " ${OpenMP_CXX_VERSION})
endif ()

# Add Executables
## Naive Old
add_executable(naive_old
        naive_old.cpp
        matrix_old.cpp matrix_old.hpp)
target_compile_options(naive_old PRIVATE -O2)

## Naive
add_executable(naive
        naive.cpp
        matrix.cpp matrix.hpp)
target_compile_options(naive PRIVATE -O2)

## Memory Locality
add_executable(locality
        locality.cpp
        matrix.cpp matrix.hpp)
target_compile_options(locality PRIVATE -O2)

## SIMD Vectorization (AVX2)
add_executable(simd
        simd.cpp
        matrix.cpp matrix.hpp)
target_compile_options(simd PRIVATE -O2 -mavx2 -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store)

## SIMD Vectorization Aligned (AVX2)
add_executable(simd_aligned
        simd_aligned.cpp
        matrix_aligned.cpp matrix_aligned.hpp)
target_compile_options(simd_aligned PRIVATE -O2 -mavx2)

## SIMD Vectorization (AVX512) Tiled
add_executable(simd_tiled
        simd_tiled.cpp
        matrix.cpp matrix.hpp)
target_compile_options(simd_tiled PRIVATE -O2 -mavx512f)

## SIMD Vectorization Aligned (AVX512) Tiled
add_executable(simd_aligned_tiled
        simd_aligned_tiled.cpp
        matrix_aligned.cpp matrix_aligned.hpp)
target_compile_options(simd_aligned_tiled PRIVATE -O2 -mavx512f)

## OpenMP
add_executable(openmp
        openmp.cpp
        matrix_aligned.cpp matrix_aligned.hpp)
target_compile_options(openmp PRIVATE -O2 -fopenmp -mavx512f)
target_include_directories(openmp PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(openmp PRIVATE ${OpenMP_CXX_LIBRARIES})

## MPI
add_executable(mpi
        mpi.cpp
        matrix_aligned.cpp matrix_aligned.hpp)
target_compile_options(mpi PRIVATE -O2 -fopenmp -mavx512f)
target_include_directories(mpi PRIVATE ${MPI_CXX_INCLUDE_DIRS} ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(mpi ${MPI_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
