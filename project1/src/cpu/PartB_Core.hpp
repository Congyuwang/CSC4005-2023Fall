#ifndef PARTB_CORE_H
#define PARTB_CORE_H

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
  { 1.0 / 9, 1.0 / 9, 1.0 / 9 },
  { 1.0 / 9, 1.0 / 9, 1.0 / 9 },
  { 1.0 / 9, 1.0 / 9, 1.0 / 9 }
};

/**
 * Sequentailly compute a single px.
 */
inline void
smooth_single_px(const unsigned char* top_base,
                 unsigned char* out_base,
                 int row_length)
{
  const unsigned char* mid_base = top_base + row_length;
  const unsigned char* bot_base = mid_base + row_length;

  unsigned char cv_l_t_r = top_base[0];
  unsigned char cv_l_t_g = top_base[1];
  unsigned char cv_l_t_b = top_base[2];
  unsigned char cv_m_t_r = top_base[3];
  unsigned char cv_m_t_g = top_base[4];
  unsigned char cv_m_t_b = top_base[5];
  unsigned char cv_r_t_r = top_base[6];
  unsigned char cv_r_t_g = top_base[7];
  unsigned char cv_r_t_b = top_base[8];

  unsigned char cv_l_m_r = mid_base[0];
  unsigned char cv_l_m_g = mid_base[1];
  unsigned char cv_l_m_b = mid_base[2];
  unsigned char cv_m_m_r = mid_base[3];
  unsigned char cv_m_m_g = mid_base[4];
  unsigned char cv_m_m_b = mid_base[5];
  unsigned char cv_r_m_r = mid_base[6];
  unsigned char cv_r_m_g = mid_base[7];
  unsigned char cv_r_m_b = mid_base[8];

  unsigned char cv_l_b_r = bot_base[0];
  unsigned char cv_l_b_g = bot_base[1];
  unsigned char cv_l_b_b = bot_base[2];
  unsigned char cv_m_b_r = bot_base[3];
  unsigned char cv_m_b_g = bot_base[4];
  unsigned char cv_m_b_b = bot_base[5];
  unsigned char cv_r_b_r = bot_base[6];
  unsigned char cv_r_b_g = bot_base[7];
  unsigned char cv_r_b_b = bot_base[8];

  int sum_r = cv_l_t_r * filter[0][0] + cv_m_t_r * filter[0][1] +
              cv_r_t_r * filter[0][2] + cv_l_m_r * filter[1][0] +
              cv_m_m_r * filter[1][1] + cv_r_m_r * filter[1][2] +
              cv_l_b_r * filter[2][0] + cv_m_b_r * filter[2][1] +
              cv_r_b_r * filter[2][2];

  int sum_g = cv_l_t_g * filter[0][0] + cv_m_t_g * filter[0][1] +
              cv_r_t_g * filter[0][2] + cv_l_m_g * filter[1][0] +
              cv_m_m_g * filter[1][1] + cv_r_m_g * filter[1][2] +
              cv_l_b_g * filter[2][0] + cv_m_b_g * filter[2][1] +
              cv_r_b_g * filter[2][2];

  int sum_b = cv_l_t_b * filter[0][0] + cv_m_t_b * filter[0][1] +
              cv_r_t_b * filter[0][2] + cv_l_m_b * filter[1][0] +
              cv_m_m_b * filter[1][1] + cv_r_m_b * filter[1][2] +
              cv_l_b_b * filter[2][0] + cv_m_b_b * filter[2][1] +
              cv_r_b_b * filter[2][2];

  *(out_base + 0) = static_cast<unsigned char>(sum_r);
  *(out_base + 1) = static_cast<unsigned char>(sum_g);
  *(out_base + 2) = static_cast<unsigned char>(sum_b);
}

#ifdef __AVX2__

#include <immintrin.h>

inline __m256
load_row(const unsigned char* base)
{
  auto chars = _mm_loadu_si128((__m128i*)base);
  auto ints = _mm256_cvtepu8_epi32(chars);
  return _mm256_cvtepi32_ps(ints);
}

__m256
row_filter(int row)
{
  return _mm256_setr_ps(filter[row][0], // 0, 0
                        filter[row][0], // 0, 1
                        filter[row][0], // 0, 2
                        filter[row][1], // 0, 3
                        filter[row][1], // 1, 0
                        filter[row][1], // 1, 1
                        filter[row][2], // 1, 2
                        filter[row][2]  // 1, 3
  );
}

/**
 * Simd compute a single px.
 */
inline void
smooth_single_px_simd(const unsigned char* top_base,
                      unsigned char* out_base,
                      int row_length,
                      __m256 filter_t,
                      __m256 filter_m,
                      __m256 filter_b)
{
  const unsigned char* mid_base = top_base + row_length;
  const unsigned char* bot_base = mid_base + row_length;

  // load value
  __m256 row_t = load_row(top_base);
  __m256 row_m = load_row(mid_base);
  __m256 row_b = load_row(bot_base);
  float cv_r_t_b = top_base[8];
  float cv_r_m_b = mid_base[8];
  float cv_r_b_b = bot_base[8];

  // apply filter
  row_t = _mm256_mul_ps(row_t, filter_t);
  row_m = _mm256_mul_ps(row_m, filter_m);
  row_b = _mm256_mul_ps(row_b, filter_b);
  cv_r_t_b *= filter[0][2];
  cv_r_m_b *= filter[1][2];
  cv_r_b_b *= filter[2][2];

  // aggregate rows
  __m256 result = _mm256_add_ps(row_t, row_m);
  result = _mm256_add_ps(result, row_b);

  // extract rgb
  __m256i result_int = _mm256_cvtps_epi32(result);
  __m128i result_int0 = _mm256_extractf128_si256(result_int, 0);
  __m128i result_int1 = _mm256_extractf128_si256(result_int, 1);
  int sum_r = _mm_extract_epi32(result_int0, 0) +
              _mm_extract_epi32(result_int0, 3) +
              _mm_extract_epi32(result_int1, 2);
  int sum_g = _mm_extract_epi32(result_int0, 1) +
              _mm_extract_epi32(result_int1, 0) +
              _mm_extract_epi32(result_int1, 3);
  int sum_b = _mm_extract_epi32(result_int0, 2) +
              _mm_extract_epi32(result_int1, 1) + static_cast<char>(cv_r_t_b) +
              static_cast<char>(cv_r_m_b) + static_cast<char>(cv_r_b_b);

  *(out_base + 0) = static_cast<unsigned char>(sum_r);
  *(out_base + 1) = static_cast<unsigned char>(sum_g);
  *(out_base + 2) = static_cast<unsigned char>(sum_b);
}

#endif // AVX2

#endif // PARTB_CORE_H
