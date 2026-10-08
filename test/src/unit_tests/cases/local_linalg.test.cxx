#include "catch2/catch_amalgamated.hpp"

#include "sdpb_util/local_kernels.hxx"
#include "sdpb_util/local_linalg.hxx"
#include "unit_tests/util/util.hxx"

#include <El.hpp>

// The one-rank fast path (local_la) must agree with Elemental's distributed
// kernels on a grid of one rank. Not bit for bit: the distributed Gemm picks
// a SUMMA variant (A, B, C or Dot) by shape, and each sums in its own order, so
// results may differ in the last bits. The check is that every entry agrees to
// precision - 32 bits relative to the largest entry.

namespace
{
  using Test_Util::random_matrix;
  using Matrix = El::Matrix<El::BigFloat>;
  using DistMatrix = El::DistMatrix<El::BigFloat>;

  DistMatrix to_dist(const Matrix &M, const El::Grid &grid)
  {
    DistMatrix D(M.Height(), M.Width(), grid);
    for(int i = 0; i < M.Height(); ++i)
      for(int j = 0; j < M.Width(); ++j)
        D.SetLocal(i, j, M(i, j));
    return D;
  }

  // Every entry equal up to rounding: |A - B| <= 2^-(precision-32) max|A|
  void require_close(const DistMatrix &A, const DistMatrix &B)
  {
    REQUIRE(A.Height() == B.Height());
    REQUIRE(A.Width() == B.Width());
    El::BigFloat max_abs(1), max_diff(0);
    for(int i = 0; i < A.Height(); ++i)
      for(int j = 0; j < A.Width(); ++j)
        {
          max_abs = std::max(max_abs, El::Abs(A.GetLocal(i, j)));
          max_diff
            = std::max(max_diff, El::Abs(A.GetLocal(i, j) - B.GetLocal(i, j)));
        }
    El::BigFloat tolerance(max_abs);
    tolerance >>= El::gmp::Precision() - 32;
    CAPTURE(A.Height(), A.Width(), max_diff, tolerance);
    CHECK(max_diff <= tolerance);
  }

  // Random lower triangular matrix with a dominant diagonal
  Matrix random_lower(int n)
  {
    Matrix L = random_matrix(n, n);
    El::MakeTrapezoidal(El::LOWER, L);
    for(int i = 0; i < n; ++i)
      L(i, i) = El::BigFloat(2) + El::Abs(L(i, i));
    return L;
  }

  // Random symmetric positive definite matrix
  Matrix random_spd(int n)
  {
    Matrix A = random_matrix(n, n + 3);
    Matrix S(n, n);
    El::Gemm(El::NORMAL, El::TRANSPOSE, El::BigFloat(1), A, A, El::BigFloat(0),
             S);
    for(int i = 0; i < n; ++i)
      S(i, i) += El::BigFloat(n);
    return S;
  }
}

TEST_CASE("local_linalg")
{
  REQUIRE(local_la::enabled());
  El::Grid grid(El::mpi::COMM_SELF);
  const El::BigFloat one(1), minus_one(-1), zero(0), half(0.5);

  SECTION("Gemm")
  {
    const std::vector<std::array<int, 3>> shapes{
      {1, 1, 1},  {3, 3, 3},   {13, 13, 26}, {26, 26, 13}, {5, 40, 13},
      {150, 1, 150}, {1, 30, 24}, {130, 130, 5}, {20, 20, 140}};
    for(const auto orientation_A : {El::NORMAL, El::TRANSPOSE})
      for(const auto orientation_B : {El::NORMAL, El::TRANSPOSE})
        for(const auto &[m, n, k] : shapes)
          for(const auto &alpha : {one, minus_one, half})
            for(const auto &beta : {zero, one, minus_one})
              {
                CAPTURE(orientation_A, orientation_B, m, n, k, alpha, beta);
                const auto A = to_dist(orientation_A == El::NORMAL
                                         ? random_matrix(m, k)
                                         : random_matrix(k, m),
                                       grid);
                const auto B = to_dist(orientation_B == El::NORMAL
                                         ? random_matrix(k, n)
                                         : random_matrix(n, k),
                                       grid);
                const auto C0 = to_dist(random_matrix(m, n), grid);
                auto C_dist = C0;
                auto C_local = C0;
                El::Gemm(orientation_A, orientation_B, alpha, A, B, beta,
                         C_dist);
                local_la::Gemm(orientation_A, orientation_B, alpha, A, B,
                               beta, C_local);
                require_close(C_dist, C_local);
              }
  }

  SECTION("Gemv")
  {
    for(const auto orientation : {El::NORMAL, El::TRANSPOSE})
      for(const auto &[m, n] :
          std::vector<std::array<int, 2>>{{1, 1}, {6, 3}, {26, 13}, {150, 24}})
        for(const auto &alpha : {one, minus_one})
          for(const auto &beta : {zero, one})
            {
              CAPTURE(orientation, m, n, alpha, beta);
              const auto A = to_dist(random_matrix(m, n), grid);
              const int x_size = orientation == El::NORMAL ? n : m;
              const int y_size = orientation == El::NORMAL ? m : n;
              const auto x = to_dist(random_matrix(x_size, 1), grid);
              const auto y0 = to_dist(random_matrix(y_size, 1), grid);
              auto y_dist = y0;
              auto y_local = y0;
              El::Gemv(orientation, alpha, A, x, beta, y_dist);
              local_la::Gemv(orientation, alpha, A, x, beta, y_local);
              require_close(y_dist, y_local);
            }
  }

  SECTION("Syrk")
  {
    for(const auto uplo : {El::LOWER, El::UPPER})
      for(const auto orientation : {El::NORMAL, El::TRANSPOSE})
        for(const auto &[n, k] : std::vector<std::array<int, 2>>{
              {1, 1}, {3, 6}, {13, 26}, {26, 13}, {60, 140}, {140, 30}})
          for(const auto &beta : {zero, one})
            {
              CAPTURE(uplo, orientation, n, k, beta);
              const auto A = to_dist(orientation == El::NORMAL
                                       ? random_matrix(n, k)
                                       : random_matrix(k, n),
                                     grid);
              const auto C0 = to_dist(random_matrix(n, n), grid);
              auto C_dist = C0;
              auto C_local = C0;
              El::Syrk(uplo, orientation, one, A, beta, C_dist);
              local_la::Syrk(uplo, orientation, one, A, beta, C_local);
              require_close(C_dist, C_local);
            }
  }

  SECTION("Trsm")
  {
    for(const auto side : {El::LEFT, El::RIGHT})
      for(const auto orientation : {El::NORMAL, El::TRANSPOSE})
        for(const auto &[n, w] : std::vector<std::array<int, 2>>{
              {1, 1}, {3, 1}, {3, 6}, {13, 5}, {13, 26}, {26, 30}, {140, 7}})
          for(const auto &alpha : {one, minus_one})
            {
              CAPTURE(side, orientation, n, w, alpha);
              const auto L = to_dist(random_lower(n), grid);
              const auto B0 = to_dist(
                side == El::LEFT ? random_matrix(n, w) : random_matrix(w, n),
                grid);
              auto B_dist = B0;
              auto B_local = B0;
              El::Trsm(side, El::LOWER, orientation, El::NON_UNIT, alpha, L,
                       B_dist);
              local_la::Trsm(side, El::LOWER, orientation, El::NON_UNIT,
                             alpha, L, B_local);
              require_close(B_dist, B_local);
            }
  }

  SECTION("Cholesky and SolveAfter")
  {
    for(const auto uplo : {El::LOWER, El::UPPER})
      for(const int n : {1, 3, 13, 60, 128, 129, 160})
        {
          CAPTURE(uplo, n);
          const auto A0 = to_dist(random_spd(n), grid);
          auto A_dist = A0;
          auto A_local = A0;
          El::Cholesky(uplo, A_dist);
          local_la::Cholesky(uplo, A_local);
          require_close(A_dist, A_local);

          for(const int w : {1, 7})
            {
              CAPTURE(w);
              const auto B0 = to_dist(random_matrix(n, w), grid);
              auto B_dist = B0;
              auto B_local = B0;
              El::cholesky::SolveAfter(uplo, El::NORMAL, A_dist, B_dist);
              local_la::cholesky_SolveAfter(uplo, El::NORMAL, A_dist, B_local);
              require_close(B_dist, B_local);
            }
        }
  }
}

// local_kernels must reproduce Elemental's generic BigFloat loops (the
// El::Matrix overloads) bit for bit, also for inputs that carry the extra
// limb that sums and quotients have and products truncate away.
namespace
{
  // Entries x + x/3 with x uniform in [-1, 1]: about a quarter of them use all
  // _mp_prec + 1 limbs.
  Matrix full_limb_matrix(int height, int width)
  {
    Matrix M = random_matrix(height, width);
    const El::BigFloat three(3);
    for(int i = 0; i < height; ++i)
      for(int j = 0; j < width; ++j)
        M(i, j) += M(i, j) / three;
    return M;
  }

  Matrix full_limb_lower(int n, bool upper)
  {
    Matrix L = full_limb_matrix(n, n);
    El::MakeTrapezoidal(upper ? El::UPPER : El::LOWER, L);
    for(int i = 0; i < n; ++i)
      L(i, i) = El::BigFloat(2) + El::Abs(L(i, i)) / El::BigFloat(7);
    return L;
  }

  void require_bitwise(const Matrix &A, const Matrix &B)
  {
    REQUIRE(A.Height() == B.Height());
    REQUIRE(A.Width() == B.Width());
    size_t num_different = 0;
    for(int i = 0; i < A.Height(); ++i)
      for(int j = 0; j < A.Width(); ++j)
        {
          const mpf_srcptr a = A(i, j).gmp_float.get_mpf_t();
          const mpf_srcptr b = B(i, j).gmp_float.get_mpf_t();
          const mp_size_t size = std::abs(a->_mp_size);
          bool same = a->_mp_size == b->_mp_size
                      && (size == 0 || a->_mp_exp == b->_mp_exp);
          for(mp_size_t l = 0; same && l < size; ++l)
            same = a->_mp_d[l] == b->_mp_d[l];
          if(!same)
            ++num_different;
        }
    CAPTURE(A.Height(), A.Width());
    CHECK(num_different == 0);
  }
}

TEST_CASE("local_kernels")
{
  const El::BigFloat one(1), minus_one(-1), zero(0), half(0.5);
  const El::BigFloat third = El::BigFloat(1) / El::BigFloat(3);

  SECTION("test inputs carry the extra limb")
  {
    const Matrix M = full_limb_matrix(10, 10);
    int num_full = 0;
    for(int i = 0; i < 10; ++i)
      for(int j = 0; j < 10; ++j)
        {
          const mpf_srcptr x = M(i, j).gmp_float.get_mpf_t();
          if(std::abs(x->_mp_size) > x->_mp_prec)
            ++num_full;
        }
    CHECK(num_full > 10);
  }

  SECTION("gemm")
  {
    const std::vector<std::array<int, 3>> shapes{
      {1, 1, 1}, {3, 3, 3}, {13, 13, 26}, {26, 26, 13}, {5, 40, 13}, {7, 1, 30}, {4, 6, 0}};
    for(const auto orientation_A : {El::NORMAL, El::TRANSPOSE})
      for(const auto orientation_B : {El::NORMAL, El::TRANSPOSE})
        for(const auto &[m, n, k] : shapes)
          for(const auto &alpha : {one, minus_one, half, third})
            for(const auto &beta : {zero, one, minus_one, third})
              {
                CAPTURE(orientation_A, orientation_B, m, n, k, alpha, beta);
                const Matrix A = orientation_A == El::NORMAL
                                   ? full_limb_matrix(m, k)
                                   : full_limb_matrix(k, m);
                const Matrix B = orientation_B == El::NORMAL
                                   ? full_limb_matrix(k, n)
                                   : full_limb_matrix(n, k);
                const Matrix C0 = full_limb_matrix(m, n);
                Matrix C_el = C0, C_own = C0;
                El::Gemm(orientation_A, orientation_B, alpha, A, B, beta,
                         C_el);
                local_kernels::gemm(orientation_A, orientation_B, alpha, A,
                                    B, beta, C_own);
                require_bitwise(C_el, C_own);
              }
  }

  SECTION("syrk")
  {
    for(const auto uplo : {El::LOWER, El::UPPER})
      for(const auto orientation : {El::NORMAL, El::TRANSPOSE})
        for(const auto &[n, k] : std::vector<std::array<int, 2>>{
              {1, 1}, {3, 6}, {13, 26}, {26, 13}})
          for(const auto &alpha : {one, minus_one, third})
            for(const auto &beta : {zero, one, third})
              {
                CAPTURE(uplo, orientation, n, k, alpha, beta);
                const Matrix A = orientation == El::NORMAL
                                   ? full_limb_matrix(n, k)
                                   : full_limb_matrix(k, n);
                const Matrix C0 = full_limb_matrix(n, n);
                Matrix C_el = C0, C_own = C0;
                El::Syrk(uplo, orientation, alpha, A, beta, C_el);
                local_kernels::syrk(uplo, orientation, alpha, A, beta, C_own);
                require_bitwise(C_el, C_own);
              }
  }

  SECTION("trsm")
  {
    for(const auto side : {El::LEFT, El::RIGHT})
      for(const auto uplo : {El::LOWER, El::UPPER})
        for(const auto orientation : {El::NORMAL, El::TRANSPOSE})
          for(const auto diag : {El::NON_UNIT, El::UNIT})
            for(const auto &[n, w] : std::vector<std::array<int, 2>>{
                  {1, 1}, {3, 1}, {3, 6}, {13, 5}, {13, 26}})
              for(const auto &alpha : {one, minus_one, third})
                {
                  CAPTURE(side, uplo, orientation, diag, n, w, alpha);
                  const Matrix A = full_limb_lower(n, uplo == El::UPPER);
                  const Matrix B0 = side == El::LEFT ? full_limb_matrix(n, w)
                                                     : full_limb_matrix(w, n);
                  Matrix B_el = B0, B_own = B0, B_reciprocal = B0;
                  El::Trsm(side, uplo, orientation, diag, alpha, A, B_el);
                  local_kernels::trsm(side, uplo, orientation, diag, alpha, A,
                                      B_own, false);
                  require_bitwise(B_el, B_own);

                  // reciprocal: equal up to rounding
                  local_kernels::trsm(side, uplo, orientation, diag, alpha, A,
                                      B_reciprocal, true);
                  El::BigFloat max_abs(1), max_diff(0);
                  for(int i = 0; i < B_el.Height(); ++i)
                    for(int j = 0; j < B_el.Width(); ++j)
                      {
                        max_abs = std::max(max_abs, El::Abs(B_el(i, j)));
                        max_diff = std::max(
                          max_diff, El::Abs(B_el(i, j) - B_reciprocal(i, j)));
                      }
                  El::BigFloat tolerance(max_abs);
                  tolerance >>= El::gmp::Precision() - 32;
                  CHECK(max_diff <= tolerance);
                }
  }

  SECTION("cholesky SolveAfter through local_la")
  {
    El::Grid grid(El::mpi::COMM_SELF);
    for(const auto uplo : {El::LOWER, El::UPPER})
      for(const int n : {1, 3, 13, 60})
        for(const int w : {1, 7})
          {
            CAPTURE(uplo, n, w);
            Matrix S = random_spd(n);
            El::Cholesky(uplo, S);
            const Matrix B0 = full_limb_matrix(n, w);
            Matrix B_el = B0;
            El::cholesky::SolveAfter(uplo, El::NORMAL, S, B_el);
            const auto S_dist = to_dist(S, grid);
            auto B_own = to_dist(B0, grid);
            local_la::cholesky_SolveAfter(uplo, El::NORMAL, S_dist, B_own);
            require_bitwise(B_el, B_own.LockedMatrix());
          }
  }
}
