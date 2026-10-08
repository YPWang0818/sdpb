#pragma once

#include <El.hpp>

#include <cstdlib>
#include <cstring>

// One-rank fast path for Elemental's dense kernels.
//
// A DistMatrix on a grid of one rank holds the whole matrix locally, but
// Elemental's distributed kernels still go through their redistribution
// machinery (SUMMA panels, [MC,*]/[*,MR] copies, LLN/Trrk temporaries) around
// the same local BLAS-like loops. For the small BigFloat blocks of many-tiny-SDP
// workloads those copies cost as much as the arithmetic.
//
// The functions below take the same arguments as the El:: kernels. When every
// operand lives entirely on one rank they call the El::Matrix overloads on the
// local matrices; otherwise they call the distributed El:: kernel unchanged.
// The local kernels run the same loops in the same order, so the results are
// bit-identical (test/src/unit_tests/cases/local_linalg.test.cxx).
//
// SDPB_LOCAL_KERNELS=0 in the environment disables the fast path.
namespace local_la
{
  inline bool enabled()
  {
    static const bool value = [] {
      const char *env = std::getenv("SDPB_LOCAL_KERNELS");
      return env == nullptr || std::strcmp(env, "0") != 0;
    }();
    return value;
  }

  template <class T> bool is_local(const El::AbstractDistMatrix<T> &A)
  {
    return A.Grid().Size() == 1 && A.ColAlign() == 0 && A.RowAlign() == 0
           && A.LocalHeight() == A.Height() && A.LocalWidth() == A.Width();
  }
  template <class T> bool is_local(const El::Matrix<T> &)
  {
    return true;
  }

  template <class T> const El::Matrix<T> &local(const El::AbstractDistMatrix<T> &A)
  {
    return A.LockedMatrix();
  }
  template <class T> El::Matrix<T> &local(El::AbstractDistMatrix<T> &A)
  {
    return A.Matrix();
  }
  template <class T> const El::Matrix<T> &local(const El::Matrix<T> &A)
  {
    return A;
  }
  template <class T> El::Matrix<T> &local(El::Matrix<T> &A)
  {
    return A;
  }

  template <class... Ms> bool all_local(const Ms &...ms)
  {
    return enabled() && (is_local(ms) && ...);
  }

  // C := alpha op(A) op(B) + beta C
  template <class TScalar, class TA, class TB, class TC>
  void Gemm(const El::Orientation orientation_A,
            const El::Orientation orientation_B, const TScalar &alpha,
            const TA &A, const TB &B, const TScalar &beta, TC &C)
  {
    if(all_local(A, B, C))
      El::Gemm(orientation_A, orientation_B, alpha, local(A), local(B), beta,
               local(C));
    else
      El::Gemm(orientation_A, orientation_B, alpha, A, B, beta, C);
  }

  // y := alpha op(A) x + beta y
  template <class TScalar, class TA, class TX, class TY>
  void Gemv(const El::Orientation orientation, const TScalar &alpha,
            const TA &A, const TX &x, const TScalar &beta, TY &y)
  {
    if(all_local(A, x, y))
      El::Gemv(orientation, alpha, local(A), local(x), beta, local(y));
    else
      El::Gemv(orientation, alpha, A, x, beta, y);
  }

  // C := alpha op(A) op(A)^T + beta C (triangle uplo of C)
  template <class TScalar, class TA, class TC>
  void Syrk(const El::UpperOrLower uplo, const El::Orientation orientation,
            const TScalar &alpha, const TA &A, const TScalar &beta, TC &C)
  {
    if(all_local(A, C))
      {
        // As the distributed Syrk: scale only the uplo triangle of C (the
        // local blas::Syrk would scale, or zero, all of C).
        auto &C_local = local(C);
        if(beta != TScalar(1))
          El::ScaleTrapezoid(beta, uplo, C_local);
        El::Syrk(uplo, orientation, alpha, local(A), TScalar(1), C_local);
      }
    else
      El::Syrk(uplo, orientation, alpha, A, beta, C);
  }

  // B := alpha op(A)^{-1} B  (or B op(A)^{-1}), A triangular
  template <class TScalar, class TA, class TB>
  void Trsm(const El::LeftOrRight side, const El::UpperOrLower uplo,
            const El::Orientation orientation, const El::UnitOrNonUnit diag,
            const TScalar &alpha, const TA &A, TB &B)
  {
    if(all_local(A, B))
      El::Trsm(side, uplo, orientation, diag, alpha, local(A), local(B));
    else
      El::Trsm(side, uplo, orientation, diag, alpha, A, B);
  }

  // A := Cholesky factor of A
  template <class TA> void Cholesky(const El::UpperOrLower uplo, TA &A)
  {
    if(all_local(A))
      El::Cholesky(uplo, local(A));
    else
      El::Cholesky(uplo, A);
  }

  // B := A^{-1} B, where A = L L^T was factored by Cholesky
  template <class TA, class TB>
  void cholesky_SolveAfter(const El::UpperOrLower uplo,
                           const El::Orientation orientation, const TA &A,
                           TB &B)
  {
    if(all_local(A, B))
      El::cholesky::SolveAfter(uplo, orientation, local(A), local(B));
    else
      El::cholesky::SolveAfter(uplo, orientation, A, B);
  }
}
