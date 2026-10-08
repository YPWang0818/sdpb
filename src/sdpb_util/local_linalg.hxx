#pragma once

#include "local_kernels.hxx"

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
// operand lives entirely on one rank they run local kernels on the local
// matrices; otherwise they call the distributed El:: kernel unchanged.
// Gemm, Syrk, Trsm and cholesky_SolveAfter use local_kernels (Elemental's
// generic loops without temporaries, bit-identical to the El::Matrix
// overloads); Gemv and Cholesky use the El::Matrix overloads.
// (test/src/unit_tests/cases/local_linalg.test.cxx)
//
// Environment, read once:
//   SDPB_LOCAL_KERNELS=0    no fast path (distributed kernels everywhere)
//   SDPB_LOCAL_KERNELS=el   the El::Matrix overloads instead of local_kernels
//   SDPB_TRSM_RECIPROCAL=1  Trsm multiplies by reciprocals of the diagonal
//                           instead of dividing (changes the last bits)
namespace local_la
{
  inline const char *env_or_empty(const char *name)
  {
    const char *value = std::getenv(name);
    return value == nullptr ? "" : value;
  }

  inline bool enabled()
  {
    static const bool value
      = std::strcmp(env_or_empty("SDPB_LOCAL_KERNELS"), "0") != 0;
    return value;
  }

  // Use local_kernels (true) or the El::Matrix overloads (false)
  inline bool own_kernels()
  {
    static const bool value
      = std::strcmp(env_or_empty("SDPB_LOCAL_KERNELS"), "el") != 0;
    return value;
  }

  inline bool trsm_reciprocal()
  {
    static const bool value
      = std::strcmp(env_or_empty("SDPB_TRSM_RECIPROCAL"), "1") == 0;
    return value;
  }

  inline bool real_orientation(const El::Orientation orientation)
  {
    return orientation == El::NORMAL || orientation == El::TRANSPOSE;
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
      {
        if(own_kernels() && real_orientation(orientation_A)
           && real_orientation(orientation_B))
          local_kernels::gemm(orientation_A, orientation_B, alpha, local(A),
                              local(B), beta, local(C));
        else
          El::Gemm(orientation_A, orientation_B, alpha, local(A), local(B),
                   beta, local(C));
      }
    else
      El::Gemm(orientation_A, orientation_B, alpha, A, B, beta, C);
  }

  // The uplo triangle (with the diagonal) of C := alpha op(A) op(B) + beta C,
  // for products that are symmetrized afterwards; the other triangle is
  // unspecified (the distributed fallback computes all of C).
  template <class TScalar, class TA, class TB, class TC>
  void Gemm_triangle(const El::UpperOrLower uplo,
                     const El::Orientation orientation_A,
                     const El::Orientation orientation_B, const TScalar &alpha,
                     const TA &A, const TB &B, const TScalar &beta, TC &C)
  {
    if(all_local(A, B, C) && own_kernels() && real_orientation(orientation_A)
       && real_orientation(orientation_B))
      local_kernels::gemm_triangle(uplo, orientation_A, orientation_B, alpha,
                                   local(A), local(B), beta, local(C));
    else
      Gemm(orientation_A, orientation_B, alpha, A, B, beta, C);
  }

  // El::MakeSymmetric(uplo, A) for a distributed A: the other triangle :=
  // 1 * this triangle (a truncated copy; see local_kernels::make_symmetric).
  // Only for DistMatrix: El::MakeSymmetric on an El::Matrix copies exactly.
  template <class T>
  void MakeSymmetric(const El::UpperOrLower uplo, El::ElementalMatrix<T> &A)
  {
    if(all_local(A) && own_kernels())
      local_kernels::make_symmetric(uplo, local(A));
    else
      El::MakeSymmetric(uplo, A);
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
        if(own_kernels() && real_orientation(orientation))
          local_kernels::syrk(uplo, orientation, alpha, local(A), TScalar(1),
                              C_local);
        else
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
      {
        if(own_kernels() && real_orientation(orientation))
          local_kernels::trsm(side, uplo, orientation, diag, alpha, local(A),
                              local(B), trsm_reciprocal());
        else
          El::Trsm(side, uplo, orientation, diag, alpha, local(A), local(B));
      }
    else
      El::Trsm(side, uplo, orientation, diag, alpha, A, B);
  }

  // Y := Y + alpha X
  template <class TScalar, class TX, class TY>
  void Axpy(const TScalar &alpha, const TX &X, TY &Y)
  {
    if(all_local(X, Y) && own_kernels() && X.Height() == Y.Height()
       && X.Width() == Y.Width())
      local_kernels::axpy(alpha, local(X), local(Y));
    else
      El::Axpy(alpha, X, Y);
  }

  // A := (A + A^T) / 2 with the roundings of "A *= 0.5; A += Transpose(A)"
  template <class TA> void symmetrize(TA &A)
  {
    if(all_local(A) && own_kernels())
      {
        local_kernels::symmetrize(local(A));
        return;
      }
    A *= El::BigFloat(0.5);
    El::DistMatrix<El::BigFloat> transpose(A.Grid());
    El::Transpose(A, transpose, false);
    A += transpose;
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
      {
        if(own_kernels() && orientation == El::NORMAL)
          {
            // As El::cholesky::SolveAfter (whose second Trsm is ADJOINT,
            // the same as TRANSPOSE for real numbers)
            const El::BigFloat one(1);
            const auto &L = local(A);
            auto &X = local(B);
            const bool reciprocal = trsm_reciprocal();
            if(uplo == El::LOWER)
              {
                local_kernels::trsm(El::LEFT, El::LOWER, El::NORMAL,
                                    El::NON_UNIT, one, L, X, reciprocal);
                local_kernels::trsm(El::LEFT, El::LOWER, El::TRANSPOSE,
                                    El::NON_UNIT, one, L, X, reciprocal);
              }
            else
              {
                local_kernels::trsm(El::LEFT, El::UPPER, El::TRANSPOSE,
                                    El::NON_UNIT, one, L, X, reciprocal);
                local_kernels::trsm(El::LEFT, El::UPPER, El::NORMAL,
                                    El::NON_UNIT, one, L, X, reciprocal);
              }
          }
        else
          El::cholesky::SolveAfter(uplo, orientation, local(A), local(B));
      }
    else
      El::cholesky::SolveAfter(uplo, orientation, A, B);
  }
}
