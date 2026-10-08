#pragma once

#include "sdpb_util/Timers/Timers.hxx"

#include <El.hpp>

#include <string>
#include <utility>

// Thin wrappers around Elemental's dense kernels.
// Each wrapper runs the kernel inside a Scoped_Timer and records the kernel
// kind and the matrix shapes as timer attributes, so that the profile knows
// (m, n, k, ranks) for every BigFloat matrix product / factorization
// of the solver. They are also the single dispatch point for a future
// GPU implementation of these kernels.
//
// With Timers::detail() == 0 the wrappers call Elemental directly
// and record nothing (legacy-sized profiles).
//
// Attribute conventions:
//   kind  : gemm | syrk | trsm | cholesky | cholesky_solve | eig
//   m, n  : shape of the output (C or B), n only for square inputs
//   k     : inner dimension of a product
//   ranks : number of MPI ranks sharing the (distributed) matrix
namespace timed
{
  template <class T> int ranks_of(const El::AbstractDistMatrix<T> &A)
  {
    return A.Grid().Size();
  }
  template <class T> int ranks_of(const El::Matrix<T> &)
  {
    return 1;
  }

  // Append (key, value) for every value >= 0
  inline Timer_Attrs
  shape_attrs(const char *kind, const int64_t m, const int64_t n,
              const int64_t k, const int ranks, Timer_Attrs extra)
  {
    Timer_Attrs attrs;
    attrs.reserve(extra.size() + 5);
    attrs.emplace_back("kind", kind);
    if(m >= 0)
      attrs.emplace_back("m", std::to_string(m));
    if(n >= 0)
      attrs.emplace_back("n", std::to_string(n));
    if(k >= 0)
      attrs.emplace_back("k", std::to_string(k));
    attrs.emplace_back("ranks", std::to_string(ranks));
    for(auto &kv : extra)
      attrs.emplace_back(std::move(kv));
    return attrs;
  }

  // C := alpha op(A) op(B) + beta C
  template <class TScalar, class TA, class TB, class TC>
  void Gemm(Timers &timers, const std::string &name,
            const El::Orientation orientation_A,
            const El::Orientation orientation_B, const TScalar &alpha,
            const TA &A, const TB &B, const TScalar &beta, TC &C,
            Timer_Attrs extra = {})
  {
    if(timers.detail() <= 0)
      {
        El::Gemm(orientation_A, orientation_B, alpha, A, B, beta, C);
        return;
      }
    const int64_t k
      = orientation_A == El::Orientation::NORMAL ? A.Width() : A.Height();
    Scoped_Timer timer(timers, name,
                       shape_attrs("gemm", C.Height(), C.Width(), k,
                                   ranks_of(C), std::move(extra)));
    El::Gemm(orientation_A, orientation_B, alpha, A, B, beta, C);
  }

  // C := alpha op(A) op(A)^T + beta C
  template <class TScalar, class TA, class TC>
  void Syrk(Timers &timers, const std::string &name,
            const El::UpperOrLower uplo, const El::Orientation orientation,
            const TScalar &alpha, const TA &A, const TScalar &beta, TC &C,
            Timer_Attrs extra = {})
  {
    if(timers.detail() <= 0)
      {
        El::Syrk(uplo, orientation, alpha, A, beta, C);
        return;
      }
    const int64_t k
      = orientation == El::Orientation::NORMAL ? A.Width() : A.Height();
    Scoped_Timer timer(timers, name,
                       shape_attrs("syrk", -1, C.Height(), k, ranks_of(C),
                                   std::move(extra)));
    El::Syrk(uplo, orientation, alpha, A, beta, C);
  }

  // B := alpha op(A)^{-1} B  (or B op(A)^{-1}), A triangular
  template <class TScalar, class TA, class TB>
  void Trsm(Timers &timers, const std::string &name, const El::LeftOrRight side,
            const El::UpperOrLower uplo, const El::Orientation orientation,
            const El::UnitOrNonUnit diag, const TScalar &alpha, const TA &A,
            TB &B, Timer_Attrs extra = {})
  {
    if(timers.detail() <= 0)
      {
        El::Trsm(side, uplo, orientation, diag, alpha, A, B);
        return;
      }
    Scoped_Timer timer(timers, name,
                       shape_attrs("trsm", B.Height(), B.Width(), -1,
                                   ranks_of(B), std::move(extra)));
    El::Trsm(side, uplo, orientation, diag, alpha, A, B);
  }

  // A := Cholesky factor of A
  template <class TA>
  void Cholesky(Timers &timers, const std::string &name,
                const El::UpperOrLower uplo, TA &A, Timer_Attrs extra = {})
  {
    if(timers.detail() <= 0)
      {
        El::Cholesky(uplo, A);
        return;
      }
    Scoped_Timer timer(timers, name,
                       shape_attrs("cholesky", -1, A.Height(), -1,
                                   ranks_of(A), std::move(extra)));
    El::Cholesky(uplo, A);
  }

  // B := A^{-1} B, where A = L L^T was factored by Cholesky
  template <class TA, class TB>
  void cholesky_SolveAfter(Timers &timers, const std::string &name,
                           const El::UpperOrLower uplo,
                           const El::Orientation orientation, const TA &A,
                           TB &B, Timer_Attrs extra = {})
  {
    if(timers.detail() <= 0)
      {
        El::cholesky::SolveAfter(uplo, orientation, A, B);
        return;
      }
    Scoped_Timer timer(timers, name,
                       shape_attrs("cholesky_solve", B.Height(), B.Width(),
                                   -1, ranks_of(B), std::move(extra)));
    El::cholesky::SolveAfter(uplo, orientation, A, B);
  }

  // w := eigenvalues of Hermitian A (A is overwritten)
  template <class TA, class TW, class TCtrl>
  void HermitianEig(Timers &timers, const std::string &name,
                    const El::UpperOrLower uplo, TA &A, TW &w,
                    const TCtrl &ctrl, Timer_Attrs extra = {})
  {
    if(timers.detail() <= 0)
      {
        El::HermitianEig(uplo, A, w, ctrl);
        return;
      }
    Scoped_Timer timer(timers, name,
                       shape_attrs("eig", -1, A.Height(), -1, ranks_of(A),
                                   std::move(extra)));
    El::HermitianEig(uplo, A, w, ctrl);
  }
}
