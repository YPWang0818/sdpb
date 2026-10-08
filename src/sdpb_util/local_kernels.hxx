#pragma once

#include <El.hpp>

// BigFloat kernels on local matrices for the one-rank fast path
// (sdpb_util/local_linalg.hxx).
//
// They run Elemental's generic blas::Gemm/Syrk/Trsm loops
// (El/src/core/imports/blas/*.hpp) operation for operation, so the results
// are bit-identical, but call GMP directly: no BigFloat temporaries (each
// costs a malloc), no copy before every multiplication, and a cheap limb
// truncation instead of Trsm's "B *= alpha" pass when alpha == 1 (mpf_mul
// truncates its inputs to the target precision, so multiplying by one is
// not a no-op for values that carry the extra limb).
//
// Trsm can optionally multiply by the reciprocal of each diagonal entry
// instead of dividing every entry of B by it: m*n divisions become n
// reciprocals and m*n multiplications. That changes the last bits of the
// result, so it is off unless requested (see local_la::trsm_reciprocal()).
namespace local_kernels
{
  using Matrix = El::Matrix<El::BigFloat>;

  // C := alpha op(A) op(B) + beta C
  void gemm(El::Orientation orientation_A, El::Orientation orientation_B,
            const El::BigFloat &alpha, const Matrix &A, const Matrix &B,
            const El::BigFloat &beta, Matrix &C);

  // uplo triangle of C := alpha op(A) op(A)^T + beta C (C is first scaled
  // by beta on all of C, as the local El::Syrk does; local_la::Syrk scales
  // only the uplo triangle beforehand and passes beta = 1)
  void syrk(El::UpperOrLower uplo, El::Orientation orientation,
            const El::BigFloat &alpha, const Matrix &A,
            const El::BigFloat &beta, Matrix &C);

  // B := alpha op(A)^{-1} B (side LEFT) or alpha B op(A)^{-1} (side RIGHT),
  // A triangular. op is NORMAL or TRANSPOSE (no ADJOINT: BigFloat is real).
  void trsm(El::LeftOrRight side, El::UpperOrLower uplo,
            El::Orientation orientation, El::UnitOrNonUnit diag,
            const El::BigFloat &alpha, const Matrix &A, Matrix &B,
            bool reciprocal);
}
