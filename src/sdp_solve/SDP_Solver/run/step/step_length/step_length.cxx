#include "sdp_solve/SDP_Solver.hxx"

// min(gamma \alpha(M, dM), 1), where \alpha(M, dM) denotes the
// largest positive real number such that M + \alpha dM is positive
// semidefinite.
//
// \alpha(M, dM) is computed with a Cholesky decomposition M = L L^T.
// The eigenvalues of M + \alpha dM are equal to the eigenvalues of 1
// + \alpha L^{-1} dM L^{-T}.  The correct \alpha is then -1/lambda,
// where lambda is the smallest eigenvalue of L^{-1} dM L^{-T}.
//
// Inputs:
// - MCholesky = L, the Cholesky decomposition of M (M itself is not needed)
// - dM, a Block_Diagonal_Matrix with the same structure as M
// - which: "X" or "Y", recorded as a timer attribute
// - fast_step_length: lambda in double precision with a BigFloat Cholesky
//   safeguard (fast_step_length.cxx); falls back to the exact eigensolver
//   for the blocks where the safeguard fails
// Workspace:
// - MInvDM (NB: overwritten when computing minEigenvalue)
// - eigenvalues, a Vector of eigenvalues for each block of M
// Output:
// - min(\gamma \alpha(M, dM), 1) (returned)

// A := L^{-1} A L^{-T}
void lower_triangular_inverse_congruence(const Block_Diagonal_Matrix &L,
                                         Block_Diagonal_Matrix &A,
                                         Timers &timers);

El::BigFloat min_eigenvalue(Block_Diagonal_Matrix &A, Timers &timers);

// Step length from double-precision eigenvalues, verified in BigFloat.
// Returns false if any block failed the verification (then nothing is
// decided and the caller must use the exact path).
bool fast_step_length_value(const Block_Diagonal_Matrix &T,
                            const El::BigFloat &gamma, Timers &timers,
                            El::BigFloat &alpha);

El::BigFloat step_length(const Block_Diagonal_Matrix &MCholesky,
                         const Block_Diagonal_Matrix &dM,
                         const El::BigFloat &gamma, const std::string &which,
                         Timers &timers, const bool fast_step_length)
{
  Scoped_Timer step_length_timer(timers, "step_length", {{"which", which}});
  // MInvDM = L^{-1} dM L^{-T}, where M = L L^T
  Block_Diagonal_Matrix MInvDM(dM);
  lower_triangular_inverse_congruence(MCholesky, MInvDM, timers);
  if(fast_step_length)
    {
      El::BigFloat alpha;
      if(fast_step_length_value(MInvDM, gamma, timers, alpha))
        return alpha;
    }
  const El::BigFloat lambda(min_eigenvalue(MInvDM, timers));
  if(lambda > -gamma)
    {
      return 1;
    }
  else
    {
      return -gamma / lambda;
    }
}
