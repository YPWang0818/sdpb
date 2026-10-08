#include "sdp_solve/SDP_Solver/run/constraint_matrix_weighted_sum.hxx"

// Compute the search direction (dx, dX, dy, dY) for the predictor and
// corrector phases.
//
// Inputs:
// - beta, the centering parameter
// - mu = Tr(X Y) / X.cols
// - correctorPhase: boolean indicating whether we're in the corrector
//   phase or predictor phase.
// Workspace (members of SDPSolver which are modified in-place but not
// used elsewhere):
// - Z, R
// Outputs (members of SDPSolver which are modified in-place):
// - dx, dX, dy, dY
//

// C := alpha*A*B + beta*C
void scale_multiply_add(const El::BigFloat &alpha,
                        const Block_Diagonal_Matrix &A,
                        const Block_Diagonal_Matrix &B,
                        const El::BigFloat &beta, Block_Diagonal_Matrix &C,
                        Timers &timers, const std::string &name);

// X := ACholesky^{-T} ACholesky^{-1} X = A^{-1} X
void cholesky_solve(const Block_Diagonal_Matrix &ACholesky,
                    Block_Diagonal_Matrix &X, Timers &timers,
                    const std::string &name);

void compute_schur_RHS(const Block_Info &block_info, const SDP &sdp,
                       const Block_Vector &dual_residues,
                       const Block_Diagonal_Matrix &Z, Block_Vector &dx,
                       Timers &timers);

void solve_schur_complement_equation(
  const Block_Diagonal_Matrix &schur_complement_cholesky,
  const Block_Matrix &schur_off_diagonal,
  const El::DistMatrix<El::BigFloat> &Q, Block_Vector &dx, Block_Vector &dy,
  Timers &timers);

void compute_search_direction(
  const Block_Info &block_info, const SDP &sdp, const SDP_Solver &solver,
  const Block_Diagonal_Matrix &minus_XY,
  const Block_Diagonal_Matrix &schur_complement_cholesky,
  const Block_Matrix &schur_off_diagonal,
  const Block_Diagonal_Matrix &X_cholesky, const El::BigFloat &beta,
  const El::BigFloat &mu, const Block_Vector &primal_residue_p,
  const bool &is_corrector_phase, const El::DistMatrix<El::BigFloat> &Q,
  Block_Vector &dx, Block_Diagonal_Matrix &dX, Block_Vector &dy,
  Block_Diagonal_Matrix &dY, Timers &timers)
{
  // R = beta mu I - X Y (predictor phase)
  // R = beta mu I - X Y - dX dY (corrector phase)
  Block_Diagonal_Matrix R(minus_XY);
  if(is_corrector_phase)
    {
      scale_multiply_add(El::BigFloat(-1), dX, dY, El::BigFloat(1), R, timers,
                         "dXdY_product");
    }
  R.add_diagonal(beta * mu);

  // Z = Symmetrize(X^{-1} (PrimalResidues Y - R))
  Block_Diagonal_Matrix Z(solver.X);
  scale_multiply_add(El::BigFloat(1), solver.primal_residues, solver.Y,
                     El::BigFloat(0), Z, timers, "PY_product");
  Z -= R;
  cholesky_solve(X_cholesky, Z, timers, "cholesky_solve_Z");
  Z.symmetrize();

  // dx[p] = -dual_residues[p] - Tr(A_p Z)
  // dy[n] = dualObjective[n] - (FreeVarMatrix^T x)_n
  compute_schur_RHS(block_info, sdp, solver.dual_residues, Z, dx, timers);
  dy = primal_residue_p;

  // Solve for dx, dy in-place
  solve_schur_complement_equation(schur_complement_cholesky,
                                  schur_off_diagonal, Q, dx, dy, timers);

  // dX = PrimalResidues + \sum_p A_p dx[p]
  constraint_matrix_weighted_sum(block_info, sdp, dx, dX, timers,
                                 "weighted_sum");
  dX += solver.primal_residues;

  // dY = Symmetrize(X^{-1} (R - dX Y))
  scale_multiply_add(El::BigFloat(1), dX, solver.Y, El::BigFloat(0), dY,
                     timers, "dXY_product");
  dY -= R;
  cholesky_solve(X_cholesky, dY, timers, "cholesky_solve_dY");
  dY.symmetrize();
  dY *= El::BigFloat(-1);
}
