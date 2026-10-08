#include "lower_triangular_solve.hxx"
#include "sdp_solve/SDP_Solver.hxx"
#include "sdp_solve/SDP_Solver/run/timed_linalg.hxx"
#include "sdp_solve/lower_triangular_transpose_solve.hxx"
#include "sdpb_util/copy_matrix.hxx"

// Solve the Schur complement equation for dx, dy.
//
// - As inputs, dx and dy are the residues r_x and r_y on the
//   right-hand side of the Schur complement equation.
// - As outputs, dx and dy are overwritten with the solutions of the
//   Schur complement equation.
//
// The equation is solved using the block-decomposition described in
// the manual.
//
void solve_schur_complement_equation(
  const Block_Diagonal_Matrix &schur_complement_cholesky,
  const Block_Matrix &schur_off_diagonal,
  const El::DistMatrix<El::BigFloat> &Q, Block_Vector &dx, Block_Vector &dy,
  Timers &timers)
{
  Scoped_Timer timer(timers, "solve_schur");
  // dx = schur_complement^{-1}.dx
  lower_triangular_solve(schur_complement_cholesky, dx, timers, "trsm_S");

  El::DistMatrix<El::BigFloat> dy_dist;
  Zeros(dy_dist, Q.Height(), 1);
  {
    El::Matrix<El::BigFloat> dy_sum;
    Zeros(dy_sum, Q.Height(), 1);

    {
      Scoped_Timer gemv_timer(timers, "gemv_P", {{"kind", "gemv"}});
      for(size_t block = 0; block < schur_off_diagonal.blocks.size();
          ++block)
        {
          Gemv(El::OrientationNS::TRANSPOSE, El::BigFloat(-1),
               schur_off_diagonal.blocks[block], dx.blocks[block],
               El::BigFloat(1), dy.blocks[block]);

          // Locally sum contributions to dy
          for(int64_t row = 0; row < dy.blocks[block].LocalHeight(); ++row)
            {
              int64_t global_row(dy.blocks[block].GlobalRow(row));
              for(int64_t column = 0;
                  column < dy.blocks[block].LocalWidth(); ++column)
                {
                  int64_t global_column(dy.blocks[block].GlobalCol(column));
                  dy_sum(global_row, global_column)
                    += dy.blocks[block].GetLocal(row, column);
                }
            }
        }
    }

    // Send out updates for dy
    Scoped_Timer gather_timer(timers, "dy_gather",
                              {{"kind", "mpi"}, {"op", "queue_update"}});
    El::BigFloat zero(0);
    for(int64_t row = 0; row < dy_sum.Height(); ++row)
      for(int64_t column = 0; column < dy_sum.Width(); ++column)
        {
          if(dy_sum(row, column) != zero)
            {
              dy_dist.QueueUpdate(row, column, dy_sum(row, column));
            }
        }
    dy_dist.ProcessQueues();
  }

  // dy_dist = Q^{-1}.dy_dist
  timed::cholesky_SolveAfter(timers, "solve_Q", El::UpperOrLowerNS::UPPER,
                             El::OrientationNS::NORMAL, Q, dy_dist);
  El::DistMatrix<El::BigFloat, El::STAR, El::STAR> dy_local(dy_dist.Grid());
  {
    Scoped_Timer broadcast_timer(timers, "dy_broadcast",
                                 {{"kind", "mpi"}, {"op", "allgather"}});
    El::Copy(dy_dist, dy_local);
  }

  // dx += schur_off_diagonal.dy
  {
    Scoped_Timer gemv_timer(timers, "gemv_P2", {{"kind", "gemv"}});
    for(size_t block = 0; block < schur_off_diagonal.blocks.size(); ++block)
      {
        copy_matrix(dy_local, dy.blocks[block]);
        Gemv(El::OrientationNS::NORMAL, El::BigFloat(1),
             schur_off_diagonal.blocks[block], dy.blocks[block],
             El::BigFloat(1), dx.blocks[block]);
      }
  }

  // dx = schur_complement^{-T}.dx
  lower_triangular_transpose_solve(schur_complement_cholesky, dx, timers,
                                   "trsm_S_T");
}

// Untimed version (used by approx_objective)
void solve_schur_complement_equation(
  const Block_Diagonal_Matrix &schur_complement_cholesky,
  const Block_Matrix &schur_off_diagonal,
  const El::DistMatrix<El::BigFloat> &Q, Block_Vector &dx, Block_Vector &dy)
{
  Timers timers;
  timers.set_detail(0);
  solve_schur_complement_equation(schur_complement_cholesky,
                                  schur_off_diagonal, Q, dx, dy, timers);
}
