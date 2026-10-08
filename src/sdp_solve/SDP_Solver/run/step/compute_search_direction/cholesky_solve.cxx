#include "sdp_solve/Block_Diagonal_Matrix.hxx"
#include "sdp_solve/SDP_Solver/run/timed_linalg.hxx"

// X := ACholesky^{-T} ACholesky^{-1} X = A^{-1} X
void cholesky_solve(const Block_Diagonal_Matrix &ACholesky,
                    Block_Diagonal_Matrix &X, Timers &timers,
                    const std::string &name)
{
  Scoped_Timer timer(timers, name);
  for(size_t b = 0; b < X.blocks.size(); b++)
    {
      timed::cholesky_SolveAfter(timers, "cholesky_solve",
                                 El::UpperOrLowerNS::LOWER,
                                 El::OrientationNS::NORMAL, ACholesky.blocks[b],
                                 X.blocks[b],
                                 {{"local_block", std::to_string(b / 2)},
                                  {"parity", std::to_string(b % 2)}});
    }
}
