#include "sdp_solve/Block_Diagonal_Matrix.hxx"
#include "sdp_solve/SDP_Solver/run/timed_linalg.hxx"

// C := alpha*A*B + beta*C
void scale_multiply_add(const El::BigFloat &alpha,
                        const Block_Diagonal_Matrix &A,
                        const Block_Diagonal_Matrix &B,
                        const El::BigFloat &beta, Block_Diagonal_Matrix &C,
                        Timers &timers, const std::string &name)
{
  Scoped_Timer timer(timers, name);
  for(size_t block = 0; block < A.blocks.size(); ++block)
    {
      timed::Gemm(timers, "gemm", El::OrientationNS::NORMAL,
                  El::OrientationNS::NORMAL, alpha, A.blocks[block],
                  B.blocks[block], beta, C.blocks[block],
                  {{"local_block", std::to_string(block / 2)},
                   {"parity", std::to_string(block % 2)}});
    }
}

// Untimed version
void scale_multiply_add(const El::BigFloat &alpha,
                        const Block_Diagonal_Matrix &A,
                        const Block_Diagonal_Matrix &B,
                        const El::BigFloat &beta, Block_Diagonal_Matrix &C)
{
  for(size_t block = 0; block < A.blocks.size(); ++block)
    {
      local_la::Gemm(El::OrientationNS::NORMAL, El::OrientationNS::NORMAL, alpha,
                     A.blocks[block], B.blocks[block], beta, C.blocks[block]);
    }
}
