#include "sdp_solve/Block_Diagonal_Matrix.hxx"
#include "sdp_solve/Block_Info.hxx"
#include "sdp_solve/SDP_Solver/run/timed_linalg.hxx"

// Compute L (lower triangular) such that A = L L^T
// One timer per block, named `name` (e.g. "X" or "Y"),
// with attributes block, parity, n, ranks.
void cholesky_decomposition(const Block_Diagonal_Matrix &A,
                            Block_Diagonal_Matrix &L,
                            const Block_Info &block_info,
                            const std::string &name, Timers &timers)
{
  for(size_t b = 0; b < A.blocks.size(); b++)
    {
      const auto block_index = block_info.block_indices.at(b / 2);
      const auto parity = b % 2;
      // FIXME: Use pivoting?
      L.blocks[b] = A.blocks[b];
      try
        {
          timed::Cholesky(timers, name, El::UpperOrLowerNS::LOWER, L.blocks[b],
                          {{"block", std::to_string(block_index)},
                           {"parity", std::to_string(parity)}});
        }
      catch(std::exception &e)
        {
          RUNTIME_ERROR("Error when computing Cholesky decomposition of "
                        "Block_Diagonal_Matrix ",
                        name, ", block index = ", block_index,
                        ", parity = ", parity, ": ", e.what());
        }
    }
}

// Untimed version (used by approx_objective)
void cholesky_decomposition(const Block_Diagonal_Matrix &A,
                            Block_Diagonal_Matrix &L,
                            const Block_Info &block_info,
                            const std::string &name)
{
  Timers timers;
  timers.set_detail(0);
  cholesky_decomposition(A, L, block_info, name, timers);
}
