#include "lower_triangular_transpose_solve.hxx"
#include "sdp_solve/SDP_Solver/run/timed_linalg.hxx"

// v := L^{-T} v, where L is lower-triangular
void lower_triangular_transpose_solve(const Block_Diagonal_Matrix &L,
                                      Block_Vector &v)
{
  for(size_t b = 0; b < L.blocks.size(); b++)
    {
      local_la::Trsm(El::LeftOrRight::LEFT, El::UpperOrLowerNS::LOWER,
                     El::OrientationNS::TRANSPOSE, El::UnitOrNonUnit::NON_UNIT,
                     El::BigFloat(1), L.blocks[b], v.blocks[b]);
    }
}

void lower_triangular_transpose_solve(const Block_Diagonal_Matrix &L,
                                      Block_Vector &v, Timers &timers,
                                      const std::string &name)
{
  Scoped_Timer timer(timers, name);
  for(size_t b = 0; b < L.blocks.size(); b++)
    {
      timed::Trsm(timers, "trsm", El::LeftOrRight::LEFT,
                  El::UpperOrLowerNS::LOWER, El::OrientationNS::TRANSPOSE,
                  El::UnitOrNonUnit::NON_UNIT, El::BigFloat(1), L.blocks[b],
                  v.blocks[b], {{"local_block", std::to_string(b)}});
    }
}
