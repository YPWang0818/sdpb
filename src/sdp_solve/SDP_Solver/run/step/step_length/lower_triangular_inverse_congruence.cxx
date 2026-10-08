#include "sdp_solve/Block_Diagonal_Matrix.hxx"
#include "sdp_solve/SDP_Solver/run/timed_linalg.hxx"

#include <optional>

// A := L^{-1} A L^{-T}
void lower_triangular_inverse_congruence(const Block_Diagonal_Matrix &L,
                                         Block_Diagonal_Matrix &A,
                                         Timers &timers)
{
  Scoped_Timer timer(timers, "congruence");
  for(size_t b = 0; b < A.blocks.size(); b++)
    {
      std::optional<Scoped_Timer> block_timer;
      if(timers.detail() >= 1)
        block_timer.emplace(timers, "block",
                            Timer_Attrs{{"local_block", std::to_string(b / 2)},
                                        {"parity", std::to_string(b % 2)}});
      timed::Trsm(timers, "trsm_right", El::LeftOrRight::RIGHT,
                  El::UpperOrLowerNS::LOWER, El::Orientation::TRANSPOSE,
                  El::UnitOrNonUnit::NON_UNIT, El::BigFloat(1), L.blocks[b],
                  A.blocks[b]);
      timed::Trsm(timers, "trsm_left", El::LeftOrRight::LEFT,
                  El::UpperOrLowerNS::LOWER, El::Orientation::NORMAL,
                  El::UnitOrNonUnit::NON_UNIT, El::BigFloat(1), L.blocks[b],
                  A.blocks[b]);
    }
}
