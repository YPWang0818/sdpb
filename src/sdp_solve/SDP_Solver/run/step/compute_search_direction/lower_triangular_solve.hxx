#pragma once

#include "sdp_solve/Block_Diagonal_Matrix.hxx"
#include "sdp_solve/SDP_Solver/run/timed_linalg.hxx"

// B := L^{-1} B, where L is the result of a previous cholesky
// factorization.  Note that this is different from computing the solution to
// A B == (L L^T) B

template <class T>
void lower_triangular_solve(const Block_Diagonal_Matrix &L_cholesky, T &B)
{
  for(size_t block = 0; block < L_cholesky.blocks.size(); block++)
    {
      local_la::Trsm(El::LeftOrRightNS::LEFT, El::UpperOrLowerNS::LOWER,
                     El::OrientationNS::NORMAL, El::UnitOrNonUnitNS::NON_UNIT,
                     El::BigFloat(1), L_cholesky.blocks[block], B.blocks[block]);
    }
}

// Timed version: one timer `name` around the loop,
// one "trsm" timer per block with its shape.
template <class T>
void lower_triangular_solve(const Block_Diagonal_Matrix &L_cholesky, T &B,
                            Timers &timers, const std::string &name)
{
  Scoped_Timer timer(timers, name);
  for(size_t block = 0; block < L_cholesky.blocks.size(); block++)
    {
      timed::Trsm(timers, "trsm", El::LeftOrRightNS::LEFT,
                  El::UpperOrLowerNS::LOWER, El::OrientationNS::NORMAL,
                  El::UnitOrNonUnitNS::NON_UNIT, El::BigFloat(1),
                  L_cholesky.blocks[block], B.blocks[block],
                  {{"local_block", std::to_string(block)}});
    }
}
