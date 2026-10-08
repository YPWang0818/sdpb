#pragma once

#include "Block_Diagonal_Matrix.hxx"
#include "Block_Vector.hxx"
#include "sdpb_util/Timers/Timers.hxx"

// v := L^{-T} v, where L is lower-triangular
void lower_triangular_transpose_solve(const Block_Diagonal_Matrix &L,
                                      Block_Vector &v);

// Timed version: one timer `name` around the loop,
// one "trsm" timer per block with its shape.
void lower_triangular_transpose_solve(const Block_Diagonal_Matrix &L,
                                      Block_Vector &v, Timers &timers,
                                      const std::string &name);
