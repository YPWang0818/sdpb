#pragma once

#include "sdp_solve/SDP_Solver.hxx"
#include "sdpb_util/Timers/Timers.hxx"

void constraint_matrix_weighted_sum(const Block_Info &block_info,
                                    const SDP &sdp, const Block_Vector &a,
                                    Block_Diagonal_Matrix &Result);

// Timed version: one timer `name` around the sum,
// with accumulators for the per-tile kernels.
void constraint_matrix_weighted_sum(const Block_Info &block_info,
                                    const SDP &sdp, const Block_Vector &a,
                                    Block_Diagonal_Matrix &Result,
                                    Timers &timers, const std::string &name);
