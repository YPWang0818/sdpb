#pragma once

#include "sdp_solve/Block_Info.hxx"
#include "sdp_solve/SDP.hxx"
#include "sdp_solve/SDP_Solver.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/BigInt_Shared_Memory_Syrk_Context.hxx"
#include "sdp_solve/memory_estimates.hxx"
#include "sdpb_util/Environment.hxx"
#include "sdpb_util/Timers/Timers.hxx"

#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <numeric>
#include <string>

// Helpers to record static run information in the per-rank profile
// (Timers::set_meta), so that the profile can be interpreted on its own:
// block shapes, which blocks this rank owns, matrix sizes,
// bigint_syrk parameters.

namespace profile_meta
{
  template <class TFill> std::string to_json(TFill &&fill)
  {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    fill(writer);
    return std::string(buffer.GetString(), buffer.GetSize());
  }

  inline std::string json_string(const std::string &value)
  {
    return to_json(
      [&](auto &writer) { writer.String(value.c_str(), value.size()); });
  }

  inline std::string block_timings_to_json(const El::Matrix<int32_t> &timings)
  {
    return to_json([&](auto &writer) {
      writer.StartArray();
      for(int64_t row = 0; row < timings.Height(); ++row)
        writer.Int(timings(row, 0));
      writer.EndArray();
    });
  }

  inline void
  set_problem_info(Timers &timers, const Environment &env,
                   const Block_Info &block_info, const SDP &sdp,
                   const SDP_Solver &solver,
                   const BigInt_Shared_Memory_Syrk_Context &bigint_syrk_context)
  {
    const size_t num_blocks = block_info.dimensions.size();

    // Shapes of all blocks of the problem
    timers.set_meta("blocks", to_json([&](auto &writer) {
      writer.StartArray();
      for(size_t index = 0; index < num_blocks; ++index)
        {
          writer.StartObject();
          writer.Key("index");
          writer.Uint64(index);
          writer.Key("dim");
          writer.Uint64(block_info.dimensions.at(index));
          writer.Key("num_points");
          writer.Uint64(block_info.num_points.at(index));
          writer.Key("schur_size");
          writer.Uint64(block_info.get_schur_block_size(index));
          writer.Key("psd_sizes");
          writer.StartArray();
          writer.Uint64(block_info.get_psd_matrix_block_size(index, 0));
          writer.Uint64(block_info.get_psd_matrix_block_size(index, 1));
          writer.EndArray();
          writer.Key("bilinear_pairing_size");
          writer.Uint64(block_info.get_bilinear_pairing_block_size(index, 0));
          writer.EndObject();
        }
      writer.EndArray();
    }));

    // Blocks owned by this rank (global indices), in local order
    timers.set_meta("local_blocks", to_json([&](auto &writer) {
      writer.StartArray();
      for(const auto &index : block_info.block_indices)
        writer.Uint64(index);
      writer.EndArray();
    }));
    timers.set_meta("group_size",
                    std::to_string(block_info.mpi_comm.value.Size()));
    timers.set_meta("group_rank",
                    std::to_string(block_info.mpi_comm.value.Rank()));
    timers.set_meta("procs_per_node",
                    std::to_string(env.comm_shared_mem.Size()));

    // Global sizes
    const auto schur_sizes = block_info.schur_block_sizes();
    const auto psd_sizes = block_info.psd_matrix_block_sizes();
    timers.set_meta("N", std::to_string(sdp.dual_objective_b.Height()));
    timers.set_meta("P", std::to_string(std::accumulate(
                           schur_sizes.begin(), schur_sizes.end(), size_t(0))));
    timers.set_meta("total_psd_rows",
                    std::to_string(std::accumulate(
                      psd_sizes.begin(), psd_sizes.end(), size_t(0))));
    timers.set_meta("bigfloat_bytes", std::to_string(bigfloat_bytes()));

    // Number of BigFloats allocated locally by the main matrices
    // (same quantities as the debug memory estimate in run.cxx)
    timers.set_meta("local_sizes", to_json([&](auto &writer) {
      writer.StartObject();
      writer.Key("X");
      writer.Uint64(get_matrix_size_local(solver.X));
      writer.Key("A_X");
      writer.Uint64(get_A_X_size_local(block_info, sdp));
      writer.Key("schur_complement");
      writer.Uint64(get_schur_complement_size_local(block_info));
      writer.Key("B");
      writer.Uint64(get_B_size_local(sdp));
      writer.Key("Q");
      writer.Uint64(get_Q_size_local(sdp));
      writer.Key("SDP");
      writer.Uint64(get_SDP_size_local(sdp));
      writer.EndObject();
    }));

    // bigint_syrk (Q = P^T P) parameters
    const auto info = bigint_syrk_context.profile_info();
    timers.set_meta("bigint_syrk", to_json([&](auto &writer) {
      writer.StartObject();
      writer.Key("num_primes");
      writer.Uint64(info.num_primes);
      writer.Key("num_groups");
      writer.Uint64(info.num_groups);
      writer.Key("total_block_height_per_node");
      writer.Int(info.total_block_height_per_node);
      writer.Key("input_window_split_factor");
      writer.Uint64(info.input_window_split_factor);
      writer.Key("output_window_split_factor");
      writer.Uint64(info.output_window_split_factor);
      writer.EndObject();
    }));
  }
}
