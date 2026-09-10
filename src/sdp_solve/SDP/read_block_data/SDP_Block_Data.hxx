#pragma once

#include "pmp2sdp/Block_File_Format.hxx"
#include "sdp_solve/Block_Info.hxx"

#include <El.hpp>

class Dual_Constraint_Group;
struct SDP;

struct SDP_Block_Data
{
  int block_index_local = -1;
  El::Matrix<El::BigFloat> constraint_matrix{};
  El::Matrix<El::BigFloat> primal_objective_c{};

  std::array<El::Matrix<El::BigFloat>, 2> bilinear_bases{};
  std::array<El::Matrix<El::BigFloat>, 2> bases_blocks{};

  SDP_Block_Data() = default;
  SDP_Block_Data(std::istream &block_stream, Block_File_Format format,
                 size_t block_index_local, const Block_Info &block_info);
  // From an in-memory Dual_Constraint_Group (the same data that
  // block_data_<i> files store). group.block_index must equal
  // block_info.block_indices.at(block_index_local).
  SDP_Block_Data(const Dual_Constraint_Group &group, size_t block_index_local,
                 const Block_Info &block_info);

  // Allow move and prohibit copy

  SDP_Block_Data(const SDP_Block_Data &other) = delete;
  SDP_Block_Data(SDP_Block_Data &&other) = default;
  SDP_Block_Data &operator=(const SDP_Block_Data &other) = delete;
  SDP_Block_Data &operator=(SDP_Block_Data &&other) = default;

private:
  void set_bases_blocks(const Block_Info &block_info);
};

// Copy sdp_block_local into the DistMatrices of sdp for its local block.
// sdp_block_local needs to be initialized only at grid.Comm().Rank() == 0;
// the block containers of sdp must already be sized (see read_block_data()).
void set_sdp_from_root(const El::Grid &grid, const Block_Info &block_info,
                       const SDP_Block_Data &sdp_block_local, SDP &sdp);
