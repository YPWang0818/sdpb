#include "SDP_Block_Data.hxx"
#include "sdp_solve/SDP.hxx"
#include "sdpb_util/assert.hxx"
#include "sdpb_util/copy_matrix.hxx"

// Kept apart from read_block_data.cxx so that the in-memory SDP constructor
// does not depend on the code that reads an SDP from files (libarchive).

// sdp_block_local in initialized only at comm.Rank() == 0
// Data from sdp_block_local is sent to DistMatrices in sdp
void set_sdp_from_root(const El::Grid &grid, const Block_Info &block_info,
                       const SDP_Block_Data &sdp_block_local, SDP &sdp)
{
  const auto &comm = grid.Comm();

  int index = sdp_block_local.block_index_local;
  El::mpi::Broadcast(index, 0, comm);
  if(index == -1)
    RUNTIME_ERROR("Block data is missing (block_data file not found or "
                  "in-memory block not initialized) on the root rank of "
                  "the block's MPI group");

  const size_t block_index = block_info.block_indices.at(index);

  // sdp.primal_objective_c
  {
    auto &c(sdp.primal_objective_c.blocks.at(index));
    c.SetGrid(grid);
    c.Resize(block_info.get_schur_block_size(block_index), 1);
    copy_matrix_from_root(sdp_block_local.primal_objective_c, c, comm);
  }

  // sdp.free_var_matrix
  {
    auto &B(sdp.free_var_matrix.blocks.at(index));
    B.SetGrid(grid);
    // B block has size P'*N
    // NB: sdp.dual_objective_b must be initialized at this moment!
    // This is done in practice in SDP constructor, but no guaranteed generally.
    // TODO initialize it in Block_Info, for consistency?
    B.Resize(block_info.get_schur_block_size(block_index),
             sdp.dual_objective_b.Height());
    copy_matrix_from_root(sdp_block_local.constraint_matrix, B, comm);
  }

  // sdp.bilinear_bases and sdp.bases_blocks
  for(const size_t parity : {0, 1})
    {
      const size_t bilinear_index_local = 2 * index + parity;

      // Set sdp.bilinear_bases:
      {
        const auto &bilinear_bases_local
          = sdp_block_local.bilinear_bases[parity];
        auto &bilinear_bases = sdp.bilinear_bases.at(bilinear_index_local);

        bilinear_bases.SetGrid(grid);
        const auto height
          = block_info.get_bilinear_bases_height(block_index, parity);
        const auto width
          = block_info.get_bilinear_bases_width(block_index, parity);
        bilinear_bases.Resize(height, width);
        copy_matrix_from_root(bilinear_bases_local, bilinear_bases, comm);
      }

      // Set sdp.bases_blocks:
      {
        auto &bases_block_local = sdp_block_local.bases_blocks[parity];

        const auto height
          = block_info.get_psd_matrix_block_size(block_index, parity);
        const auto width
          = block_info.get_bilinear_pairing_block_size(block_index, parity);

        auto &bases_block = sdp.bases_blocks.at(bilinear_index_local);
        bases_block.SetGrid(grid);
        bases_block.Resize(height, width);
        copy_matrix_from_root(bases_block_local, bases_block, comm);
      }
    }
}
