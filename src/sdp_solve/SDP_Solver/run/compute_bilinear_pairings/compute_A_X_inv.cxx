#include "sdp_solve/Block_Diagonal_Matrix.hxx"
#include "sdp_solve/Block_Info.hxx"
#include "sdp_solve/SDP_Solver/run/timed_linalg.hxx"

#include <optional>

// A_X_inv = bilinear_base^T X^{-1} bilinear_base for each block

void compute_A_X_inv(
  const Block_Info &block_info, const Block_Diagonal_Matrix &X_cholesky,
  const std::vector<El::DistMatrix<El::BigFloat>> &bases_blocks,
  std::array<std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>,
             2> &A_X_inv,
  Timers &timers)
{
  Scoped_Timer A_X_inv_timer(timers, "A_X_inv");
  A_X_inv[0].resize(bases_blocks.size());
  A_X_inv[1].resize(bases_blocks.size());

  for(size_t index(0); index < bases_blocks.size(); ++index)
    {
      const size_t parity(index % 2), Q_index(index / 2);
      const size_t block_index(block_info.block_indices.at(Q_index));
      const size_t block_size(block_info.num_points.at(block_index)),
        dim(block_info.dimensions.at(block_index));

      // Per-block timer (profile detail >= 1)
      std::optional<Scoped_Timer> block_timer;
      if(timers.detail() >= 1)
        block_timer.emplace(timers, "block",
                            Timer_Attrs{{"block", std::to_string(block_index)},
                                        {"parity", std::to_string(parity)},
                                        {"dim", std::to_string(dim)},
                                        {"K", std::to_string(block_size)}});

      auto &block(bases_blocks[index]);
      auto &X_cholesky_block(X_cholesky.blocks[index]);
      El::DistMatrix<El::BigFloat> temp_space(block);
      timed::Trsm(timers, "trsm", El::LeftOrRight::LEFT,
                  El::UpperOrLowerNS::LOWER, El::Orientation::NORMAL,
                  El::UnitOrNonUnit::NON_UNIT, El::BigFloat(1),
                  X_cholesky_block, temp_space);

      if(dim == 1)
        {
          // The only tile is a copy of the whole symmetric matrix: compute it
          // in place (reusing the tile of the previous iteration)
          auto &tiles(A_X_inv[parity][Q_index]);
          tiles.resize(1);
          if(tiles[0].size() != 1 || tiles[0][0].Height() != block.Width()
             || &tiles[0][0].Grid() != &block.Grid())
            {
              tiles[0].clear();
              tiles[0].emplace_back(block.Width(), block.Width(),
                                    block.Grid());
              tiles[0][0].Align(0, 0);
            }
          auto &tile(tiles[0][0]);
          El::Zero(tile);
          timed::Syrk(timers, "syrk", El::UpperOrLowerNS::LOWER,
                      El::Orientation::TRANSPOSE, El::BigFloat(1), temp_space,
                      El::BigFloat(0), tile);
          local_la::MakeSymmetric(El::UpperOrLower::LOWER, tile);
          continue;
        }

      El::DistMatrix<El::BigFloat> A_X_inv_matrix(block.Width(), block.Width(),
                                                  block.Grid());

      // We have to set this to zero because the values can be NaN.
      // Multiplying 0*NaN = NaN.
      El::Zero(A_X_inv_matrix);
      timed::Syrk(timers, "syrk", El::UpperOrLowerNS::LOWER,
                  El::Orientation::TRANSPOSE, El::BigFloat(1), temp_space,
                  El::BigFloat(0), A_X_inv_matrix);
      local_la::MakeSymmetric(El::UpperOrLower::LOWER, A_X_inv_matrix);

      std::optional<Scoped_Timer> split_timer;
      if(timers.detail() >= 1)
        split_timer.emplace(timers, "split_tiles");
      auto &A_X_inv_block(A_X_inv[parity][Q_index]);
      A_X_inv_block.resize(dim);
      for(size_t column_block = 0; column_block < dim; ++column_block)
        {
          A_X_inv_block[column_block].clear();
          A_X_inv_block[column_block].reserve(dim);
          const size_t column_offset(column_block * block_size);
          for(size_t row_block = 0; row_block < dim; ++row_block)
            {
              const size_t row_offset(row_block * block_size);
              El::DistMatrix<El::BigFloat> submatrix(
                El::View(A_X_inv_matrix, column_offset, row_offset, block_size,
                         block_size));

              A_X_inv_block[column_block].emplace_back(block_size, block_size,
                                                       block.Grid());
              A_X_inv_block[column_block].back().Align(0, 0);
              El::Copy(submatrix, A_X_inv_block[column_block].back());
            }
        }
    }
}

// Untimed version (used by approx_objective)
void compute_A_X_inv(
  const Block_Info &block_info, const Block_Diagonal_Matrix &X_cholesky,
  const std::vector<El::DistMatrix<El::BigFloat>> &bases_blocks,
  std::array<std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>,
             2> &A_X_inv)
{
  Timers timers;
  timers.set_detail(0);
  compute_A_X_inv(block_info, X_cholesky, bases_blocks, A_X_inv, timers);
}
