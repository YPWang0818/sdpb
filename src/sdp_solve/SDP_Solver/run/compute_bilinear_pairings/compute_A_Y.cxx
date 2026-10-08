#include "sdp_solve/Block_Diagonal_Matrix.hxx"
#include "sdp_solve/Block_Info.hxx"
#include "sdp_solve/SDP_Solver/run/timed_linalg.hxx"

#include <optional>

// A_Y[b] = Q[b]'^T A[b] Q[b]' for each block 0 <= b < Q.size()
// A_Y[b], A[b] denote the b-th blocks of A_Y,
// A, resp.

// Q[b]' = Q[b] \otimes 1, where \otimes denotes tensor product

// for each b, L.blocks[b], Q[b], Work[b], and A_Y.blocks[b]
// must have the structure described above for `tensorTransposeCongruence'

// TODO: rename this to compute_A_Y, since this Q is
// different from the big Q that gets inverted.

void compute_A_Y(
  const Block_Info &block_info, const Block_Diagonal_Matrix &Y,
  const std::vector<El::DistMatrix<El::BigFloat>> &bases_blocks,
  std::array<std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>,
             2> &A_Y,
  Timers &timers)
{
  Scoped_Timer A_Y_timer(timers, "A_Y");
  A_Y[0].resize(bases_blocks.size() / 2);
  A_Y[1].resize(bases_blocks.size() / 2);

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
      auto &Y_block(Y.blocks[index]);

      // gemm1 overwrites Y_Q (beta = 0): no need to copy block into it
      El::DistMatrix<El::BigFloat> Y_Q(block.Height(), block.Width(),
                                       block.Grid());
      timed::Gemm(timers, "gemm1", El::Orientation::NORMAL,
                  El::Orientation::NORMAL, El::BigFloat(1), Y_block, block,
                  El::BigFloat(0), Y_Q);

      if(dim == 1)
        {
          // The only tile is Transpose(MakeSymmetric(LOWER, block^T Y_Q)).
          // Computed in place, with the same roundings: the upper triangle
          // of Y_Q^T block (the same products as the lower triangle of
          // block^T Y_Q), then the lower := 1 * upper.
          auto &tiles(A_Y[parity][Q_index]);
          tiles.resize(1);
          if(tiles[0].size() != 1 || tiles[0][0].Height() != block.Width()
             || &tiles[0][0].Grid() != &block.Grid())
            {
              tiles[0].clear();
              tiles[0].emplace_back(block.Width(), block.Width(),
                                    block.Grid());
            }
          auto &tile(tiles[0][0]);
          timed::Gemm_triangle(timers, "gemm2", El::UpperOrLower::UPPER,
                               El::Orientation::TRANSPOSE,
                               El::Orientation::NORMAL, El::BigFloat(1), Y_Q,
                               block, El::BigFloat(0), tile);
          local_la::MakeSymmetric(El::UpperOrLower::UPPER, tile);
          continue;
        }

      El::DistMatrix<El::BigFloat> A_Y_matrix_temp(block.Width(), block.Width(),
                                                   block.Grid());

      timed::Gemm(timers, "gemm2", El::Orientation::TRANSPOSE,
                  El::Orientation::NORMAL, El::BigFloat(1), block, Y_Q,
                  El::BigFloat(0), A_Y_matrix_temp);

      auto &A_Y_block(A_Y[parity][Q_index]);
      A_Y_block.resize(dim);

      local_la::MakeSymmetric(El::UpperOrLower::LOWER, A_Y_matrix_temp);

      std::optional<Scoped_Timer> split_timer;
      if(timers.detail() >= 1)
        split_timer.emplace(timers, "split_tiles");
      for(size_t column_block = 0; column_block < dim; ++column_block)
        {
          A_Y_block[column_block].clear();
          A_Y_block[column_block].reserve(dim);
          const size_t column_offset(column_block * block_size);
          for(size_t row_block = 0; row_block < dim; ++row_block)
            {
              const size_t row_offset(row_block * block_size);
              El::DistMatrix<El::BigFloat> submatrix(
                El::LockedView(A_Y_matrix_temp, column_offset, row_offset,
                               block_size, block_size));

              A_Y_block[column_block].emplace_back(block_size, block_size,
                                                   block.Grid());

              El::Transpose(submatrix, A_Y_block[column_block].back());
            }
        }
    }
}

// Untimed version (used by approx_objective)
void compute_A_Y(
  const Block_Info &block_info, const Block_Diagonal_Matrix &Y,
  const std::vector<El::DistMatrix<El::BigFloat>> &bases_blocks,
  std::array<std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>,
             2> &A_Y)
{
  Timers timers;
  timers.set_detail(0);
  compute_A_Y(block_info, Y, bases_blocks, A_Y, timers);
}
