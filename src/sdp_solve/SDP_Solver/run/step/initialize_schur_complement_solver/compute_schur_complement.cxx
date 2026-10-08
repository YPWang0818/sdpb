#include "sdp_solve/Block_Diagonal_Matrix.hxx"
#include "sdp_solve/Block_Info.hxx"
#include "sdpb_util/Timers/Timers.hxx"
#include "sdpb_util/local_linalg.hxx"

#include <optional>

// Compute the SchurComplement matrix using A_X_inv and
// A_Y and the formula
//
//   S_{(j,r1,s1,k1), (j,r2,s2,k2)} = \sum_{b \in blocks[j]}
//          (1/4) (A_X_inv_{ej s1 + k1, ej r2 + k2}*
//                 A_Y_{ej s2 + k2, ej r1 + k1} +
//                 swaps (r1 <-> s1) and (r2 <-> s2))
//
// where ej = d_j + 1.

void compute_schur_complement(
  const Block_Info &block_info,
  const std::array<
    std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
    &A_X_inv,
  const std::array<
    std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
    &A_Y,
  Block_Diagonal_Matrix &schur_complement, Timers &timers)
{
  Scoped_Timer schur_complement_timer(timers, "schur_complement");

  auto schur_complement_block(schur_complement.blocks.begin());
  size_t Q_index(0);
  // Put these BigFloats at the beginning to avoid memory churn
  El::BigFloat element, product;
  for(auto &block_index : block_info.block_indices)
    {
      const size_t block_size(block_info.num_points[block_index]),
        dim(block_info.dimensions[block_index]);

      // Per-block timer (profile detail >= 1)
      std::optional<Scoped_Timer> block_timer;
      if(timers.detail() >= 1)
        block_timer.emplace(
          timers, "block",
          Timer_Attrs{
            {"kind", "elementwise"},
            {"block", std::to_string(block_index)},
            {"dim", std::to_string(dim)},
            {"K", std::to_string(block_size)},
            {"S", std::to_string(schur_complement_block->Height())},
            {"ranks", std::to_string(schur_complement_block->Grid().Size())}});

      if(dim == 1 && local_la::own_kernels()
         && local_la::all_local(*schur_complement_block))
        {
          // One tile, the Schur block itself, on one rank: write the
          // elements straight into it, and only the lower triangle, which
          // MakeSymmetric(LOWER) copies (truncated) to the upper one. The
          // operations per element are those of the general code below.
          auto &S(schur_complement_block->Matrix());
          for(int64_t column(0); column < S.Width(); ++column)
            for(int64_t row(column); row < S.Height(); ++row)
              {
                element.Zero();
                for(size_t parity(0); parity < 2; ++parity)
                  {
                    const auto &A_X_inv_tile(
                      A_X_inv[parity][Q_index][0][0].LockedMatrix());
                    const auto &A_Y_tile(
                      A_Y[parity][Q_index][0][0].LockedMatrix());
                    for(int term(0); term < 4; ++term)
                      {
                        product = A_X_inv_tile.CRef(row, column);
                        product *= A_Y_tile.CRef(row, column);
                        element += product;
                      }
                  }
                element /= 4;
                S(row, column) = element;
              }
          local_la::MakeSymmetric(El::UpperOrLower::LOWER,
                                  *schur_complement_block);
          ++schur_complement_block;
          ++Q_index;
          continue;
        }

      El::DistMatrix<El::BigFloat> temp(block_size, block_size,
                                        schur_complement_block->Grid()),
        temp_result(temp);
      for(size_t column_block_0 = 0; column_block_0 < dim; ++column_block_0)
        {
          for(size_t row_block_0 = 0; row_block_0 <= column_block_0;
              ++row_block_0)
            {
              size_t result_row_offset(
                ((column_block_0 * (column_block_0 + 1)) / 2 + row_block_0)
                * block_size);

              for(size_t column_block_1 = 0; column_block_1 < dim;
                  ++column_block_1)
                {
                  for(size_t row_block_1 = 0; row_block_1 <= column_block_1;
                      ++row_block_1)
                    {
                      for(int64_t row(0); row < temp_result.LocalHeight();
                          ++row)
                        {
                          for(int64_t column(0);
                              column < temp_result.LocalWidth(); ++column)
                            {
                              element.Zero();
                              for(size_t parity(0); parity < 2; ++parity)
                                {
                                  // Do this the hard way to avoid memory
                                  // allocations
                                  product
                                    = A_X_inv[parity][Q_index][column_block_0]
                                             [row_block_1]
                                               .GetLocalCRef(row, column);
                                  product *= A_Y[parity][Q_index]
                                                [column_block_1][row_block_0]
                                                  .GetLocalCRef(row, column);
                                  element += product;

                                  product
                                    = A_X_inv[parity][Q_index][row_block_0]
                                             [row_block_1]
                                               .GetLocalCRef(row, column);
                                  product
                                    *= A_Y[parity][Q_index][column_block_1]
                                          [column_block_0]
                                            .GetLocalCRef(row, column);
                                  element += product;

                                  product
                                    = A_X_inv[parity][Q_index][column_block_0]
                                             [column_block_1]
                                               .GetLocalCRef(row, column);
                                  product *= A_Y[parity][Q_index][row_block_1]
                                                [row_block_0]
                                                  .GetLocalCRef(row, column);
                                  element += product;

                                  product
                                    = A_X_inv[parity][Q_index][row_block_0]
                                             [column_block_1]
                                               .GetLocalCRef(row, column);
                                  product *= A_Y[parity][Q_index][row_block_1]
                                                [column_block_0]
                                                  .GetLocalCRef(row, column);
                                  element += product;
                                }
                              element /= 4;
                              temp_result.SetLocal(row, column, element);
                            }
                        }
                      size_t result_column_offset(
                        ((column_block_1 * (column_block_1 + 1)) / 2
                         + row_block_1)
                        * block_size);

                      El::DistMatrix<El::BigFloat> result_submatrix(El::View(
                        *schur_complement_block, result_row_offset,
                        result_column_offset, block_size, block_size));

                      El::Copy(temp_result, result_submatrix);
                    }
                }
            }
        }

      local_la::MakeSymmetric(El::UpperOrLower::LOWER, *schur_complement_block);
      ++schur_complement_block;
      ++Q_index;
    }
}
