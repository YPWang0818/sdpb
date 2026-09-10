// Check that SDP built in memory from Dual_Constraint_Groups
// is identical to SDP written by write_sdp() and read back from disk.
#include "pmp/Polynomial_Matrix_Program.hxx"
#include "pmp2sdp/Output_SDP/Output_SDP.hxx"
#include "pmp2sdp/write_sdp.hxx"
#include "sdp_solve/sdp_solve.hxx"
#include "sdpb_util/Environment.hxx"
#include "sdpb_util/copy_matrix.hxx"

#include <catch2/catch_amalgamated.hpp>

#include "unit_tests/util/util.hxx"

#include <filesystem>

namespace fs = std::filesystem;
using Test_Util::REQUIRE_Equal::diff;

namespace
{
  Polynomial poly(std::vector<El::BigFloat> coefficients)
  {
    Polynomial p;
    p.coefficients = std::move(coefficients);
    return p;
  }

  // Deterministic PVM for global block j: dim = 1 + j % 3,
  // degree = j % 4, N = num_dual_variables.
  Polynomial_Vector_Matrix make_pvm(const size_t j, const size_t N)
  {
    const size_t dim = 1 + j % 3;
    const size_t degree = j % 4;
    Simple_Matrix<Polynomial_Vector> polynomials(dim, dim);
    for(size_t r = 0; r < dim; ++r)
      for(size_t s = 0; s < dim; ++s)
        {
          Polynomial_Vector vec;
          for(size_t n = 0; n <= N; ++n)
            {
              std::vector<El::BigFloat> coeffs;
              for(size_t k = 0; k <= degree; ++k)
                {
                  // symmetric in (r,s)
                  const auto value
                    = El::BigFloat(1 + j + n + k + (r + 1) * (s + 1))
                      / El::BigFloat(7 + k);
                  coeffs.push_back((r == s) ? value + 10 : value);
                }
              vec.push_back(poly(coeffs));
            }
          polynomials(r, s) = vec;
        }
    const std::optional<Damped_Rational> prefactor
      = (j % 2 == 0) ? std::nullopt
                     : std::optional<Damped_Rational>(
                         Damped_Rational{1, Boost_Float("0.5"), {-1, -2}});
    return Polynomial_Vector_Matrix(polynomials, prefactor, std::nullopt,
                                    std::nullopt, std::nullopt, std::nullopt,
                                    std::nullopt, std::nullopt);
  }

  El::Matrix<El::BigFloat> to_local(const El::DistMatrix<El::BigFloat> &dist)
  {
    El::DistMatrix<El::BigFloat, El::STAR, El::STAR> star(dist);
    El::Matrix<El::BigFloat> local;
    copy_matrix(star, local);
    return local;
  }

  void diff_sdp(const SDP &a, const SDP &b, const Block_Info &block_info)
  {
    DIFF(a.objective_const, b.objective_const);
    DIFF(to_local(a.dual_objective_b), to_local(b.dual_objective_b));
    if(El::mpi::Rank() == 0)
      {
        REQUIRE(a.normalization.has_value() == b.normalization.has_value());
        if(a.normalization.has_value())
          DIFF(a.normalization.value(), b.normalization.value());
      }
    const size_t num_blocks = block_info.block_indices.size();
    DIFF(a.primal_objective_c.blocks.size(), num_blocks);
    DIFF(b.primal_objective_c.blocks.size(), num_blocks);
    for(size_t i = 0; i < num_blocks; ++i)
      {
        CAPTURE(i);
        CAPTURE(block_info.block_indices.at(i));
        DIFF(to_local(a.primal_objective_c.blocks.at(i)),
             to_local(b.primal_objective_c.blocks.at(i)));
        DIFF(to_local(a.free_var_matrix.blocks.at(i)),
             to_local(b.free_var_matrix.blocks.at(i)));
        for(const size_t parity : {0, 1})
          {
            CAPTURE(parity);
            DIFF(to_local(a.bilinear_bases.at(2 * i + parity)),
                 to_local(b.bilinear_bases.at(2 * i + parity)));
            DIFF(to_local(a.bases_blocks.at(2 * i + parity)),
                 to_local(b.bases_blocks.at(2 * i + parity)));
          }
      }
  }
}

TEST_CASE("sdp_in_memory")
{
  INFO("SDP from in-memory Dual_Constraint_Groups should be identical to "
       "SDP written by write_sdp() and read from disk");
  // Exact comparison: both routes use the same sampled data,
  // and the binary block format stores BigFloats losslessly.
  Test_Util::REQUIRE_Equal::Diff_Precision exact(-1);

  Environment env;
  const Verbosity verbosity = Verbosity::none;
  Timers timers(env, verbosity);

  const size_t num_blocks = 12;
  const size_t N = 3;
  const std::vector<El::BigFloat> objective{1, 2, 3, 4};
  const std::optional<std::vector<El::BigFloat>> normalization
    = std::vector<El::BigFloat>{1, 0, El::BigFloat("0.5"), 0};

  // All ranks can build every PVM deterministically.
  std::vector<size_t> dimensions, num_points;
  for(size_t j = 0; j < num_blocks; ++j)
    {
      const auto pvm = make_pvm(j, N);
      dimensions.push_back(pvm.polynomials.Height());
      num_points.push_back(pvm.sample_points.size());
    }

  // Route 1: PMP distributed round-robin -> Output_SDP -> write_sdp -> SDP
  const fs::path sdp_path
    = fs::temp_directory_path() / "sdpb_unit_tests" / "sdp_in_memory" / "sdp";
  if(El::mpi::Rank() == 0)
    fs::remove_all(sdp_path.parent_path());
  El::mpi::Barrier(El::mpi::COMM_WORLD);
  {
    std::vector<Polynomial_Vector_Matrix> matrices;
    std::vector<size_t> local_to_global;
    std::vector<fs::path> block_paths;
    for(size_t j = El::mpi::Rank(); j < num_blocks; j += El::mpi::Size())
      {
        matrices.push_back(make_pvm(j, N));
        local_to_global.push_back(j);
        block_paths.emplace_back("in-memory/block_" + std::to_string(j));
      }
    const Polynomial_Matrix_Program pmp(objective, normalization, num_blocks,
                                        std::move(matrices),
                                        std::move(local_to_global),
                                        std::move(block_paths));
    const Output_SDP output_sdp(pmp, {"unit_tests"}, timers);
    write_sdp(sdp_path, output_sdp, pmp, Block_File_Format::bin, false,
              timers, verbosity);
  }
  El::mpi::Barrier(El::mpi::COMM_WORLD);
  const Block_Info block_info(env, sdp_path, fs::path("no_checkpoint"), 1,
                              verbosity);
  const El::Grid grid(block_info.mpi_comm.value);
  const SDP sdp_from_disk(sdp_path, block_info, grid, timers);

  // Route 2: Dual_Constraint_Groups for this rank's blocks -> SDP
  {
    // Objective conversion (eq. 3.1 -> 2.2) as in Output_SDP:
    // reuse Output_SDP on a PMP holding only the local blocks.
    std::vector<Polynomial_Vector_Matrix> matrices;
    std::vector<size_t> local_to_global;
    std::vector<fs::path> block_paths;
    for(const size_t j : block_info.block_indices)
      {
        matrices.push_back(make_pvm(j, N));
        local_to_global.push_back(j);
        block_paths.emplace_back("in-memory/block_" + std::to_string(j));
      }
    const Polynomial_Matrix_Program pmp(objective, normalization, num_blocks,
                                        std::move(matrices),
                                        std::move(local_to_global),
                                        std::move(block_paths));
    const Output_SDP output_sdp(pmp, {"unit_tests"}, timers);
    const SDP sdp_in_memory(output_sdp.objective_const,
                            output_sdp.dual_objective_b,
                            output_sdp.dual_constraint_groups,
                            output_sdp.normalization, block_info, grid);
    diff_sdp(sdp_in_memory, sdp_from_disk, block_info);
  }

  SECTION("Block_Info in-memory constructor")
  {
    const Block_Info in_memory(env, dimensions, num_points, 1, verbosity);
    DIFF(in_memory.dimensions, block_info.dimensions);
    DIFF(in_memory.num_points, block_info.num_points);
    // Every block is assigned to exactly one rank
    std::vector<int> owners(num_blocks, 0);
    for(const size_t j : in_memory.block_indices)
      owners.at(j) += 1;
    std::vector<int> total(num_blocks, 0);
    El::mpi::AllReduce(owners.data(), total.data(), num_blocks, El::mpi::SUM,
                       El::mpi::COMM_WORLD);
    for(size_t j = 0; j < num_blocks; ++j)
      {
        CAPTURE(j);
        // A block may be shared by several ranks of one group
        REQUIRE(total.at(j) >= 1);
      }
    // SDP can be built for this mapping too
    std::vector<Dual_Constraint_Group> groups;
    for(size_t i = 0; i < in_memory.block_indices.size(); ++i)
      {
        const size_t j = in_memory.block_indices.at(i);
        groups.emplace_back(j, make_pvm(j, N));
      }
    const El::Grid grid2(in_memory.mpi_comm.value);
    const SDP sdp2(El::BigFloat(0), std::vector<El::BigFloat>(N, 0), groups,
                   std::nullopt, in_memory, grid2);
    DIFF(sdp2.primal_objective_c.blocks.size(),
         in_memory.block_indices.size());
  }
  El::mpi::Barrier(El::mpi::COMM_WORLD);
  if(El::mpi::Rank() == 0)
    fs::remove_all(sdp_path.parent_path());
}
