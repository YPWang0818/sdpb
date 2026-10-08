#include "sdp_solve/Block_Diagonal_Matrix.hxx"
#include "sdpb_util/Timers/Timers.hxx"

#include <cmath>
#include <optional>

// Fast step length (--fastStepLength).
//
// For every block T_b = L^{-1} dM L^{-T} (symmetric), lambda_min(T_b) is
// computed in double precision with LAPACK (Elemental's HermitianEig<double>)
// and turned into the block's step alpha_b = 1 if lambda_min > -gamma, else
// -gamma / lambda_min, exactly as the arbitrary-precision path does.  The
// result is verified in arbitrary precision: I + alpha_b T_b must be positive
// definite (Cholesky).  With the exact lambda_min its smallest eigenvalue is
// 1 - gamma (or 1 + lambda_min > 1 - gamma), so the check only fails when the
// double computation was wrong (entries outside the double range, severe
// cancellation); then false is returned and the caller uses the exact
// eigensolver.  Since every alpha_b passes its own check, the global step
// min_b alpha_b keeps every block positive definite (the eigenvalues of
// I + alpha T are affine in alpha).
//
// The result differs from the exact path only in the last digits of alpha
// (relative 1e-16), which changes the iteration path slightly but not the
// optimum the solver converges to within its thresholds.

namespace
{
  // Full copy of a (distributed) block on every rank, as doubles.
  // Returns false if an entry does not fit a double.
  bool to_double_matrix(const El::DistMatrix<El::BigFloat> &block,
                        El::Matrix<double> &out, El::Matrix<El::BigFloat> &full)
  {
    El::DistMatrix<El::BigFloat, El::STAR, El::STAR> star(block);
    full = star.LockedMatrix();
    const int n = full.Height();
    out.Resize(n, n);
    for(int j = 0; j < n; ++j)
      for(int i = 0; i < n; ++i)
        {
          const double v = static_cast<double>(full(i, j));
          if(!std::isfinite(v))
            return false;
          out(i, j) = v;
        }
    return true;
  }
}

bool fast_step_length_value(const Block_Diagonal_Matrix &T,
                            const El::BigFloat &gamma, Timers &timers,
                            El::BigFloat &alpha)
{
  Scoped_Timer timer(timers, "eig_fast");
  El::BigFloat local_alpha(1);
  bool ok = true;
  El::Matrix<double> Td;
  El::Matrix<El::BigFloat> full;
  for(size_t b = 0; b < T.blocks.size() && ok; ++b)
    {
      const auto &block = T.blocks[b];
      if(block.Height() == 0)
        continue;
      {
        Scoped_Timer convert_timer(timers, "to_double");
        if(!to_double_matrix(block, Td, full))
          {
            ok = false;
            break;
          }
      }
      double lambda_min;
      {
        Scoped_Timer eig_timer(timers, "hermitian_eig_double",
                               {{"kind", "eig_double"},
                                {"n", std::to_string(Td.Height())},
                                {"local_block", std::to_string(b / 2)},
                                {"parity", std::to_string(b % 2)}});
        El::Matrix<double> w;
        El::Matrix<double> A(Td);
        El::HermitianEig(El::UpperOrLowerNS::LOWER, A, w);
        lambda_min = w(0, 0);
        for(int i = 1; i < w.Height(); ++i)
          lambda_min = std::min(lambda_min, w(i, 0));
      }
      // Block step: alpha_b = 1 or -gamma / lambda_min, as in step_length()
      El::BigFloat alpha_b(1);
      const El::BigFloat lambda_big(lambda_min);
      if(!(lambda_big > -gamma))
        alpha_b = -gamma / lambda_big;

      // Safeguard: I + alpha_b T_b must be positive definite in BigFloat
      {
        Scoped_Timer check_timer(timers, "safeguard_cholesky",
                                 {{"kind", "cholesky"},
                                  {"n", std::to_string(full.Height())}});
        El::Matrix<El::BigFloat> C(full);
        C *= alpha_b;
        El::ShiftDiagonal(C, El::BigFloat(1));
        try
          {
            El::Cholesky(El::UpperOrLowerNS::LOWER, C);
          }
        catch(std::exception &)
          {
            ok = false;
            break;
          }
      }
      local_alpha = El::Min(local_alpha, alpha_b);
    }
  // Every rank must agree on the path: the global step is the minimum over
  // all blocks of all ranks, and any failure switches everyone to the exact path.
  Scoped_Timer allreduce_timer(timers, "allreduce",
                               {{"kind", "mpi"}, {"op", "allreduce"}});
  El::byte any_failed = ok ? 0 : 1;
  any_failed = El::mpi::AllReduce(any_failed, El::mpi::LOGICAL_OR,
                                  El::mpi::COMM_WORLD);
  if(any_failed)
    return false;
  alpha = El::mpi::AllReduce(local_alpha, El::mpi::MIN, El::mpi::COMM_WORLD);
  return true;
}
