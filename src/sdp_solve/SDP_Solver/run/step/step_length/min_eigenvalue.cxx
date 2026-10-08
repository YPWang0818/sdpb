#include "sdp_solve/Block_Diagonal_Matrix.hxx"
#include "sdp_solve/SDP_Solver/run/timed_linalg.hxx"

// Minimum eigenvalue of A.  A is assumed to be symmetric.

// Annoyingly, El::HermitianEig modifies 'block'.  It is OK, because
// it is only called from step_length(), which passes in a temporary.
// Still ugly.
El::BigFloat min_eigenvalue(Block_Diagonal_Matrix &A, Timers &timers)
{
  Scoped_Timer timer(timers, "eig");
  El::BigFloat local_min(El::limits::Max<El::BigFloat>());

  for(size_t b = 0; b < A.blocks.size(); ++b)
    {
      auto &block = A.blocks[b];
      El::DistMatrix<El::BigFloat, El::VR, El::STAR> eigenvalues(block.Grid());
      /// There is a bug in El::HermitianEig when there is more than
      /// one level of recursion when computing eigenvalues.  One fix
      /// is to increase the cutoff so that there is no more than one
      /// level of recursion.

      /// An alternate workaround is to compute both eigenvalues and
      /// eigenvectors, but that seems to be significantly slower.
      El::HermitianEigCtrl<El::BigFloat> hermitian_eig_ctrl;
      hermitian_eig_ctrl.tridiagEigCtrl.dcCtrl.cutoff = block.Height() / 2 + 1;

      /// The default number of iterations is 40.  That is sometimes
      /// not enough, so we bump it up significantly.
      hermitian_eig_ctrl.tridiagEigCtrl.dcCtrl.secularCtrl.maxIterations = 16384;
      timed::HermitianEig(timers, "hermitian_eig", El::UpperOrLowerNS::LOWER,
                          block, eigenvalues, hermitian_eig_ctrl,
                          {{"local_block", std::to_string(b / 2)},
                           {"parity", std::to_string(b % 2)}});
      local_min = El::Min(local_min, El::Min(eigenvalues));
    }
  Scoped_Timer allreduce_timer(timers, "allreduce",
                               {{"kind", "mpi"}, {"op", "allreduce"}});
  return El::mpi::AllReduce(local_min, El::mpi::MIN, El::mpi::COMM_WORLD);
}
