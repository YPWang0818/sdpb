#pragma once

#include <El.hpp>

struct Environment
{
  // Shared memory communicator.
  // Contains all ranks for a given node.
  El::mpi::Comm comm_shared_mem;

  Environment();
  Environment(int argc, char **argv);
  ~Environment();

  // Set precision for El::BigFloat (GMP) and Boost_Float (MPFR)
  static void set_precision(mp_bitcnt_t digits2);

  // Total number of nodes
  [[nodiscard]] int num_nodes() const;
  // Node index for current rank, 0..(num_nodes-1)
  [[nodiscard]] int node_index() const;

  // Memory used on a node at the initialization (in bytes)
  [[nodiscard]] size_t initial_node_mem_used() const;

  [[nodiscard]] bool sigterm_received() const;
  // Behave as if SIGTERM was received: SDP_Solver::run() will stop
  // at the next iteration with SDP_Solver_Terminate_Reason::SIGTERM_Received.
  // Useful for host applications embedding SDPB.
  static void request_termination();
  // Forget a termination request (or a received SIGTERM), so that a
  // subsequent SDP_Solver::run() proceeds normally.
  static void clear_termination_request();

private:
  El::Environment env;
  size_t _initial_node_mem_used = 0;
  int _num_nodes = -1;
  int _node_index = -1;

  void initialize();
  void finalize();
};
