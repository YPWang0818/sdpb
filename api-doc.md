# SDPB C++ API Reference (for embedding SDPB as a library)

This document maps the **complete C++ API** of this repository, i.e. everything a host
application can call once it links against SDPB's static libraries — including the large
part of the API that is *not* declared in any header (free functions that are only
forward-declared inside other `.cxx` files). It complements `architecture.md`, which
describes the design; this file is about **signatures, contracts, and how to call them**.

Conventions:

- All paths are relative to `src/` unless they start with `test/`, `docs/`, or `wscript`.
- Signatures are copied verbatim from the source. For functions without a header, the
  signature is taken from the **definition** and the defining `file:line` is given.
- 🔒 marks an API that has **no header**: to call it you must write the forward
  declaration yourself (copy the signature shown here).
- ⚠️ marks a pitfall that matters specifically when embedding.
- There is no namespace: every SDPB symbol lives in the global namespace. Matrix types come
  from Elemental (`El::`), multiprecision floats are `El::BigFloat` (GMP `mpf`) and
  `Boost_Float` (`boost::multiprecision::mpfr_float`).

---

## Contents

1. [Building and linking against SDPB](#1-building-and-linking-against-sdpb)
2. [Runtime model, global state and embedding pitfalls](#2-runtime-model-global-state-and-embedding-pitfalls)
3. [Recipes](#3-recipes)
4. [`sdpb_util` — infrastructure](#4-sdpb_util--infrastructure)
5. [`pmp` — polynomial matrix program model](#5-pmp--polynomial-matrix-program-model)
6. [`pmp_read` — PMP readers](#6-pmp_read--pmp-readers)
7. [`pmp2sdp_lib` — PMP → SDP conversion and writers](#7-pmp2sdp_lib--pmp--sdp-conversion-and-writers)
8. [`sdp_solve` — SDP data and the solver](#8-sdp_solve--sdp-data-and-the-solver)
9. [`src/sdpb` — driver code (executable only)](#9-srcsdpb--driver-code-executable-only)
10. [Post-processing tools (executables only)](#10-post-processing-tools-executables-only)
11. [Known API defects](#11-known-api-defects)

---

## 1. Building and linking against SDPB

### 1.1 What is a library and what is not

`wscript` builds five **static libraries** and several executables:

| Library (`lib<name>.a`) | Sources | Uses (link after it) |
|---|---|---|
| `sdp_solve`   | `sdp_solve/**` | `pmp2sdp_lib`, `sdpb_util` |
| `pmp_read`    | `pmp_read/**`  | `pmp`, `pmp2sdp_lib`, `sdpb_util` |
| `pmp2sdp_lib` | `pmp2sdp/**` except `main.cxx`, `Pmp2sdp_Parameters/` | `pmp`, `sdpb_util` |
| `pmp`         | `pmp/**` | `sdpb_util` |
| `sdpb_util`   | 10 `.cxx` files in `sdpb_util/` (rest is header-only) | externals |

Suggested link order: `-lsdp_solve -lpmp_read -lpmp2sdp_lib -lpmp -lsdpb_util` followed by
the external libraries.

⚠️ **Not in any library** (compiled straight into their executables — copy the sources
into your build if you need them):

- `src/sdpb/*.cxx` — `solve()`, `save_solution()`, `write_block_timings()`,
  `write_profiling()`, `SDPB_Parameters`. The high-level "solve an SDP directory" routine is
  therefore **not** available from `libsdp_solve.a`; §3.1 shows the ~20 lines to reproduce it.
- `src/pmp2sdp/main.cxx`, `Pmp2sdp_Parameters/`.
- `src/spectrum/`, `src/approx_objective/`, `src/outer_limits/`, `src/pmp2functions/`,
  `src/sdp2input/`, `src/pvm2sdp/`.

⚠️ No headers are installed. Include paths used by the code are rooted at `src/`
(`#include "sdp_solve/SDP.hxx"`), and `external/` (Catch2 only).

### 1.2 Compiler settings

From `wscript:25-31` — reproduce these in your build:

- C++17, compiled with the MPI wrapper (`mpicxx`).
- Include dirs: `<sdpb>/src`, plus the external packages' include dirs.
- Defines: `OMPI_SKIP_MPICXX`, and `SDPB_VERSION_STRING="<string>"` (**required**:
  `sdp_solve/SDP_Solver/save_checkpoint.cxx:107` uses it, as do several mains).
- External dependencies: MPI, Boost (program_options, serialization, iostreams, filesystem
  pieces, stacktrace), GMP with C++ bindings (`gmpxx`), MPFR, the bootstrap-collaboration
  fork of Elemental (`El`), libxml2, RapidJSON (headers), libarchive, a CBLAS (OpenBLAS),
  FLINT; MPSolve only for `spectrum`. See `waf-tools/*.py` for the exact flags each tool adds.

---

## 2. Runtime model, global state and embedding pitfalls

SDPB was written as a set of MPI programs, not as a re-entrant library. The following
constraints are **not negotiable without patching**:

| Topic | Behaviour | Consequence for a host application |
|---|---|---|
| **MPI communicator** | Block distribution, reductions and the Q matrix all use `MPI_COMM_WORLD` / `El::mpi::COMM_WORLD` (31 uses in `sdp_solve/`). `Block_Info::allocate_blocks` sizes itself from `COMM_WORLD`. | Every rank of `MPI_COMM_WORLD` must call the solver collectively. You cannot run SDPB on a sub-communicator. |
| **Node layout** | `Environment` splits `COMM_WORLD` by shared-memory node; `allocate_blocks` asserts every node has the same number of ranks (`Block_Info/allocate_blocks.cxx:14`). | Launch with a uniform ranks-per-node count. |
| **MPI init/finalize** | `Environment` wraps `El::Environment` (calls `El::Initialize`/`El::Finalize`). In Elemental, `Initialize` only calls `MPI_Init` if MPI isn't initialized yet, and `Finalize` only finalizes what it initialized — check this against the Elemental fork you build with. | Create **one** `Environment` for the process lifetime. Destroy every SDPB object (they hold MPI communicators, e.g. `Block_Info::mpi_comm`) **before** the `Environment`. |
| **Precision** | `Environment::set_precision(bits)` sets the *global* GMP default precision (`El::gmp::SetPrecision`) and MPFR's `Boost_Float::default_precision`. `El::BigFloat`s keep the precision they were created with. | Call it before creating any `BigFloat` (including `Solver_Parameters` defaults). Binary SDP block files **must** match this precision exactly (`SDP_Block_Data.cxx:41`). Running two precisions in one process means re-calling `set_precision` and rebuilding everything. |
| **SIGTERM** | `Environment::initialize` installs its own `SIGTERM` handler (`Environment.cxx:62`). `SDP_Solver::run` polls it every iteration and returns `SIGTERM_Received`. `src/sdpb/solve.cxx` then calls `MPI_Finalize(); exit(SIGTERM)` — the library itself does not. | Your SIGTERM handler is replaced. If you need yours, re-install it after constructing `Environment`. To stop a running solve from the host (e.g. another thread), call `Environment::request_termination()`; `run()` returns `SIGTERM_Received` at the next iteration. ⚠️ The flag is sticky: call `Environment::clear_termination_request()` before the next `run()`, or it returns immediately. The flag is a lock-free `std::atomic<bool>`, so both calls are thread- and signal-safe. |
| **BLAS threading** | `BigInt_Shared_Memory_Syrk_Context` calls `openblas_set_num_threads(1)` when built with `OPENBLAS_THREAD` (`BigInt_Shared_Memory_Syrk_Context.cxx:365-371`). | Global side effect on the host's OpenBLAS. |
| **Shared memory** | Each `run()` allocates MPI shared-memory windows (`MPI_Win_allocate_shared`), sized by `Solver_Parameters::max_shared_memory_bytes` (0 = ~50% of free RAM). | Budget memory, or set `max_shared_memory_bytes` explicitly. Allocation happens again on every `run()`. |
| **Errors** | Errors are `std::runtime_error`/`std::logic_error` thrown via `ASSERT`/`RUNTIME_ERROR` (with stack trace in the message). They are usually thrown on *some* ranks only. | Catching an exception and continuing is generally unsafe: other ranks will hang at the next collective. SDPB's own mains do `El::ReportException(e); El::mpi::Abort(COMM_WORLD, 1)`. `Shared_Window_Array`'s destructor deliberately skips `MPI_Win_free` during stack unwinding. |
| **Console output** | Progress goes to stdout via `El::Output`/`std::cout` on rank 0, gated by `Verbosity`. Warnings go to stderr via `PRINT_WARNING`. | Use `Verbosity::none` to silence most output. |
| **Filesystem** | See the table below. | Point every path to directories you own. |

Filesystem side effects of the solver API:

| Call | Reads | Writes |
|---|---|---|
| `Block_Info(env, sdp_path, checkpoint_in, …)` | `sdp_path/{control,block_info_*,objectives}.json`, `checkpoint_in/block_timings` or `sdp_path/block_timings` | — |
| `SDP(sdp_path, …)` | `objectives.json`, `normalization.json`, `block_data_*` | — |
| `SDP_Solver(…)` → `load_checkpoint(parameters.checkpoint_in)` | `checkpoint_in/checkpoint.json`, `checkpoint_<g>_<rank>`, else `x_0.txt`… | — |
| `SDP_Solver::run(…, iterations_json_path, …)` | — | `iterations_json_path` and `iterations_json_path.parent_path()/c_minus_By/c_minus_By.json` (both skipped if the path is empty); checkpoints to `parameters.checkpoint_out` every `checkpoint_interval` seconds (skipped if empty) |
| `SDP_Solver::save_checkpoint(dir, …)` | — | `dir/checkpoint_<g>_<rank>`, `dir/checkpoint.json` (no-op if `dir` is empty) |

⚠️ **Empty `checkpoint_in`**: `fs::path("") / "checkpoint.json"` is the *relative* path
`checkpoint.json`, so an empty `checkpoint_in` makes `SDP_Solver`'s constructor look for a
checkpoint (and `x_0.txt`) **in the current working directory**. Set `checkpoint_in` to a
dedicated (possibly non-existent) directory.

⚠️ **`Solver_Parameters` has no in-class defaults.** The defaulted constructor leaves every
field uninitialized; defaults exist only inside `Solver_Parameters::options()`. Use the
helper in §3.6.

---

## 3. Recipes

All recipes run on **every** MPI rank.

### 3.1 Solve an SDP directory (equivalent of the `sdpb` executable)

```cpp
#include "sdp_solve/sdp_solve.hxx"          // Block_Info, SDP_Solver, Write_Solution, read_text_block
#include "sdpb_util/Environment.hxx"
#include "sdpb_util/Timers/Timers.hxx"

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
  Environment env(argc, argv);                 // MPI + Elemental, SIGTERM handler
  Environment::set_precision(768);             // before any BigFloat is created
  try
    {
      const auto start = std::chrono::high_resolution_clock::now();
      const Verbosity verbosity = Verbosity::regular;

      Solver_Parameters params = default_solver_parameters(); // §3.6
      params.precision = 768;
      params.checkpoint_in = "/scratch/run1/ck";    // never leave empty (see §2)
      params.checkpoint_out = "/scratch/run1/ck";   // "" disables checkpoints

      const fs::path sdp_path = "/scratch/run1/sdp"; // directory or .zip written by pmp2sdp

      Block_Info block_info(env, sdp_path, params.checkpoint_in,
                            /*proc_granularity*/ 1, verbosity);
      El::Grid grid(block_info.mpi_comm.value);
      Timers timers(env, verbosity);
      SDP sdp(sdp_path, block_info, grid, timers);
      SDP_Solver solver(params, verbosity, /*require_initial_checkpoint*/ false,
                        block_info, grid, sdp.dual_objective_b.Height());

      El::Matrix<int32_t> block_timings_ms;
      const auto properties = to_property_tree(params); // stored in checkpoint.json
      SDP_Solver_Terminate_Reason reason = solver.run(
        env, params, verbosity, properties, block_info, sdp, grid, start,
        /*iterations_json_path*/ fs::path{}, timers, block_timings_ms);

      if(El::mpi::Rank() == 0)
        std::cout << reason << " primal=" << solver.primal_objective
                  << " dual=" << solver.dual_objective
                  << " gap=" << solver.duality_gap << "\n";
      // optional: solver.save_checkpoint(params.checkpoint_out, verbosity, properties);
      // optional: copy src/sdpb/save_solution.cxx to write out.txt / y.txt / x_*.txt
    }
  catch(std::exception &e)
    {
      El::ReportException(e);
      El::mpi::Abort(El::mpi::COMM_WORLD, 1);
    }
}
```

Optional load-balancing pass (what `sdpb` does when there is no `block_timings` file and
more than one rank; `sdpb/main.cxx:86-151`): run `solve` once with `max_iterations = 2` and
all thresholds set to 0, take the resulting `block_timings_ms`, then rebuild the mapping:

```cpp
Block_Info tuned(env, sdp_path, block_timings_ms, /*proc_granularity*/ 1, verbosity);
swap(block_info, tuned);   // free function in Block_Info.hxx
```

### 3.2 PMP file(s) → SDP directory (equivalent of `pmp2sdp`)

```cpp
#include "pmp_read/pmp_read.hxx"
#include "pmp2sdp/write_sdp.hxx"

Environment::set_precision(768);
Timers timers(env, verbosity);
Polynomial_Matrix_Program pmp = read_polynomial_matrix_program(
  env, fs::path("pmp.json"), /*max_num_poles*/ -1, verbosity, timers); // .json/.m/.xml/.nsv
Output_SDP out(pmp, /*command_arguments*/ {"my_app"}, timers);
write_sdp("/scratch/run1/sdp", out, pmp, Block_File_Format::bin, /*zip*/ false,
          timers, verbosity);
```

### 3.3 Build a PMP in memory, then solve (no patches needed)

`Polynomial_Matrix_Program` can be constructed directly. This recipe converts it to an SDP
through `write_sdp`, i.e. a directory on disk; use a tmpfs such as `/dev/shm` to make this
cheap. §3.5 does the same without any files. ⚠️ With several nodes, the directory must be on a filesystem visible to all
ranks.

```cpp
#include "pmp/Polynomial_Matrix_Program.hxx"
#include "pmp2sdp/write_sdp.hxx"

// Constraint: (1,y) . M(x) >= 0 for x >= 0, M = 1x1 matrix of N+1 = 3 polynomials
auto poly = [](std::vector<El::BigFloat> coeffs) {  // a_0 + a_1 x + ...
  Polynomial p; p.coefficients = std::move(coeffs); return p;
};
Simple_Matrix<Polynomial_Vector> M(1, 1);
M(0, 0) = {poly({1, 0, 1}), poly({0, 1}), poly({-1, 0, 0, 1})};

std::vector<Polynomial_Vector_Matrix> matrices;
std::vector<size_t> local_to_global;
std::vector<fs::path> block_paths;
if(El::mpi::Rank() == 0)   // each block must be owned by exactly one rank
  {
    matrices.emplace_back(M, std::nullopt, std::nullopt, std::nullopt,
                          std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    local_to_global.push_back(0);
    block_paths.emplace_back("in-memory/block_0");   // label only, must be non-empty
  }
Polynomial_Matrix_Program pmp(
  /*objective a_0..a_N*/ {El::BigFloat(0), El::BigFloat(-1), El::BigFloat(0)},
  /*normalization*/ std::vector<El::BigFloat>{1, 0, 0}, // or std::nullopt
  /*num_matrices*/ 1, std::move(matrices), std::move(local_to_global),
  std::move(block_paths));

Output_SDP out(pmp, {"my_app"}, timers);
write_sdp("/dev/shm/job42/sdp", out, pmp, Block_File_Format::bin, false, timers,
          verbosity);
// ...then continue with §3.1 using sdp_path = "/dev/shm/job42/sdp"
```

All `std::nullopt` arguments of `Polynomial_Vector_Matrix` get the same defaults as omitted
fields in `pmp.json` (§5.2). The objective and normalization must be identical on all ranks.

### 3.4 Standard SDP in LMI form, fully in memory (no patches needed)

SDPB can solve a general (block-diagonal) linear matrix inequality directly:

```
maximize  f + b·y   over y ∈ R^N   subject to   M_0 + Σ_n y_n M_n ⪰ 0
```

where every `M_n` is a symmetric matrix with the same block structure. This is the
`num_points == 1` special case of a PMP, and it is exactly what the in-memory `SDP`
constructor (§8.4) supports, so no disk round trip and no patch is needed. It is the path
`outer_limits` uses (`outer_limits/compute_optimal/setup_constraints.cxx`).

Convention (verified in `SDP_Solver/run/constraint_matrix_weighted_sum.cxx:55`): for
`r ≠ s` the solver uses `A_p = ½(E_rs + E_sr)`, so constraint row `p ↔ (r,s)` reads
`Y_rs = c_p − (B y)_p` with no factor-of-2 on off-diagonal entries. The solver's `Y` is
therefore `M_0 + Σ y_n M_n`, and `c_p = M_0[r,s]`, `B[p,n] = −M_{n+1}[r,s]`.

Row order within a block of size `d` is `s` outer, `r ≤ s` inner:
`(0,0), (0,1), (1,1), (0,2), (1,2), (2,2), …`, i.e. `p = s(s+1)/2 + r`, for a total of
`d(d+1)/2` rows.

```cpp
#include "sdp_solve/sdp_solve.hxx"
#include "sdpb_util/Environment.hxx"
#include "sdpb_util/Timers/Timers.hxx"

// Inputs (identical on every rank):
//   N        number of variables y
//   dims[j]  size of block j
//   M[n][j]  block j of matrix M_n (El::Matrix<El::BigFloat>, symmetric), n = 0..N
//   b[n]     objective coefficients, f the constant term
std::vector<std::vector<El::BigFloat>> c;      // c[j], length dims[j]*(dims[j]+1)/2
std::vector<El::Matrix<El::BigFloat>> B;       // B[j], same number of rows, N columns
for(size_t j = 0; j < dims.size(); ++j)
  {
    const size_t d = dims[j];
    c.emplace_back();
    B.emplace_back(d * (d + 1) / 2, N);
    size_t p = 0;
    for(size_t s = 0; s < d; ++s)
      for(size_t r = 0; r <= s; ++r, ++p)
        {
          c[j].push_back(M[0][j](r, s));
          for(size_t n = 0; n < N; ++n)
            B[j](p, n) = -M[n + 1][j](r, s);
        }
  }

Block_Info block_info(env, dims, verbosity);   // in-memory ctor: num_points = 1 per block
El::Grid grid(block_info.mpi_comm.value);
El::Grid global_grid;                          // [STAR,STAR] inputs live on the default grid
El::DistMatrix<El::BigFloat, El::STAR, El::STAR> yp_to_y(N, N, global_grid),
  b_star(N, 1, global_grid);
El::Identity(yp_to_y, N, N);                   // identity = no change of variables
for(size_t n = 0; n < N; ++n)
  b_star.Set(n, 0, b[n]);
std::vector<El::BigFloat> normalization(N + 1, El::BigFloat(0));
normalization[0] = 1;                          // stored only; used for z.txt, harmless here

SDP sdp(/*f*/ f, c, B, yp_to_y, b_star, normalization, /*primal_c_scale*/ El::BigFloat(1),
        block_info, grid);
SDP_Solver solver(params, verbosity, /*require_initial_checkpoint*/ false, block_info,
                  grid, N);
El::Matrix<int32_t> block_timings_ms;
Timers timers(env, verbosity);
SDP_Solver_Terminate_Reason reason = solver.run(
  env, params, verbosity, to_property_tree(params), block_info, sdp, grid, start,
  /*iterations_json_path*/ std::filesystem::path{}, timers, block_timings_ms);

// Results: solver.dual_objective == f + b.y at the optimum, solver.y (see §3.7),
// solver.Y.blocks[2*i] is the PSD matrix M_0 + sum y_n M_n for local block i
// (the odd-parity blocks [2*i+1] are empty, 0x0).
```

Notes:
- `env`, `params` and `start` are set up as in §3.1. `c` and `B` are **global** (indexed by
  block index); the constructor picks out this rank's blocks via `block_info.block_indices`.
- Minimization: negate `b` (and `f`). A problem in primal standard form
  (`min C·X s.t. Tr(A_i X) = b_i, X ⪰ 0`) must be dualized to the LMI form first; SDPB has
  no API for arbitrary constraint matrices `A_i`.
- With `num_points == 1` the bilinear bases are trivial (`[1]`), so none of SDPB's
  polynomial-structure machinery is used: this is a dense multiprecision interior-point
  LMI solver. The `N×N` matrix `Q` is formed and Cholesky-factored every iteration, so it
  suits moderate sizes (`N` up to a few thousand, blocks up to a few hundred), not large
  sparse SDPs.
- `Block_Info(env, dims, verbosity)` assigns each block a cost of `(d(d+1)/2)²` and requires
  every rank to receive at least one block; a tiny problem on many ranks fails the assertion
  in `allocate_blocks.cxx:73`. One rank always works.
- A warm start is possible: write into `solver.y.blocks[i]` (every local copy) and
  `solver.X` / `solver.Y` before calling `run()`.

### 3.5 PMP → SDP fully in memory (no files)

The `python-api` branch adds the pieces that make this possible without a disk round trip:

- `Block_Info(env, dimensions, num_points, proc_granularity, verbosity)` — an in-memory
  block mapping with the real number of sample points per block (§8.1).
- `SDP(objective_const, dual_objective_b, groups, normalization, block_info, grid)` — an
  `SDP` built from sampled `Dual_Constraint_Group`s (§8.4). The result is bit-for-bit
  identical to writing with `write_sdp` and reading back; the unit test
  `test/src/unit_tests/cases/sdp_in_memory.test.cxx` checks exactly that.

The order matters: the block mapping decides which blocks each rank owns, and the `SDP`
constructor needs the groups for exactly those blocks, in that order.

```cpp
#include "pmp/Polynomial_Matrix_Program.hxx"
#include "pmp2sdp/Output_SDP/Output_SDP.hxx"
#include "sdp_solve/sdp_solve.hxx"

// Inputs, identical on every rank:
//   objective (a_0..a_N), normalization (std::optional, n_0..n_N), num_blocks,
//   make_pvm(j): builds the Polynomial_Vector_Matrix of global block j (§3.3)

// 1. Global block sizes. dim = polynomials.Height(), num_points = sample_points.size();
//    either compute them analytically or build every block once and read them off.
std::vector<size_t> dimensions(num_blocks), num_points(num_blocks);
for(size_t j = 0; j < num_blocks; ++j)
  {
    const Polynomial_Vector_Matrix pvm = make_pvm(j);
    dimensions[j] = pvm.polynomials.Height();
    num_points[j] = pvm.sample_points.size();
  }

// 2. Block mapping (collective)
Block_Info block_info(env, dimensions, num_points, /*proc_granularity*/ 1, verbosity);
El::Grid grid(block_info.mpi_comm.value);

// 3. A PMP holding exactly this rank's blocks, in block_info.block_indices order
std::vector<Polynomial_Vector_Matrix> matrices;
std::vector<size_t> local_to_global;
std::vector<std::filesystem::path> block_paths;
for(const size_t j : block_info.block_indices)
  {
    matrices.push_back(make_pvm(j));
    local_to_global.push_back(j);
    block_paths.emplace_back("in-memory/block_" + std::to_string(j)); // label, non-empty
  }
const Polynomial_Matrix_Program pmp(objective, normalization, num_blocks,
                                    std::move(matrices), std::move(local_to_global),
                                    std::move(block_paths));

// 4. Eliminate the normalization and sample (rank-local), then build the SDP (collective)
Timers timers(env, verbosity);
const Output_SDP out(pmp, {"my_app"}, timers);
const SDP sdp(out.objective_const, out.dual_objective_b, out.dual_constraint_groups,
              out.normalization, block_info, grid);

// 5. Solve as in §3.1
SDP_Solver solver(params, verbosity, false, block_info, grid, sdp.dual_objective_b.Height());
```

Notes:
- **Every rank of a block's MPI group must pass that block's group**, with matching `dim`
  and `num_points` (asserted on every rank), although only the group root's data is
  actually copied into the distributed matrices. With several ranks per block this means a
  redundant copy of the sampled block on each of them. Ranks in the same group have the
  same `block_indices`, so the loop in step 3 does the right thing.
- `out.normalization` is stored on every rank here (the file-based constructor only sets it
  on rank 0); it is only used to reconstruct `z` from `y` (§3.7).
- Block costs for the mapping are estimated as `schur_block_size²`. For a long-running
  family of solves you can measure `block_timings_ms` with a 2-iteration `run()`, but there
  is no in-memory `Block_Info` constructor taking timings (and no default constructor). The
  only route is `block_info.allocate_blocks(env, costs, 1, verbosity)` on an existing
  object, which overwrites its MPI group/communicator without freeing the old ones (§8.1);
  everything built on the old mapping (`grid`, `sdp`, `solver`) must be rebuilt afterwards.
- If you already have sampled data (`B`, `c`, sampled bilinear bases) from elsewhere, fill
  `Dual_Constraint_Group`s by hand (§7.2) and skip steps 3–4. `B` must then already have
  the normalization eliminated.

### 3.6 Default `Solver_Parameters`

```cpp
#include <boost/program_options.hpp>

Solver_Parameters default_solver_parameters() // call after Environment::set_precision
{
  Solver_Parameters p;
  auto desc = p.options();                  // binds option values to p's fields
  boost::program_options::variables_map vm;
  const char *argv[] = {"sdpb"};
  boost::program_options::store(
    boost::program_options::parse_command_line(1, argv, desc), vm);
  boost::program_options::notify(vm);       // applies defaults, including maxSharedMemory
  return p;                                 // checkpoint_in/out are left empty
}
```

The defaults are listed in §8.7.

### 3.7 Reading results from an `SDP_Solver`

```cpp
// y (length N) is duplicated on every local block; gather it onto every group rank:
El::DistMatrix<El::BigFloat, El::STAR, El::STAR> y_star(solver.y.blocks.at(0));
// y_star.LockedMatrix()(i, 0) == y_i, identical on all ranks

// x, X, Y are distributed by block: local entry i is global block block_info.block_indices[i]
// X.blocks[2*i + parity] and Y.blocks[2*i + parity] are the even/odd PSD blocks.

// z (the PMP variables of Manual eq. 3.1) from y and the normalization (rank 0 only):
if(sdp.normalization)
  {
    std::vector<El::BigFloat> z(sdp.normalization->size());
    fill_weights(y_star.LockedMatrix(),
                 max_normalization_index(*sdp.normalization), *sdp.normalization, z);
  }

// c - B.y as JSON (used by spectrum):
save_c_minus_By("/scratch/run1/out/c_minus_By/c_minus_By.json", block_info, sdp,
                solver.y, verbosity, timers);
```

`solver.primal_objective`, `dual_objective`, `duality_gap`, `primal_error()`, `dual_error`
and `R_error` are global and identical on all ranks after `run()`.

---

## 4. `sdpb_util` — infrastructure

### 4.1 `Environment` — `sdpb_util/Environment.hxx`

```cpp
struct Environment
{
  El::mpi::Comm comm_shared_mem;               // all ranks on this node
  Environment();                               // = Environment(0, nullptr)
  Environment(int argc, char **argv);
  ~Environment();
  static void set_precision(mp_bitcnt_t digits2);
  [[nodiscard]] int num_nodes() const;
  [[nodiscard]] int node_index() const;        // 0..num_nodes-1
  [[nodiscard]] size_t initial_node_mem_used() const; // bytes, from /proc/meminfo
  [[nodiscard]] bool sigterm_received() const;
  static void request_termination();        // as if SIGTERM arrived: run() stops at the next iteration
  static void clear_termination_request();  // forget a request / received SIGTERM before the next run()
private:
  El::Environment env; /* ... */
};
```

- The constructor (`Environment.cxx:21`): `El::Environment(argc, argv)`, installs the
  SIGTERM handler, `MPI_Comm_split_type(COMM_WORLD, MPI_COMM_TYPE_SHARED)`, computes node
  indices (collective), and reads MemUsed on node rank 0.
- `set_precision` is static: `El::gmp::SetPrecision(digits2)` plus
  `Boost_Float::default_precision(floor(Precision()*log10(2)))`. GMP rounds the value up to a
  multiple of the limb size; the actual value is `mpf_get_default_prec()`.

### 4.2 `Verbosity` — `sdpb_util/Verbosity.hxx`

```cpp
enum class Verbosity { none = 0, regular = 1, debug = 2, trace = 3 };
inline std::istream &operator>>(std::istream &in, Verbosity &value); // "0".."3" or names
inline std::ostream &operator<<(std::ostream &os, const Verbosity verbosity);
```

### 4.3 Timers — `sdpb_util/Timers/Timers.hxx`, `Timer.hxx`

```cpp
struct Timer
{
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time, stop_time;
  Timer();                                     // starts running
  void stop();
  [[nodiscard]] int64_t elapsed_milliseconds() const;
  [[nodiscard]] int64_t elapsed_seconds() const;
  [[nodiscard]] bool is_running() const;
};
std::ostream &operator<<(std::ostream &os, const Timer &timer);

struct Timers
{
  Timers();                                    // verbosity = regular, no env
  Timers(const Environment &env, const Verbosity &verbosity);
  ~Timers() noexcept;                          // prints peak MemUsed at debug verbosity
  void write_profile(const std::filesystem::path &path) const;
  [[nodiscard]] int64_t elapsed_milliseconds(const std::string &s) const;
};

struct Scoped_Timer : boost::noncopyable      // RAII; appends "name." to the prefix
{
  Scoped_Timer(Timers &timers, const std::string &name);
  virtual ~Scoped_Timer();
  [[nodiscard]] std::chrono::time_point<std::chrono::high_resolution_clock> start_time() const;
  void stop();
  [[nodiscard]] const Timer &timer() const;
  [[nodiscard]] int64_t elapsed_milliseconds() const;
};
```

Nearly every API takes a `Timers &`; pass any instance. Scoped timers must be strictly
nested. Timer names form dotted paths such as
`sdpb.solve.run.iter_2.step.initializeSchurComplementSolver.Q.syrk`.

### 4.4 Errors — `sdpb_util/assert.hxx`

```cpp
#define THROW(exception_type, ...)   // message: function, file:line, args, boost::stacktrace
#define RUNTIME_ERROR(...)           // THROW(std::runtime_error, ...)
#define LOGIC_ERROR(...)             // THROW(std::logic_error, ...)
#define ASSERT(condition, ...)       // RUNTIME_ERROR on failure (always on, not NDEBUG-gated)
#define ASSERT_EQUAL(a, b, ...)
#define DEBUG_STRING(expr)           // " expr='value' "
#define PRINT_WARNING(...)           // std::cerr << "Warning: ..."
```

### 4.5 Numbers, matrices and conversions

```cpp
// sdpb_util/Boost_Float.hxx
using Boost_Float = boost::multiprecision::mpfr_float;
std::string to_string(const Boost_Float &boost_float);
Boost_Float to_Boost_Float(const El::BigFloat &alpha);
El::BigFloat to_BigFloat(const Boost_Float &value);
std::vector<El::BigFloat> to_BigFloat_Vector(const std::vector<Boost_Float> &input);
std::vector<Boost_Float> to_Boost_Float_Vector(const std::vector<El::BigFloat> &input);

// sdpb_util/Damped_Rational.hxx   —   constant * base^x / prod_i (x - poles[i])
struct Damped_Rational
{
  Boost_Float constant, base;
  std::vector<Boost_Float> poles;
  bool is_constant() const;                   // poles.empty() && base == 1
  [[nodiscard]] Boost_Float evaluate(const Boost_Float &x,
                                     const Boost_Float &min_pole_distance = 0) const;
};
inline void swap(Damped_Rational &a, Damped_Rational &b) noexcept;
inline std::ostream &operator<<(std::ostream &os, const Damped_Rational &damped); // JSON-ish

// sdpb_util/Simple_Matrix.hxx  — fixed-size, column-major, bounds-checked
template <class T> class Simple_Matrix
{
public:
  Simple_Matrix(const size_t height, const size_t width);
  Simple_Matrix();                                         // 0x0
  explicit Simple_Matrix(const std::vector<std::vector<T>> &rows);
  [[nodiscard]] size_t Height() const;
  [[nodiscard]] size_t Width() const;
  T &operator()(const size_t i, const size_t j);
  const T &operator()(const size_t i, const size_t j) const;
};
// NB: height/width are const members, so a Simple_Matrix cannot be copy-assigned.

// sdpb_util/to_matrix.hxx
template <class T, class U>
El::Matrix<T> to_matrix(const std::vector<std::vector<U>> &elements,
                        std::function<T(const U &)> convert) noexcept(false);
template <class T>
El::Matrix<T> to_matrix(const std::vector<std::vector<T>> &elements) noexcept(false);

// sdpb_util/copy_matrix.hxx
// source: same Matrix copied over all ranks of destination.DistComm()
void copy_matrix(const El::Matrix<El::BigFloat> &source,
                 El::DistMatrix<El::BigFloat> &destination);
void copy_matrix(const El::DistMatrix<El::BigFloat, El::STAR, El::STAR> &source,
                 El::Matrix<El::BigFloat> &destination);
void copy_matrix(const El::DistMatrix<El::BigFloat, El::STAR, El::STAR> &source,
                 El::DistMatrix<El::BigFloat> &destination);
// source: valid on comm.Rank()==0 only; comm must equal destination.DistComm()
void copy_matrix_from_root(const El::Matrix<El::BigFloat> &source,
                           El::DistMatrix<El::BigFloat> &destination,
                           const El::mpi::Comm &comm);

// sdpb_util/write_distmatrix.hxx  — text: "height width\n" then rows
template <class T> void write_matrix(const El::Matrix<T> &matrix, const std::filesystem::path &path);
template <class T> void write_distmatrix(const El::AbstractDistMatrix<T> &A,
                                         const std::filesystem::path &path);

// sdpb_util/fill_weights.hxx  — rebuild z (length N+1) from y (length N) with n.z = 1
inline void fill_weights(const El::Matrix<El::BigFloat> &y, const size_t &max_index,
                         const std::vector<El::BigFloat> &normalization,
                         std::vector<El::BigFloat> &weights);   // weights pre-sized N+1

// sdpb_util/split_range.hxx
std::vector<El::Range<El::Int>> inline split_range(const El::Range<El::Int> &range,
                                                   size_t split_factor);

// sdpb_util/cholesky_condition_number.hxx  — (max diag / min diag)^2, collective on DistComm
template <class T> T cholesky_condition_number(const El::AbstractDistMatrix<T> &matrix);
```

### 4.6 Memory — `sdpb_util/memory_estimates.hxx`, `Proc_Meminfo.hxx`, `Shared_Window_Array.hxx`

```cpp
size_t get_max_shared_memory_bytes(size_t nonshared_memory_required_per_node_bytes,
                                   const Environment &env, Verbosity verbosity);
size_t bigfloat_bytes();
size_t get_heap_allocated_bytes(const El::BigFloat &f);
size_t get_heap_allocated_bytes(const El::AbstractDistMatrix<El::BigFloat> &m);
size_t get_heap_allocated_bytes(const El::Matrix<El::BigFloat> &m);
template <class T> size_t get_heap_allocated_bytes(const std::vector<T> &vec);
template <class T, std::size_t N> size_t get_heap_allocated_bytes(const std::array<T, N> &arr);
template <class T> size_t get_allocated_bytes(const T &value);
void print_allocation_message_per_node(const Environment &env, const std::string &name,
                                       size_t bytes);

struct Proc_Meminfo
{
  const size_t mem_total, mem_available;       // bytes
  [[nodiscard]] size_t mem_used() const;       // total - available
  static Proc_Meminfo read() noexcept(false);
  static Proc_Meminfo try_read(bool &result, bool print_error_msg = false) noexcept;
};

template <class T> class Shared_Window_Array    // MPI_Win_allocate_shared, all memory on node rank 0
{
public:
  MPI_Win win{}; El::mpi::Comm comm; T *data; size_t size = 0;
  Shared_Window_Array() = default;
  Shared_Window_Array(El::mpi::Comm shared_memory_comm, size_t size);  // collective
  ~Shared_Window_Array();
  void Fence() const;
  T &operator[](size_t index);
  const T &operator[](size_t index) const;
};
```

### 4.7 Block mapping — `sdpb_util/block_mapping/`

```cpp
struct Block_Cost { size_t cost, index; Block_Cost(const size_t &Cost, const size_t &Index); };
// cost 0 is clamped to 1; ordered by (cost, index)

struct Block_Map
{
  size_t num_procs = 0; size_t cost = 0; std::vector<size_t> block_indices;
  Block_Map(); Block_Map(const size_t &Num_procs, const size_t &Cost,
                         const std::initializer_list<size_t> &Block_indices);
  void clear();
  bool operator<(const Block_Map &b) const;   // by average cost
};

inline std::vector<std::vector<Block_Map>>     // [node][map]
compute_block_grid_mapping(const size_t &procs_per_node, const size_t &num_nodes,
                           std::vector<Block_Cost> block_costs);

inline void create_mpi_block_mapping_groups(
  const std::vector<std::vector<Block_Map>> &mapping,
  const El::mpi::Comm &node_comm, const int node_index,
  El::mpi::Group &mpi_group, El::mpi::Comm &mpi_group_comm,
  std::vector<size_t> &block_indices);

struct MPI_Comm_Wrapper  { El::mpi::Comm value;  /* non-copyable; frees in dtor unless COMM_WORLD */ };
struct MPI_Group_Wrapper { El::mpi::Group value; /* non-copyable; frees in dtor */ };
inline void swap(MPI_Comm_Wrapper &a, MPI_Comm_Wrapper &b) noexcept;
```

### 4.8 Archives, serialization, JSON

```cpp
// sdpb_util/Archive_Reader.hxx  — libarchive streambuf; iterate entries of zip/tar/...
struct Archive_Reader : public std::streambuf
{
  archive_entry *entry_ptr; bool entry_is_valid = false;
  explicit Archive_Reader(const std::filesystem::path &filename);
  bool next_entry();                           // then: std::istream s(&reader);
  int underflow() override;
};

// sdpb_util/boost_serialization.hxx
// Boost.Serialization for El::BigFloat (class version 1: zero flag + Serialize() bytes)
// and El::Matrix<Ring> (height, width, ldim, raw buffer). Tracking disabled.

// sdpb_util/json/parse_json.hxx — drive a RapidJSON SAX parser
template <class TParser> void parse_json(std::istream &input_stream, TParser &parser,
                                         const std::filesystem::path &input_path = {});
template <class TParser> void parse_json(const std::filesystem::path &input_path, TParser &parser);

// sdpb_util/json/Json_Writer.hxx — RapidJSON writer with full-precision BigFloat output
template <class TBaseWriter> class Json_BigFloat_Writer : public TBaseWriter
{
public:
  template <class... TArgs> explicit Json_BigFloat_Writer(TArgs &&...args);
  auto BigFloat(const El::BigFloat &value);
  auto BigFloat(const Boost_Float &value);
  template <class TFloat> auto BigFloatArray(const std::vector<TFloat> &arr);
};
using Json_Writer = Json_BigFloat_Writer<rapidjson::Writer<rapidjson::OStreamWrapper>>;
using Json_PrettyWriter = Json_BigFloat_Writer<rapidjson::PrettyWriter<rapidjson::OStreamWrapper>>;
```

The SAX parser framework in `sdpb_util/json/` is for writing new readers:
`Abstract_Json_Reader_Handler` (maps RapidJSON callbacks to virtual `json_*` methods),
`Abstract_Json_Element_Parser<TValue>`, `Abstract_Json_Object_Parser<TResult>`
(`get_result`, `reset_element_parsers(bool skip)`, `clear_result`),
`Abstract_Json_Array_Parser_With_Skip<TResult, TElementParser>`,
`Json_Vector_Parser<TElementParser>`, `Json_Vector_Parser_With_Skip<TElementParser>`
(element-level skip predicate, so a rank can ignore blocks it doesn't own),
`Json_Matrix_Parser<TMatrixElementParser>`, `Json_Float_Parser<TFloat>`,
`Json_Int64_Parser`, `Json_UInt64_Parser`, `Json_String_Parser`,
`Json_Damped_Rational_Parser`, `Json_Skip_Element_Parser`,
`Vector_Parse_Result_With_Skip<TValue>`. Each parser is constructed with an `on_parsed`
callback that receives the finished value. The legacy libxml2 counterparts are
`Number_State<Float_Type>` and `Vector_State<T>`.

### 4.9 Miscellaneous

```cpp
// sdpb_util/Mesh.hxx — adaptive 1-D mesh (outer_limits)
struct Mesh
{
  std::array<El::BigFloat, 5> x, f; std::unique_ptr<Mesh> lower, upper;
  Mesh(const El::BigFloat &x_0, const El::BigFloat &x_2, const El::BigFloat &x_4,
       const El::BigFloat &f_0, const El::BigFloat &f_2, const El::BigFloat &f_4,
       const std::function<El::BigFloat(const El::BigFloat &x)> &fn,
       const El::BigFloat &mesh_threshold, const El::BigFloat &block_epsilon);
  Mesh(const El::BigFloat &x_0, const El::BigFloat &x_4,
       const std::function<El::BigFloat(const El::BigFloat &x)> &fn,
       const El::BigFloat &mesh_threshold, const El::BigFloat &block_epsilon);
};
std::ostream &operator<<(std::ostream &os, const Mesh &mesh);

// sdpb_util/ostream/*.hxx
// operator<< for std::array, std::map, std::pair, std::set, std::vector
inline std::string pretty_print_bytes(const size_t bytes, const bool also_print_exact = false);
inline void set_stream_precision(std::ostream &os);  // digits for current GMP precision
```

`sdpb_util/flint.hxx` is an include shim for different FLINT versions.

---

## 5. `pmp` — polynomial matrix program model

### 5.1 `Polynomial` — `pmp/Polynomial.hxx`

```cpp
using Boost_Polynomial = boost::math::tools::polynomial<Boost_Float>;

class Polynomial
{
public:
  std::vector<El::BigFloat> coefficients;       // a_0, a_1, ..., a_n (increasing degree)
  Polynomial();                                 // zero polynomial, coefficients = {0}
  Polynomial(const size_t &size, const El::BigFloat &default_element);
  int64_t degree() const;                       // coefficients.size() - 1
  El::BigFloat operator()(const El::BigFloat &x) const;  // Horner
  friend bool operator==(const Polynomial &lhs, const Polynomial &rhs);
  friend bool operator!=(const Polynomial &lhs, const Polynomial &rhs);
  friend std::ostream &operator<<(std::ostream &os, const Polynomial &p);
};
inline Polynomial operator/(const Polynomial &a, const El::BigFloat &b);
using Polynomial_Vector = std::vector<Polynomial>;

// move-in helpers from nested coefficient vectors:
inline void swap(Polynomial_Vector &, std::vector<std::vector<El::BigFloat>> &);
inline void swap(std::vector<Polynomial_Vector> &,
                 std::vector<std::vector<std::vector<El::BigFloat>>> &);
inline void swap(std::vector<std::vector<Polynomial_Vector>> &,
                 std::vector<std::vector<std::vector<std::vector<El::BigFloat>>>> &);
```

### 5.2 `Polynomial_Vector_Matrix` — `pmp/Polynomial_Vector_Matrix.hxx`

One positivity constraint `(1,y)·M(x) ⪰ 0` for `x ≥ 0`, i.e. one SDP block.

```cpp
struct Polynomial_Vector_Matrix
{
  Simple_Matrix<Polynomial_Vector> polynomials;   // dim x dim, symmetric; each entry N+1 polys
  Damped_Rational prefactor;
  Damped_Rational reduced_prefactor;
  std::vector<El::BigFloat> sample_points;
  std::vector<El::BigFloat> sample_scalings;
  std::vector<El::BigFloat> reduced_sample_scalings;
  std::array<Polynomial_Vector, 2> bilinear_basis; // [0]: even, [1]: odd (times sqrt(x))

  Polynomial_Vector_Matrix(
    const Simple_Matrix<Polynomial_Vector> &polynomials,
    const std::optional<Damped_Rational> &prefactor_opt,
    const std::optional<Damped_Rational> &reduced_prefactor_opt,
    const std::optional<int64_t> &max_num_poles_opt,
    const std::optional<std::vector<El::BigFloat>> &sample_points_opt,
    const std::optional<std::vector<El::BigFloat>> &sample_scalings_opt,
    const std::optional<std::vector<El::BigFloat>> &reduced_sample_scalings_opt,
    const std::optional<std::array<Polynomial_Vector, 2>> &bilinear_basis_opt) noexcept(false);

  void validate(int64_t num_points) const;
};
```

Defaults filled in by the constructor (`Polynomial_Vector_Matrix.cxx:126-196`):

- `prefactor`: `e^{-x}` (constant 1 if the block has degree 0).
- `reduced_prefactor`: `prefactor`, keeping only the `max_num_poles` rightmost poles if
  `max_num_poles_opt` is set (negative means no limit).
- `num_points = max_degree + 1 + #reduced_poles − #poles`.
- `sample_points`: analytic points for `reduced_prefactor` (`sample_points()`, §5.5).
- `sample_scalings`: `prefactor` evaluated at the points; `reduced_sample_scalings`
  likewise for `reduced_prefactor` (equal to `sample_scalings` if the prefactors match).
- `bilinear_basis`: orthogonal polynomials for the points and reduced scalings
  (`bilinear_basis()`, §5.5); user-supplied bases are truncated to the required sizes.
- `validate()` checks sizes and that `polynomials` is symmetric.

### 5.3 `Polynomial_Matrix_Program` — `pmp/Polynomial_Matrix_Program.hxx`

Manual eq. (3.1): maximize `a·z` subject to `n·z = 1` and `Σ_n z_n M_n^j(x) ⪰ 0`.

```cpp
struct Polynomial_Matrix_Program
{
  std::vector<El::BigFloat> objective;                     // a_0..a_N
  std::optional<std::vector<El::BigFloat>> normalization;  // n_0..n_N
  size_t num_matrices = 0;                                 // global
  std::vector<Polynomial_Vector_Matrix> matrices;          // rank-local subset
  std::vector<size_t> matrix_index_local_to_global;        // each in [0, num_matrices)
  std::vector<std::filesystem::path> block_paths;          // label per local matrix

  Polynomial_Matrix_Program(
    std::vector<El::BigFloat> objective,
    std::optional<std::vector<El::BigFloat>> normalization,
    size_t num_matrices, std::vector<Polynomial_Vector_Matrix> matrices,
    std::vector<size_t> matrix_index_local_to_global,
    std::vector<std::filesystem::path> block_paths);       // validates sizes
  // move-only
};
```

The global indices across all ranks must be unique and cover `[0, num_matrices)`. The
constructor doesn't check this; `write_sdp` does, indirectly.

### 5.4 `PMP_Info` / `PVM_Info` — `pmp/PMP_Info.hxx`

Per-block metadata written to `pmp_info.json` and consumed by `spectrum`.

```cpp
struct PVM_Info
{
  int block_index = -1; std::filesystem::path block_path; size_t dim = 0;
  Damped_Rational prefactor, reduced_prefactor;
  std::vector<El::BigFloat> sample_points, sample_scalings, reduced_sample_scalings;
  PVM_Info();
  PVM_Info(const Polynomial_Vector_Matrix &pvm, int block_index, std::filesystem::path block_path);
  void validate() const;
  void validate(const size_t num_blocks) const;
};
struct PMP_Info
{
  size_t num_blocks; std::vector<PVM_Info> blocks;   // blocks: rank-local
  explicit PMP_Info(const size_t num_blocks, const std::vector<PVM_Info> &blocks);
  explicit PMP_Info(const Polynomial_Matrix_Program &pmp);
};

// pmp/max_normalization_index.hxx
inline size_t max_normalization_index(const std::vector<El::BigFloat> &normalization); // argmax |n_i|
```

### 5.5 🔒 Sampling and bases (no header)

Declared inside `pmp/Polynomial_Vector_Matrix.cxx:9-16`.

```cpp
// pmp/convert/sample_points.cxx:179 — analytic points minimising interpolation error on [0,inf)
std::vector<Boost_Float>
sample_points(const size_t &num_points, const Damped_Rational &prefactor)

// pmp/convert/sample_scalings.cxx:5 — prefactor at each point (|x - pole| clamped to 1e-16)
std::vector<Boost_Float>
sample_scalings(const std::vector<Boost_Float> &points,
                const Damped_Rational &damped_rational)

// pmp/convert/bilinear_basis/bilinear_basis.cxx:72 — orthogonal polynomials for
// the discrete measure sum_k scaling_k delta(x - point_k); warns if ill-conditioned
std::array<Polynomial_Vector, 2>
bilinear_basis(const std::vector<El::BigFloat> &sample_points,
               const std::vector<El::BigFloat> &sample_scalings)
```

---

## 6. `pmp_read` — PMP readers

### 6.1 Public — `pmp_read/pmp_read.hxx`

```cpp
// Collective. Distributes files/matrices over ranks; returns the rank-local PMP.
Polynomial_Matrix_Program read_polynomial_matrix_program(
  const Environment &env,
  const std::vector<std::filesystem::path> &input_files, int64_t max_num_poles,
  const Verbosity &verbosity, Timers &timers);

Polynomial_Matrix_Program
read_polynomial_matrix_program(const Environment &env,
                               const std::filesystem::path &input_file,
                               int64_t max_num_poles,
                               const Verbosity &verbosity, Timers &timers);

std::vector<std::filesystem::path>
read_nsv_file_list(const std::filesystem::path &input_file);   // '\0'-separated, relative to the .nsv

std::vector<std::filesystem::path>
collect_files_expanding_nsv(const std::filesystem::path &input_file);  // recursive

std::vector<std::filesystem::path> collect_files_expanding_nsv(
  const std::vector<std::filesystem::path> &input_files);
```

Formats are dispatched by extension: `.json`, `.m` (Mathematica), `.xml` (legacy; no
prefactor or normalization) and `.nsv` (file list). `max_num_poles < 0` means no limit.
Objective and normalization are broadcast and checked for consistency across files.

### 6.2 Single-file parsing — `pmp_read/PMP_File_Parse_Result.hxx`

```cpp
struct PMP_File_Parse_Result
{
  std::optional<std::vector<El::BigFloat>> objective;
  std::optional<std::vector<El::BigFloat>> normalization;
  size_t num_matrices = 0;                                  // total in file
  std::map<size_t, Polynomial_Vector_Matrix> parsed_matrices; // index -> matrix (selected only)

  static void validate(const PMP_File_Parse_Result &result);
  static PMP_File_Parse_Result
  read(const std::filesystem::path &input_path, int64_t max_num_poles,
       bool should_parse_objective, bool should_parse_normalization,
       const std::function<bool(size_t matrix_index)> &should_parse_matrix);
  // move-only
};
```

This is the non-collective way to parse one file on one rank. Only the JSON reader honours
`should_parse_objective`/`should_parse_normalization`.

### 6.3 🔒 Format readers (no header)

Declared in `pmp_read/PMP_File_Parse_Result.cxx:7-16`.

```cpp
// pmp_read/read_json/read_json.cxx:6
PMP_File_Parse_Result
read_json(const std::filesystem::path &input_path, const int64_t max_num_poles,
          const bool should_parse_objective,
          const bool should_parse_normalization,
          const std::function<bool(size_t matrix_index)> &should_parse_matrix)

// pmp_read/read_mathematica/read_mathematica.cxx:17
PMP_File_Parse_Result read_mathematica(
  const std::filesystem::path &input_path,const int64_t max_num_poles,
  const std::function<bool(size_t matrix_index)> &should_parse_matrix)

// pmp_read/read_xml/read_xml.cxx:50
PMP_File_Parse_Result
read_xml(const std::filesystem::path &input_file, const int64_t max_num_poles,
         const std::function<bool(size_t matrix_index)> &should_parse_matrix)

// pmp_read/read_mathematica/parse_SDP/parse_SDP.cxx:17 — parses SDP[obj, norm, {matrices}]
const char *
parse_SDP(const char *begin, const char *end,const int64_t max_num_poles,
          const std::function<bool(size_t matrix_index)> &should_parse_matrix,
          std::optional<std::vector<El::BigFloat>> &objectives,
          std::optional<std::vector<El::BigFloat>> &normalization,
          size_t &num_matrices,
          std::map<size_t, Polynomial_Vector_Matrix> &parsed_matrices)

// pmp_read/read_mathematica/parse_SDP/parse_matrices.cxx:10
const char *parse_matrices(
  const char *begin, const char *end, const int64_t max_num_poles,
  const std::function<bool(size_t matrix_index)> &should_parse_matrix,
  size_t &num_matrices,
  std::map<size_t, Polynomial_Vector_Matrix> &parsed_matrices)

// pmp_read/read_mathematica/parse_SDP/parse_matrix/parse_damped_rational.cxx:10
const char *parse_damped_rational(const char *begin, const char *end,
                                  Damped_Rational &damped_rational)
```

The Mathematica scanner helpers with headers are in `read_mathematica/parse_SDP/`:
`parse_number`, `parse_vector<T>`, `parse_generic`, `parse_matrix`, `parse_polynomial`
and `is_valid_char`. The parser classes are `Json_PMP_Parser`,
`Json_Positive_Matrix_With_Prefactor_Parser`, `Json_Polynomial_Parser`, `Xml_Parser` and
`Xml_Polynomial_Vector_Matrix_State`.

---

## 7. `pmp2sdp_lib` — PMP → SDP conversion and writers

### 7.1 `Block_File_Format` — `pmp2sdp/Block_File_Format.hxx`

```cpp
enum class Block_File_Format { bin, json };
inline std::istream &operator>>(std::istream &in, Block_File_Format &format);  // "bin"/"json"
inline std::ostream &operator<<(std::ostream &out, const Block_File_Format &format);
```

### 7.2 `Dual_Constraint_Group` — `pmp2sdp/Dual_Constraint_Group.hxx`

One sampled SDP block. Its layout is exactly what `block_data_<i>.{bin,json}` stores.

```cpp
class Dual_Constraint_Group
{
public:
  size_t block_index{}; size_t dim{}; size_t num_points{};
  El::Matrix<El::BigFloat> constraint_matrix;           // B block, P' x N
  std::vector<El::BigFloat> constraint_constants;       // c block, length P'
  std::array<El::Matrix<El::BigFloat>, 2> bilinear_bases; // sampled even/odd bases
  Dual_Constraint_Group() = default;
  Dual_Constraint_Group(const size_t &Block_index, const Polynomial_Vector_Matrix &m);
};
```

`P' = num_points · dim(dim+1)/2`. Rows are indexed by `(r ≤ s, k)`, with
`c_p = s_k P^{rs}_0(x_k)` and `B_{p,n} = −s_k P^{rs}_{n+1}(x_k)`. The PVM passed in must
already have the normalization eliminated; `Output_SDP` does that.

🔒 `pmp2sdp/Dual_Constraint_Group/sample_bilinear_basis.cxx:36` (declared in
`Dual_Constraint_Group.cxx:14`):

```cpp
std::array<El::Matrix<El::BigFloat>, 2>
sample_bilinear_basis(const std::array<Polynomial_Vector, 2> &bilinear_basis,
                      const std::vector<El::BigFloat> &sample_points,
                      const std::vector<El::BigFloat> &sample_scalings)
// plus the single-basis overload at :18:
El::Matrix<El::BigFloat>
sample_bilinear_basis(const Polynomial_Vector &bilinearBasis,
                      const std::vector<El::BigFloat> &samplePoints,
                      const std::vector<El::BigFloat> &sampleScalings)
```

### 7.3 `Output_SDP` — `pmp2sdp/Output_SDP/Output_SDP.hxx`

Converts Manual (3.1) to (2.2) by eliminating the normalization. It is rank-local and not
collective.

```cpp
struct Output_SDP : boost::noncopyable
{
  El::BigFloat objective_const;                            // b_0
  std::vector<El::BigFloat> dual_objective_b;              // b_1..b_N
  std::optional<std::vector<El::BigFloat>> normalization;
  size_t num_blocks = 0;
  std::vector<Dual_Constraint_Group> dual_constraint_groups; // rank-local
  std::vector<std::string> command_arguments;              // stored in control.json
  Output_SDP(const Polynomial_Matrix_Program &pmp,
             const std::vector<std::string> &command_arguments, Timers &timers);
};
```

With a non-trivial normalization and `k = argmax|n|`: `objective_const = a_k/n_k`,
`b_i = a_i − n_i·a_k/n_k` for `i ≠ k`, and each polynomial vector becomes
`P'_0 = P_k/n_k`, `P'_i = P_i − n_i P'_0`.

### 7.4 `write_sdp` — `pmp2sdp/write_sdp.hxx`

```cpp
void write_sdp(const std::filesystem::path &output_path, const Output_SDP &sdp,
               const Polynomial_Matrix_Program &pmp,
               Block_File_Format block_file_format, bool zip, Timers &timers,
               Verbosity verbosity);
```

This call is collective. It writes into `<output_path>_temp` (or packs a zip), checks
that every block was written exactly once, then renames atomically. It writes:
- `control.json`
- `objectives.json`
- `pmp_info.json`
- `normalization.json` (only if the PMP has a normalization)
- `block_info_<i>.json`
- `block_data_<i>.{bin,json}`

⚠️ `output_path` must be identical on all ranks, and must be on a shared filesystem when
running on several nodes.

### 7.5 🔒 Individual writers (no header)

Declared in `pmp2sdp/write_sdp.cxx:19-37`. Useful for writing blocks to arbitrary streams,
e.g. a `std::stringstream`.

```cpp
// pmp2sdp/write_block_data.cxx:105 — bin: boost binary archive
// [precision, B, c, bases_even, bases_odd]; json: {bilinear_bases_even, bilinear_bases_odd, c, B}
void write_block_data(std::ostream &os, const Dual_Constraint_Group &group,
                      Block_File_Format format)

// pmp2sdp/write_block_info_json.cxx:5 — {"dim":..,"num_points":..}
void write_block_info_json(std::ostream &output_stream,
                           const Dual_Constraint_Group &group)

// pmp2sdp/write_objectives_json.cxx:7 — {"constant":..,"b":[..]}
void write_objectives_json(std::ostream &output_stream,
                           const El::BigFloat &objective_const,
                           const std::vector<El::BigFloat> &dual_objective_b)

// pmp2sdp/write_normalization_json.cxx:7
void write_normalization_json(std::ostream &output_stream,
                           const std::vector<El::BigFloat> &normalization)

// pmp2sdp/write_control_json.cxx:21 — {"num_blocks":..,"command":".."}
// ⚠️ declared as returning size_t in write_sdp.cxx:19 but defined void — declare it void
void write_control_json(std::ostream &output_stream, const size_t &num_blocks,
                        const std::vector<std::string> &command_arguments)

// pmp2sdp/write_sdp.cxx:410 (file-local use)
void print_matrix_sizes(
  const int &rank, const std::vector<El::BigFloat> &dual_objective_b,
  const std::vector<Dual_Constraint_Group> &dual_constraint_groups)
```

### 7.6 Other headers

```cpp
// pmp2sdp/write_pmp_info_json.hxx
inline void write_pmp_info_json(std::ostream &output_stream,
                                const std::vector<PVM_Info> &pmp_info);   // rank 0 only (asserted)
inline void synchronize_pvm_info(PVM_Info &pvm_info, const int from);    // p2p send to rank 0
inline std::vector<PVM_Info> synchronize_pmp_info_blocks(const PMP_Info &pmp_info); // gather on rank 0

// pmp2sdp/write_vector.hxx — JSON array writers
template <typename T> inline void write_vector(std::ostream &output_stream, const std::vector<T> &v, ...);
inline void write_vector(std::ostream &output_stream, const std::vector<size_t> &v);

// pmp2sdp/Archive_Writer.hxx, Archive_Entry.hxx — libarchive zip writer (store, no compression)
struct Archive_Writer
{
  explicit Archive_Writer(const std::filesystem::path &filename);
  ~Archive_Writer();
  void write_entry(const Archive_Entry &entry, std::istream &stream);
};
struct Archive_Entry { Archive_Entry(const std::filesystem::path &filename, const int64_t &num_bytes); };

// pmp2sdp/byte_counter.hxx — boost::iostreams filter counting written bytes

// pmp2sdp/Pmp2sdp_Parameters/Pmp2sdp_Parameters.hxx (executable only, argv-only constructor)
struct Pmp2sdp_Parameters
{
  int precision; int64_t max_num_poles; std::filesystem::path input_file, output_path;
  Block_File_Format output_format; bool zip = false; Verbosity verbosity;
  std::vector<std::string> command_arguments;
  Pmp2sdp_Parameters(int argc, char **argv);
  [[nodiscard]] bool is_valid() const;
};
```

---

## 8. `sdp_solve` — SDP data and the solver

`sdp_solve/sdp_solve.hxx` is the umbrella header:

```cpp
#include "Block_Info.hxx"
#include "SDP_Solver.hxx"
#include "Write_Solution.hxx"
#include "read_text_block.hxx"
El::BigFloat dot(const Block_Vector &A, const Block_Vector &B);   // run/compute_objectives/dot.cxx:4
```

### 8.1 `Block_Info` — `sdp_solve/Block_Info.hxx`

Global block sizes plus this rank's block assignment and MPI group.

```cpp
class Block_Info
{
public:
  std::filesystem::path block_timings_filename;   // where costs came from ("" = estimated)
  std::vector<size_t> dimensions;                  // m_j, global
  std::vector<size_t> num_points;                  // d_j + 1, global
  std::vector<size_t> block_indices;               // global indices of local blocks
  MPI_Group_Wrapper mpi_group;
  MPI_Comm_Wrapper mpi_comm;                        // communicator of this rank's group

  Block_Info() = delete;
  // From an SDP dir/zip; costs from checkpoint_in/block_timings, sdp_path/block_timings,
  // or a memory estimate (which reads objectives.json)
  Block_Info(const Environment &env, const std::filesystem::path &sdp_path,
             const std::filesystem::path &checkpoint_in,
             const size_t &proc_granularity, const Verbosity &verbosity);
  // From an SDP dir/zip with measured per-block costs (block_timings.Height() == #blocks)
  Block_Info(const Environment &env, const std::filesystem::path &sdp_path,
             const El::Matrix<int32_t> &block_timings,
             const size_t &proc_granularity, const Verbosity &verbosity);
  // In-memory: dimensions and num_points for every block; costs = schur_block_size^2
  Block_Info(const Environment &env,
             const std::vector<size_t> &matrix_dimensions,
             const std::vector<size_t> &matrix_num_points,
             const size_t &proc_granularity, const Verbosity &verbosity);
  // In-memory: num_points = 1 for every block (delegates to the constructor above)
  Block_Info(const Environment &env,
             const std::vector<size_t> &matrix_dimensions,
             const size_t &proc_granularity, const Verbosity &verbosity);
  Block_Info(const Environment &env,
             const std::vector<size_t> &matrix_dimensions,
             const Verbosity &verbosity);      // proc_granularity = 1

  void read_block_info(const std::filesystem::path &sdp_path);   // fills dimensions/num_points
  std::vector<Block_Cost>
  read_block_costs(const std::filesystem::path &sdp_path,
                   const std::filesystem::path &checkpoint_in, const Environment &env);
  void allocate_blocks(const Environment &env, const std::vector<Block_Cost> &block_costs,
                       const size_t &proc_granularity, const Verbosity &verbosity);

  [[nodiscard]] size_t get_schur_block_size(const size_t index) const;   // num_points*dim*(dim+1)/2
  [[nodiscard]] std::vector<size_t> schur_block_sizes() const;
  [[nodiscard]] size_t get_bilinear_pairing_block_size(const size_t index, const size_t parity) const; // num_points*dim
  [[nodiscard]] std::vector<size_t> bilinear_pairing_block_sizes() const;  // 2 per block
  [[nodiscard]] size_t get_psd_matrix_block_size(const size_t index, const size_t parity) const;
  [[nodiscard]] std::vector<size_t> psd_matrix_block_sizes() const;        // 2 per block
  [[nodiscard]] size_t get_bilinear_bases_height(const size_t index, const size_t parity) const;
  [[nodiscard]] size_t get_bilinear_bases_width(const size_t index, const size_t /*parity*/) const;
};
inline void swap(Block_Info &a, Block_Info &b) noexcept;
```

- All constructors are collective, because they create the MPI groups.
- `allocate_blocks` asserts that every rank got at least one block
  (`allocate_blocks.cxx:73`). Expensive blocks are shared by several ranks, so this holds
  in practice, but a tiny SDP on many ranks can trip it. ⚠️ If it does, run on fewer ranks.
- ⚠️ Calling `allocate_blocks` again overwrites `mpi_group`/`mpi_comm` without freeing
  the old ones.

### 8.2 Block containers

All of them hold `std::vector<El::DistMatrix<El::BigFloat>> blocks` for the local blocks
only, on the group grid. When the size vector has `2 × num_schur_blocks` entries, each
block contributes two entries (even/odd parity).

```cpp
// sdp_solve/Block_Vector.hxx — stacked column vectors
struct Block_Vector
{
  std::vector<El::DistMatrix<El::BigFloat>> blocks;
  Block_Vector(const std::vector<size_t> &block_heights, const std::vector<size_t> &block_indices,
               const size_t &num_schur_blocks, const El::Grid &grid);
  Block_Vector() = default;
};

// sdp_solve/Block_Matrix.hxx — stacked rectangular bands of common width
struct Block_Matrix
{
  std::vector<El::DistMatrix<El::BigFloat>> blocks;
  Block_Matrix(const std::vector<size_t> &block_heights, const size_t &width,
               const std::vector<size_t> &block_indices,
               const size_t &num_schur_blocks, const El::Grid &grid);
  Block_Matrix() = default;
};

// sdp_solve/Block_Diagonal_Matrix.hxx — square blocks
class Block_Diagonal_Matrix
{
public:
  std::vector<El::DistMatrix<El::BigFloat>> blocks;
  explicit Block_Diagonal_Matrix(const std::vector<size_t> &block_sizes,
                                 const std::vector<size_t> &block_indices,
                                 const size_t &num_schur_blocks, const El::Grid &grid);
  void add_block(const size_t &block_size, const El::Grid &grid);
  void set_zero();
  void add_diagonal(const El::BigFloat &c);
  void operator+=(const Block_Diagonal_Matrix &A);
  void operator-=(const Block_Diagonal_Matrix &A);
  void operator*=(const El::BigFloat &c);
  void symmetrize();                              // (M + M^T)/2
  [[nodiscard]] El::BigFloat max_abs() const;     // AllReduce over COMM_WORLD
  [[nodiscard]] El::BigFloat trace() const;       // AllReduce over COMM_WORLD
  friend std::ostream &operator<<(std::ostream &os, const Block_Diagonal_Matrix &A);
};

// sdp_solve/Index_Tuple.hxx — named (p, r, s, k)
class Index_Tuple { public: int p, r, s, k; Index_Tuple(int p, int r, int s, int k); Index_Tuple(); };
```

Free helpers with headers:

```cpp
// sdp_solve/lower_triangular_transpose_solve.hxx — v := L^{-T} v
void lower_triangular_transpose_solve(const Block_Diagonal_Matrix &L, Block_Vector &v);
// sdp_solve/SDP_Solver/run/step/compute_search_direction/lower_triangular_solve.hxx — B := L^{-1} B
template <class T> void lower_triangular_solve(const Block_Diagonal_Matrix &L_cholesky, T &B);
// sdp_solve/SDP_Solver/run/constraint_matrix_weighted_sum.hxx — Result = sum_p a_p A_p
void constraint_matrix_weighted_sum(const Block_Info &block_info, const SDP &sdp,
                                    const Block_Vector &a, Block_Diagonal_Matrix &Result);
```

### 8.3 Reading text blocks — `sdp_solve/read_text_block.hxx`

These read the format produced by `write_distmatrix` / `--writeSolution`.

```cpp
inline void set_element(El::DistMatrix<El::BigFloat> &block, const int64_t &row,
                        const int64_t &column, const El::BigFloat &element);
inline void set_element(El::Matrix<El::BigFloat> &block, const int64_t &row,
                        const int64_t &column, const El::BigFloat &element);
template <typename Matrix>
void read_text_block(Matrix &block, const std::filesystem::path &block_path);
template <typename Matrix>   // reads block_directory/(prefix + block_index + ".txt")
void read_text_block(Matrix &block, const std::filesystem::path &block_directory,
                     const std::string &prefix, const size_t &block_index);
```

### 8.4 `SDP` — `sdp_solve/SDP.hxx`

```
Dual:   maximize f + b·y  s.t.  Tr(A_p Y) + (B y)_p = c_p,  Y ⪰ 0
Primal: minimize f + c·x  s.t.  X = Σ_p A_p x_p − C,  Bᵀx = b,  X ⪰ 0
```

```cpp
struct SDP
{
  std::vector<El::DistMatrix<El::BigFloat>> bilinear_bases;  // 2 per local block
  std::vector<El::DistMatrix<El::BigFloat>> bases_blocks;    // 2 per local block (precomputed)
  Block_Matrix free_var_matrix;                              // B: local bands P_j x N
  Block_Vector primal_objective_c;                           // c
  El::DistMatrix<El::BigFloat> dual_objective_b;             // b (N x 1), on the group grid
  El::BigFloat objective_const;                              // f
  std::optional<std::vector<El::BigFloat>> normalization;    // length N+1, rank 0 only

  // From an SDP directory or archive (collective)
  SDP(const std::filesystem::path &sdp_path, const Block_Info &block_info,
      const El::Grid &grid, Timers &timers);
  // In-memory, single-point blocks only (outer_limits); inputs are GLOBAL, indexed by block
  SDP(const El::BigFloat &objective_const,
      const std::vector<std::vector<El::BigFloat>> &primal_objective_c_input,
      const std::vector<El::Matrix<El::BigFloat>> &free_var_input,
      const El::DistMatrix<El::BigFloat, El::STAR, El::STAR> &yp_to_y_star,
      const El::DistMatrix<El::BigFloat, El::STAR, El::STAR> &dual_objective_b_star,
      const std::vector<El::BigFloat> &normalization,
      const El::BigFloat &primal_c_scale, const Block_Info &block_info,
      const El::Grid &grid);
  // In-memory, from sampled Dual_Constraint_Groups (e.g. Output_SDP::dual_constraint_groups):
  // groups.at(i) must describe block block_info.block_indices.at(i); objective_const,
  // dual_objective_b and normalization must be identical on all ranks. Collective. See §3.5.
  SDP(const El::BigFloat &objective_const,
      const std::vector<El::BigFloat> &dual_objective_b,
      const std::vector<Dual_Constraint_Group> &groups,
      const std::optional<std::vector<El::BigFloat>> &normalization,
      const Block_Info &block_info, const El::Grid &grid);
private:
  void validate(const Block_Info &block_info) const noexcept(false);
};
```

Notes on the in-memory constructor (`SDP/SDP.cxx:38`):
- It sets bilinear bases to `[1]` (even) and a 0×1 matrix (odd), so every block is a
  `dim×dim` PSD matrix at one point. Use it with `Block_Info(env, dims, verbosity)`.
- It multiplies `c` and `B` by `primal_c_scale`.
- It replaces `B` with `B · yp_to_y` (pass the N×N identity for no change).
- The `[STAR,STAR]` inputs should live on a default `El::Grid()`.

Invariants checked by `validate` (`SDP/SDP.cxx:163`), for each local block `j` with
`P_j = get_schur_block_size(j)`:
- `c` is `P_j × 1` and `B` is `P_j × N`.
- `bilinear_bases[2i+parity]` is `get_bilinear_bases_height × get_bilinear_bases_width`.
- `bases_blocks[2i+parity]` is `get_psd_matrix_block_size × get_bilinear_pairing_block_size`.
- Every one of them is on `block_info.mpi_comm`.

Related (headers in `sdp_solve/SDP/`):

```cpp
// SDP/read_block_data/SDP_Block_Data.hxx — one parsed block_data file (local matrices)
struct SDP_Block_Data
{
  int block_index_local = -1;
  El::Matrix<El::BigFloat> constraint_matrix{};       // B
  El::Matrix<El::BigFloat> primal_objective_c{};      // c (column)
  std::array<El::Matrix<El::BigFloat>, 2> bilinear_bases{};
  std::array<El::Matrix<El::BigFloat>, 2> bases_blocks{};
  SDP_Block_Data() = default;
  SDP_Block_Data(std::istream &block_stream, Block_File_Format format,
                 size_t block_index_local, const Block_Info &block_info); // bin asserts precision
  // From an in-memory group; asserts group.block_index/dim/num_points match block_info
  SDP_Block_Data(const Dual_Constraint_Group &group, size_t block_index_local,
                 const Block_Info &block_info);
  // move-only
};
// Scatter one block into the DistMatrices of sdp. sdp_block_local needs to be valid only on
// grid.Comm().Rank() == 0; sdp's block containers must already be sized and
// sdp.dual_objective_b initialized. Collective over the group.
void set_sdp_from_root(const El::Grid &grid, const Block_Info &block_info,
                       const SDP_Block_Data &sdp_block_local, SDP &sdp);
// SDP/read_block_data/Block_Data_Parse_Result.hxx, Json_Block_Data_Parser.hxx — JSON block parser

// SDP/set_bases_blocks.hxx — expand a bilinear basis into its block-diagonal "bases block"
void set_bilinear_bases_block_local(const El::Matrix<El::BigFloat> &bilinear_base_local,
                                    El::Matrix<El::BigFloat> &bases_block_local);
void set_bilinear_bases_block(const El::Matrix<El::BigFloat> &bilinear_base_local,
                              El::DistMatrix<El::BigFloat> &bases_block);
void set_bases_blocks(const Block_Info &block_info,
                      const std::vector<El::Matrix<El::BigFloat>> &bilinear_bases_local,
                      std::vector<El::DistMatrix<El::BigFloat>> &bases_blocks,
                      const El::Grid &grid);
// SDP/assign_bilinear_bases_dist.hxx
void assign_bilinear_bases_dist(const std::vector<El::Matrix<El::BigFloat>> &bilinear_bases_local,
                                const El::Grid &grid,
                                std::vector<El::DistMatrix<El::BigFloat>> &bilinear_bases_dist);
```

🔒 Readers used by `SDP::SDP` (no header; declared in `SDP/SDP.cxx:15-22`):

```cpp
// SDP/read_objectives.cxx:39 — objectives.json → f, b (b on `grid`); also used by read_block_costs
void read_objectives(const fs::path &sdp_path, const El::Grid &grid,
                     El::BigFloat &objective_const,
                     El::DistMatrix<El::BigFloat> &dual_objective_b,
                     Timers &timers)

// SDP/read_normalization.cxx:32 — normalization.json, or {} if absent
std::optional<std::vector<El::BigFloat>>
read_normalization(const fs::path &sdp_path, Timers &timers)

// SDP/read_block_data/read_block_data.cxx:97 — fills c, B, bilinear_bases, bases_blocks
// of `sdp` for block_info.block_indices (sdp.dual_objective_b must already be sized)
void read_block_data(const fs::path &sdp_path, const El::Grid &grid,
                     const Block_Info &block_info, SDP &sdp, Timers &timers)
```

### 8.5 `SDP_Solver` — `sdp_solve/SDP_Solver.hxx`

```cpp
class SDP_Solver
{
public:
  Block_Vector x;                    // length P (Schur block sizes)
  Block_Diagonal_Matrix X;           // 2 PSD blocks per SDP block
  Block_Vector y;                    // length N, duplicated on every local block
  Block_Diagonal_Matrix Y;
  El::BigFloat primal_objective, dual_objective, duality_gap;
  Block_Diagonal_Matrix primal_residues;               // P = sum_p A_p x_p - X
  El::BigFloat primal_error_P, primal_error_p;         // |P|, |b - B^T x|
  El::BigFloat primal_error() const;                   // max of the two
  Block_Vector dual_residues;                          // d = c - Tr(A_* Y) - B y
  El::BigFloat dual_error;                             // max|d|
  El::BigFloat R_error;                                // max|mu I - XY|
  size_t num_iterations = 0;                           // iterations completed by the last run()
  int64_t current_generation;
  boost::optional<int64_t> backup_generation;

  // Allocates state, then load_checkpoint(parameters.checkpoint_in, ...);
  // if nothing was loaded: x = y = 0, X = initial_matrix_scale_primal*I, Y = initial_matrix_scale_dual*I
  SDP_Solver(const Solver_Parameters &parameters, const Verbosity &verbosity,
             const bool &require_initial_checkpoint,
             const Block_Info &block_info, const El::Grid &grid,
             const size_t &dual_objective_b_height);

  // Interior-point loop until a termination criterion fires. Collective.
  SDP_Solver_Terminate_Reason
  run(const Environment &env, const Solver_Parameters &parameters,
      const Verbosity &verbosity,
      const boost::property_tree::ptree &parameter_properties,   // copied into checkpoint.json
      const Block_Info &block_info, const SDP &sdp, const El::Grid &grid,
      const std::chrono::time_point<std::chrono::high_resolution_clock> &start_time, // for maxRuntime
      const std::filesystem::path &iterations_json_path,        // "" disables file output
      Timers &timers,
      El::Matrix<int32_t> &block_timings_ms);                   // out: per-block ms (iteration >= 2)

  // One predictor-corrector step (called by run; needs its precomputed inputs)
  void step(const Environment &env,
    const Solver_Parameters &parameters,const Verbosity &verbosity, const std::size_t &total_psd_rows,
    const bool &is_primal_and_dual_feasible, const Block_Info &block_info,
    const SDP &sdp, const El::Grid &grid,
    const Block_Diagonal_Matrix &X_cholesky,
    const Block_Diagonal_Matrix &Y_cholesky,
    const std::array<
      std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
      &A_X_inv,
    const std::array<
      std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
      &A_Y,
    const Block_Vector &primal_residue_p,
    BigInt_Shared_Memory_Syrk_Context &bigint_syrk_context, El::BigFloat &mu,
    El::BigFloat &beta_corrector, El::BigFloat &primal_step_length,
    El::BigFloat &dual_step_length, bool &terminate_now, Timers &timers,
    El::Matrix<int32_t> &block_timings_ms, El::BigFloat &Q_cond_number,
    El::BigFloat &max_block_cond_number,
    std::string &max_block_cond_number_name);

  void save_checkpoint(const std::filesystem::path &checkpoint_directory,
                       const Verbosity &verbosity,
                       const boost::property_tree::ptree &parameter_properties);
  bool load_checkpoint(const std::filesystem::path &checkpoint_directory,
                       const Block_Info &block_info, const Verbosity &verbosity,
                       const bool &require_initial_checkpoint);
};
```

Behaviour of `run()`:
- It always starts from the solver's current `x, X, y, Y`. You can warm-start by writing
  into them after construction; `outer_limits` does this with `copy_matrix(yp, solver.y.blocks[i])`.
- It can be called again, e.g. with a tighter `duality_gap_threshold`.
- It re-creates the `bigint_syrk` context (shared-memory windows) on every call.
- `block_timings_ms` is resized and filled by `step`. The first iteration's values are
  discarded (`Empty(false)`), so the output is meaningful only after at least 2 iterations.

Checkpoints:
- **Binary** (`save_checkpoint`): per-rank files `checkpoint_<gen>_<rank>` plus
  `checkpoint.json` (`current`, `backup`, `version`, `options`). They can only be loaded
  with the same precision, number of ranks, and block mapping.
- **Text**: `x_<j>.txt`, `y.txt`, `X_matrix_<2j+p>.txt`, `Y_matrix_<2j+p>.txt`. These can
  be loaded with any rank layout.

### 8.6 `SDP_Solver_Terminate_Reason` — `sdp_solve/SDP_Solver_Terminate_Reason.hxx`

```cpp
enum class SDP_Solver_Terminate_Reason
{
  PrimalDualOptimal, PrimalFeasible, DualFeasible, PrimalFeasibleJumpDetected,
  DualFeasibleJumpDetected, MaxComplementarityExceeded, MaxIterationsExceeded,
  MaxRuntimeExceeded, PrimalStepTooSmall, DualStepTooSmall, SIGTERM_Received,
};
std::ostream &operator<<(std::ostream &os, const SDP_Solver_Terminate_Reason &r);
```

`compute_feasible_and_termination` checks them in this priority order:
1. optimal (gap, primal and dual errors all below threshold)
2. dual feasible (if `find_dual_feasible`)
3. primal feasible (if `find_primal_feasible`)
4. dual jump (step = 1, if `detect_dual_feasible_jump`)
5. primal jump
6. max iterations
7. max runtime
8. primal step < `min_primal_step`
9. dual step < `min_dual_step`

`MaxComplementarityExceeded` is raised in `step`, and `SIGTERM_Received` in `run`.

### 8.7 `Solver_Parameters` — `sdp_solve/Solver_Parameters.hxx`

```cpp
struct Solver_Parameters
{
  int64_t max_iterations, max_runtime, checkpoint_interval;
  size_t max_shared_memory_bytes;
  bool find_primal_feasible, find_dual_feasible, detect_primal_feasible_jump,
    detect_dual_feasible_jump;
  size_t precision;
  El::BigFloat duality_gap_threshold, primal_error_threshold,
    dual_error_threshold, initial_matrix_scale_primal,
    initial_matrix_scale_dual, feasible_centering_parameter,
    infeasible_centering_parameter, step_length_reduction, max_complementarity,
    min_primal_step, min_dual_step;
  std::filesystem::path checkpoint_in, checkpoint_out;
  Solver_Parameters() = default;                               // ⚠️ fields uninitialized
  boost::program_options::options_description options();       // binds &fields, holds defaults
};
std::ostream &operator<<(std::ostream &os, const Solver_Parameters &p);
boost::property_tree::ptree to_property_tree(const Solver_Parameters &p);
```

| Field | CLI option | Default |
|---|---|---|
| `precision` | `--precision` | 400 (bits; informational for the solver — the effective precision is whatever `set_precision` set) |
| `max_iterations` | `--maxIterations` | 500 |
| `max_runtime` | `--maxRuntime` | `INT64_MAX` s |
| `checkpoint_interval` | `--checkpointInterval` | 3600 s |
| `max_shared_memory_bytes` | `--maxSharedMemory` | 0 (automatic) |
| `find_primal_feasible` / `find_dual_feasible` | `--findPrimalFeasible` / `--findDualFeasible` | false |
| `detect_primal_feasible_jump` / `detect_dual_feasible_jump` | `--detect…FeasibleJump` | false |
| `duality_gap_threshold` | `--dualityGapThreshold` | 1e-30 |
| `primal_error_threshold` / `dual_error_threshold` | | 1e-30 |
| `initial_matrix_scale_primal` / `_dual` | `--initialMatrixScale{Primal,Dual}` | 1e20 |
| `feasible_centering_parameter` | | 0.1 |
| `infeasible_centering_parameter` | | 0.3 |
| `step_length_reduction` | | 0.7 |
| `min_primal_step` / `min_dual_step` | | 0 |
| `max_complementarity` | | 1e100 |
| `checkpoint_out` | `--checkpointDir,-c` | none (the `sdpb` executable uses `<sdp>.ck`) |
| `checkpoint_in` | `--initialCheckpointDir,-i` | none (the `sdpb` executable uses `checkpoint_out`) |

Related: `sdp_solve/Solver_Parameters/String_To_Bytes_Translator.hxx`
(`String_To_Bytes_Translator::from_string("64G")` and the reverse, used by ptree).

### 8.8 `Write_Solution` — `sdp_solve/Write_Solution.hxx`

```cpp
struct Write_Solution
{
  std::string input_string;
  bool matrix_X=false, matrix_Y=false, vector_x=false, vector_y=false, vector_z=false;
  Write_Solution(const std::string &input);     // e.g. "x,y,z,X,Y"
  Write_Solution()=default;
};
inline std::ostream &operator<<(std::ostream &os, const Write_Solution &write_solution);
```

### 8.9 Header-only solver helpers

```cpp
// sdp_solve/SDP_Solver/run/save_c_minus_By.hxx — collective; rank 0 writes {"c_minus_By":[[..],..]}
inline void save_c_minus_By(const std::filesystem::path &path, const Block_Info &block_info,
                            const SDP &sdp, const Block_Vector &y,
                            const Verbosity &verbosity, Timers &timers);

// sdp_solve/SDP_Solver/run/step/compute_R_error.hxx — max|mu I - XY|, AllReduce
[[nodiscard]] inline El::BigFloat
compute_R_error(const El::BigFloat &mu, const Block_Diagonal_Matrix &minus_XY, Timers &timers);

// sdp_solve/SDP_Solver/run/step/update_cond_numbers.hxx
inline void
update_cond_numbers(const El::DistMatrix<El::BigFloat> &Q,
                    const Block_Info &block_info,
                    const Block_Diagonal_Matrix &schur_complement_cholesky,
                    const Block_Diagonal_Matrix &X_cholesky,
                    const Block_Diagonal_Matrix &Y_cholesky, Timers &timers,
                    // Output variables:
                    El::BigFloat &Q_cond_number,
                    El::BigFloat &max_block_cond_number,
                    std::string &max_block_cond_number_name);

// sdp_solve/memory_estimates.hxx — local memory estimates (bytes / element counts)
inline size_t get_matrix_size_local(const Block_Diagonal_Matrix &X);
inline size_t get_A_X_size_local(const Block_Info &block_info, const SDP &sdp);
inline size_t get_schur_complement_size_local(const Block_Info &block_info);
inline size_t get_B_size_local(const SDP &sdp);
inline size_t get_Q_size_local(const SDP &sdp);
inline size_t get_SDP_size_local(const SDP &sdp);
inline size_t get_heap_allocated_bytes(const Block_Diagonal_Matrix &m);
inline size_t get_heap_allocated_bytes(const Block_Matrix &m);
inline size_t get_heap_allocated_bytes(const Block_Vector &v);
inline size_t get_heap_allocated_bytes(const SDP &sdp);
inline size_t get_heap_allocated_bytes(const SDP_Solver &solver);
```

### 8.10 🔒 Solver internals (no header)

These are the building blocks of `run()`/`step()`. You can call them to build a custom
iteration, sensitivity analysis (as `approx_objective` does), or diagnostics. They are
listed in the order `run()` calls them. The type `A_pairing_t` below is shorthand, not
something defined in the code:
`std::array<std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>`,
indexed `[parity][local_block][column_block][row_block]`.

**Checkpoint loading** (declared in `load_checkpoint/load_checkpoint.cxx:6-11`):

```cpp
// SDP_Solver/load_checkpoint/load_binary_checkpoint.cxx:58
bool load_binary_checkpoint(const fs::path &checkpoint_directory,
                            const Verbosity &verbosity, SDP_Solver &solver)
// SDP_Solver/load_checkpoint/load_text_checkpoint.cxx:6
bool load_text_checkpoint(const fs::path &checkpoint_directory,
                          const std::vector<size_t> &block_indices,
                          const Verbosity &verbosity, SDP_Solver &solver)
```

**Per-iteration quantities** (declared in `SDP_Solver/run/run.cxx:14-73`):

```cpp
// run/print_header.cxx:5
void print_header(const Verbosity &verbosity)

// run/compute_objectives/compute_objectives.cxx:6
void compute_objectives(const SDP &sdp, const Block_Vector &x,
                        const Block_Vector &y, El::BigFloat &primal_objective,
                        El::BigFloat &dual_objective,
                        El::BigFloat &duality_gap, Timers &timers)

// run/cholesky_decomposition.cxx:5 — L = chol(A) per block; throws naming the block
void cholesky_decomposition(const Block_Diagonal_Matrix &A,
                            Block_Diagonal_Matrix &L,
                            const Block_Info &block_info,
                            const std::string &name)

// run/compute_bilinear_pairings/compute_bilinear_pairings.cxx:17
void compute_bilinear_pairings(
  const Block_Info &block_info, const Block_Diagonal_Matrix &X_cholesky,
  const Block_Diagonal_Matrix &Y,
  const std::vector<El::DistMatrix<El::BigFloat>> &bases_blocks,
  std::array<std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>,
             2> &A_X_inv,
  std::array<std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>,
             2> &A_Y,
  Timers &timers)

// run/compute_bilinear_pairings/compute_A_X_inv.cxx:6 — (L_X^{-1} V)^T (L_X^{-1} V)
void compute_A_X_inv(
  const Block_Info &block_info, const Block_Diagonal_Matrix &X_cholesky,
  const std::vector<El::DistMatrix<El::BigFloat>> &bases_blocks,
  std::array<std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>,
             2> &A_X_inv)

// run/compute_bilinear_pairings/compute_A_Y.cxx:16 — V^T Y V
void compute_A_Y(
  const Block_Info &block_info, const Block_Diagonal_Matrix &Y,
  const std::vector<El::DistMatrix<El::BigFloat>> &bases_blocks,
  std::array<std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>,
             2> &A_Y)

// run/compute_dual_residues_and_error.cxx:7 — d = c - Tr(A_* Y) - B y
void compute_dual_residues_and_error(
  const Block_Info &block_info, const SDP &sdp, const Block_Vector &y,
  const std::array<
    std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
    &A_Y,
  Block_Vector &dual_residues, El::BigFloat &dual_error, Timers &timers)

// run/compute_primal_residues_and_error_P_Ax_X.cxx:5 — P = sum A_p x_p - X
void compute_primal_residues_and_error_P_Ax_X(
  const Block_Info &block_info, const SDP &sdp, const Block_Vector &x,
  const Block_Diagonal_Matrix &X, Block_Diagonal_Matrix &primal_residues,
  El::BigFloat &primal_error, Timers &timers)

// run/compute_primal_residues_and_error_p_b_Bx.cxx:9 — p = b - B^T x
// (initialize primal_residue_p as a copy of solver.y)
void compute_primal_residues_and_error_p_b_Bx(const Block_Info &block_info,
                                              const SDP &sdp,
                                              const Block_Vector &x,
                                              Block_Vector &primal_residue_p,
                                              El::BigFloat &primal_error)

// run/compute_feasible_and_termination.cxx:4
void compute_feasible_and_termination(
  const Solver_Parameters &parameters, const El::BigFloat &primal_error,
  const El::BigFloat &dual_error, const El::BigFloat &duality_gap,
  const El::BigFloat &primal_step_length, const El::BigFloat &dual_step_length,
  const int &iteration,
  const std::chrono::time_point<std::chrono::high_resolution_clock>
    &solver_start_time,
  bool &is_primal_and_dual_feasible,
  SDP_Solver_Terminate_Reason &terminate_reason, bool &terminate_now)

// run/print_iteration.cxx:8 — stdout row + append to iterations.json (if path non-empty)
void print_iteration(
  const std::filesystem::path &iterations_json_path, const int &iteration,
  const El::BigFloat &mu, const El::BigFloat &primal_step_length,
  const El::BigFloat &dual_step_length, const El::BigFloat &beta_corrector,
  const SDP_Solver &sdp_solver,
  const std::chrono::time_point<std::chrono::high_resolution_clock>
    &solver_start_time,
  const std::chrono::time_point<std::chrono::high_resolution_clock>
    &iteration_start_time,
  const El::BigFloat &Q_cond_number, const El::BigFloat &max_block_cond_number,
  const std::string &max_block_cond_number_name, const Verbosity &verbosity)
```

**The step** (declared in `SDP_Solver/run/step/step.cxx:7-49`):

```cpp
// step/initialize_schur_complement_solver/initialize_schur_complement_solver.cxx:62
// S (Schur complement) → L_S, schur_off_diagonal = L_S^{-1} B, Q = sum (L^{-1}B)^T(L^{-1}B),
// then Cholesky(UPPER, Q) in place. Q must be N x N on the default (COMM_WORLD) grid.
void initialize_schur_complement_solver(
  const Environment &env, const Block_Info &block_info, const SDP &sdp,
  const std::array<
    std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
    &A_X_inv,
  const std::array<
    std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
    &A_Y,
  const El::Grid &group_grid, Block_Diagonal_Matrix &schur_complement_cholesky,
  Block_Matrix &schur_off_diagonal,
  BigInt_Shared_Memory_Syrk_Context &bigint_syrk_context,
  El::DistMatrix<El::BigFloat> &Q, Timers &timers,
  El::Matrix<int32_t> &block_timings_ms, const Verbosity verbosity)

// .../initialize_schur_complement_solver/compute_schur_complement.cxx:15
void compute_schur_complement(
  const Block_Info &block_info,
  const std::array<
    std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
    &A_X_inv,
  const std::array<
    std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
    &A_Y,
  Block_Diagonal_Matrix &schur_complement, Timers &timers)

// .../initialize_schur_complement_solver/compute_Q.cxx:134
void compute_Q(const Environment &env, const SDP &sdp,
               const Block_Info &block_info,
               const Block_Diagonal_Matrix &schur_complement,
               Block_Matrix &schur_off_diagonal,
               Block_Diagonal_Matrix &schur_complement_cholesky,
               BigInt_Shared_Memory_Syrk_Context &bigint_syrk_context,
               El::DistMatrix<El::BigFloat> &Q, Timers &timers,
               El::Matrix<int32_t> &block_timings_ms,
               const Verbosity verbosity)

// step/compute_search_direction/scale_multiply_add.cxx:4 — C = alpha A B + beta C
void scale_multiply_add(const El::BigFloat &alpha,
                        const Block_Diagonal_Matrix &A,
                        const Block_Diagonal_Matrix &B,
                        const El::BigFloat &beta, Block_Diagonal_Matrix &C)

// step/predictor_centering_parameter.cxx:4 — 0 if feasible else infeasible_centering_parameter
El::BigFloat predictor_centering_parameter(const Solver_Parameters &parameters,
                                           const bool is_primal_dual_feasible)

// step/corrector_centering_parameter/corrector_centering_parameter.cxx:12
El::BigFloat corrector_centering_parameter(
  const Solver_Parameters &parameters, const Block_Diagonal_Matrix &X,
  const Block_Diagonal_Matrix &dX, const Block_Diagonal_Matrix &Y,
  const Block_Diagonal_Matrix &dY, const El::BigFloat &mu,
  const bool is_primal_dual_feasible, const size_t &total_psd_rows)

// step/corrector_centering_parameter/frobenius_product_of_sums.cxx:6 — Tr((X+dX)(Y+dY))
El::BigFloat frobenius_product_of_sums(const Block_Diagonal_Matrix &X,
                                       const Block_Diagonal_Matrix &dX,
                                       const Block_Diagonal_Matrix &Y,
                                       const Block_Diagonal_Matrix &dY)

// step/compute_search_direction/compute_search_direction.cxx:44
void compute_search_direction(
  const Block_Info &block_info, const SDP &sdp, const SDP_Solver &solver,
  const Block_Diagonal_Matrix &minus_XY,
  const Block_Diagonal_Matrix &schur_complement_cholesky,
  const Block_Matrix &schur_off_diagonal,
  const Block_Diagonal_Matrix &X_cholesky, const El::BigFloat &beta,
  const El::BigFloat &mu, const Block_Vector &primal_residue_p,
  const bool &is_corrector_phase, const El::DistMatrix<El::BigFloat> &Q,
  Block_Vector &dx, Block_Diagonal_Matrix &dX, Block_Vector &dy,
  Block_Diagonal_Matrix &dY)

// step/compute_search_direction/cholesky_solve.cxx:4 — X := A^{-1} X
void cholesky_solve(const Block_Diagonal_Matrix &ACholesky,
                    Block_Diagonal_Matrix &X)

// step/compute_search_direction/compute_schur_RHS.cxx:21 — dx = -d - Tr(A_p Z)
void compute_schur_RHS(const Block_Info &block_info, const SDP &sdp,
                       const Block_Vector &dual_residues,
                       const Block_Diagonal_Matrix &Z,
                       Block_Vector &dx)

// step/compute_search_direction/solve_schur_complement_equation.cxx:16
// Solves the Schur system for (dx, dy) in place, given L_S, L_S^{-1}B and chol(Q)
void solve_schur_complement_equation(
  const Block_Diagonal_Matrix &schur_complement_cholesky,
  const Block_Matrix &schur_off_diagonal,
  const El::DistMatrix<El::BigFloat> &Q, Block_Vector &dx, Block_Vector &dy)

// step/step_length/step_length.cxx:27 — alpha = 1 if min eig(L^{-1} dM L^{-T}) > -gamma, else -gamma/lambda
El::BigFloat step_length(const Block_Diagonal_Matrix &MCholesky,
                         const Block_Diagonal_Matrix &dM,
                         const El::BigFloat &gamma,
                         const std::string &timer_name,
                         Timers &timers)

// step/step_length/lower_triangular_inverse_congruence.cxx:4 — A := L^{-1} A L^{-T}
void lower_triangular_inverse_congruence(const Block_Diagonal_Matrix &L,
                                         Block_Diagonal_Matrix &A)

// step/step_length/min_eigenvalue.cxx:8 — global min eigenvalue; destroys A
El::BigFloat min_eigenvalue(Block_Diagonal_Matrix &A)

// step/frobenius_product_symmetric.cxx:4 — defined but unused
```

Internal helpers with external linkage that other files don't use:
- `initialize_schur_off_diagonal`, `check_normalized_Q_diagonal`, `syrk_Q` (`compute_Q.cxx:9/65/94`)
- `multiply` (`compute_search_direction.cxx:25`)
- `write_local_blocks<T>` (`save_checkpoint.cxx:12`)
- `read_local_binary_blocks<T>` (`load_binary_checkpoint.cxx:9`)
- `parse_block_data` (`SDP_Block_Data.cxx:31`)
- the `create_blas_jobs_*` helpers (`create_blas_job_schedule.cxx`)

**Sensitivity solve without iterating.** This is the pattern `approx_objective` uses
(`approx_objective/setup_solver.cxx:174-222`). Given `X`, `Y` from a solution:
1. Call `cholesky_decomposition(X, X_chol, block_info, "X")`.
2. Call `compute_A_X_inv(block_info, X_chol, sdp.bases_blocks, A_X_inv)` and
   `compute_A_Y(block_info, Y, sdp.bases_blocks, A_Y)`.
3. Call `initialize_bigint_syrk_context(env, block_info, sdp, max_shared_memory_bytes, verbosity)`.
4. Call `initialize_schur_complement_solver(...)`, where `Q` is `N×N` on a default grid.
5. From then on, `solve_schur_complement_equation(schur_complement_cholesky,
   schur_off_diagonal, Q, dx, dy)` solves any right-hand side.

### 8.11 `bigint_syrk` — exact `Q = PᵀP` with double-precision BLAS

Directory: `sdp_solve/SDP_Solver/run/bigint_syrk/` (design notes in its `Readme.md`).

```cpp
// initialize_bigint_syrk_context.hxx
inline std::tuple<size_t, std::vector<int>, size_t>
get_child_index_sizes_and_count(const El::mpi::Comm &parent_comm, const El::mpi::Comm &child_comm);
struct Grouped_Block_Size_Info
{
  El::mpi::Comm node_comm, group_comm; size_t group_index, num_groups;
  std::vector<int> group_comm_sizes; size_t block_width;
  std::vector<El::Int> blocks_height_per_group;
  Grouped_Block_Size_Info(const Environment &env, const Block_Info &block_info, const SDP &sdp);
};
inline BigInt_Shared_Memory_Syrk_Context
initialize_bigint_syrk_context(const Environment &env, const Block_Info &block_info,
                               const SDP &sdp, const size_t max_shared_memory_bytes,
                               const Verbosity verbosity);   // collective per node

// BigInt_Shared_Memory_Syrk_Context.hxx
class BigInt_Shared_Memory_Syrk_Context : boost::noncopyable
{
public:
  BigInt_Shared_Memory_Syrk_Context(
    const El::mpi::Comm &shared_memory_comm, size_t group_index,
    const std::vector<int> &group_comm_sizes, mp_bitcnt_t precision,
    size_t max_shared_memory_bytes,
    const std::vector<El::Int> &blocks_height_per_group, int block_width,
    const std::vector<size_t> &block_index_local_to_global,
    Verbosity verbosity,
    const std::function<Blas_Job_Schedule(
      Blas_Job::Kind kind, El::UpperOrLower uplo, size_t num_ranks,
      size_t num_primes, int output_height, int output_width,
      Verbosity _verbosity)> &create_job_schedule
    = create_blas_job_schedule);
  // output = sum over all blocks on all nodes of P_b^T P_b (P already normalized to big integers)
  void bigint_syrk_blas(El::UpperOrLower uplo,
                        const std::vector<El::DistMatrix<El::BigFloat>> &bigint_input_matrix_blocks,
                        El::DistMatrix<El::BigFloat> &bigint_output,
                        Timers &timers, El::Matrix<int32_t> &block_timings_ms);
};
// ⚠️ stores a reference to group_comm_sizes: the vector must outlive the context

// Matrix_Normalizer.hxx — column normalisation and 2^precision shift (and restore)
class Matrix_Normalizer : boost::noncopyable
{
public:
  const int precision; const std::vector<El::BigFloat> column_norms;
  template <class TMatrix> Matrix_Normalizer(const TMatrix &P_matrix, int precision, El::mpi::Comm comm);
  template <class TMatrix> Matrix_Normalizer(const std::vector<TMatrix> &P_matrix_blocks,
                                             int P_matrix_width, int precision, El::mpi::Comm comm);
  template <class TMatrix> void normalize_and_shift_P(TMatrix &P_block);
  template <class TMatrix_Blocks> void normalize_and_shift_P_blocks(TMatrix_Blocks &P_matrix_blocks);
  template <class TMatrix> void restore_P(TMatrix &P_block);
  template <class TMatrix_Blocks> void restore_P_blocks(TMatrix_Blocks &P_matrix_blocks);
  template <class TMatrix> void restore_Q(El::UpperOrLower uplo, TMatrix &Q_matrix);
};

// blas_jobs/
struct Blas_Job
{
  enum Kind { syrk, gemm };
  struct Cost { size_t elements, blas_calls; /* <, +, += , << */ };
  const Kind kind; const size_t prime_index; const El::Range<El::Int> I, J; const Cost cost;
  static Blas_Job create_syrk_job(size_t prime_index, const El::Range<El::Int> &I);
  static Blas_Job create_gemm_job(size_t prime_index, const El::Range<El::Int> &I,
                                  const El::Range<El::Int> &J);
};
struct Blas_Job_Schedule
{
  const std::vector<std::vector<Blas_Job>> jobs_by_rank;
  Blas_Job_Schedule(size_t num_ranks, const std::vector<Blas_Job> &jobs);
  [[nodiscard]] Blas_Job::Cost max_rank_cost() const;
};
Blas_Job_Schedule create_blas_job_schedule(Blas_Job::Kind kind,
                                           El::UpperOrLowerNS::UpperOrLower uplo,
                                           size_t num_ranks, size_t num_primes,
                                           El::Int output_matrix_height,
                                           El::Int output_matrix_width, Verbosity verbosity);
// blas_jobs/LPT_scheduling.hxx — greedy longest-processing-time scheduling (template)

// fmpz/ — FLINT wrappers
struct Fmpz_Comb : boost::noncopyable      // primes with p^2 k < 2^53 whose product > 2^bits
{
  fmpz_comb_t comb{}; fmpz_comb_temp_t comb_temp{};
  const std::vector<mp_limb_t> primes; size_t num_primes;
  std::vector<nmod_t> mods; std::vector<ulong> shifts;
  Fmpz_Comb(flint_bitcnt_t bits, slong k);
  Fmpz_Comb(flint_bitcnt_t Abits, flint_bitcnt_t Bbits, int sign, slong k);
  ~Fmpz_Comb();
};
struct Fmpz_BigInt : private boost::noncopyable
{ fmpz_t value; void from_BigFloat(const El::BigFloat &input); void to_BigFloat(El::BigFloat &output) const; };
class Fmpz_Matrix                          // RAII fmpz_mat_t; Height/Width/Resize/Get/Set/(), conversions
{ public: explicit Fmpz_Matrix(const El::Matrix<El::BigFloat> &input);
          explicit Fmpz_Matrix(const El::DistMatrix<El::BigFloat> &input);
          void ToBigFloatMatrix(El::Matrix<El::BigFloat> &output) const; /* ... */ };
// fmpz/fmpz_BigFloat_convert.hxx — fmpz_set_mpf / fmpz_get_mpf wrappers
// fmpz/fmpz_mul_blas_util.hxx     — helpers adapted from FLINT mul_blas.c

// shared-memory residue storage
template <class T> class Residue_Matrices_Window : boost::noncopyable
{
public:
  const size_t num_primes, height, width, prime_stride;
  std::vector<El::Matrix<T>> residues;   // views into the window, one per prime
  Residue_Matrices_Window(El::mpi::Comm shared_memory_comm, size_t num_primes,
                          size_t height, size_t width);
  [[nodiscard]] El::mpi::Comm Comm() const; void Fence();
};
template <class T> class Block_Residue_Matrices_Window : public Residue_Matrices_Window<T>
{
public:
  const size_t num_blocks;
  std::vector<std::vector<El::Matrix<T>>> block_residues;   // [prime][block] views
  Block_Residue_Matrices_Window(El::mpi::Comm shared_memory_comm, size_t num_primes,
                                size_t num_blocks, const std::vector<El::Int> &block_heights,
                                size_t block_width);
};
```

`test/src/unit_tests/cases/calculate_matrix_square.test.cxx` is a complete usage example
of the context and normalizer outside the solver.

---

## 9. `src/sdpb` — driver code (executable only)

These functions are not in any library. Copy the files if you want them.

```cpp
// sdpb/SDPB_Parameters.hxx — argv-only constructor (also reads --paramFile)
struct SDPB_Parameters
{
  bool no_final_checkpoint; size_t proc_granularity = 1;
  bool require_initial_checkpoint = false; Write_Solution write_solution;
  Solver_Parameters solver; Verbosity verbosity;
  std::filesystem::path sdp_path, out_directory, param_path;
  SDPB_Parameters(int argc, char *argv[]);   // side effect: creates out_directory, test-writes out.txt
  bool is_valid() const;                     // !sdp_path.empty()
};
std::ostream &operator<<(std::ostream &os, const SDPB_Parameters &p);
boost::property_tree::ptree to_property_tree(const SDPB_Parameters &p);

// sdpb/solve.cxx:23 — grid, SDP, SDP_Solver, run, final checkpoint, save_solution;
// on SIGTERM calls MPI_Finalize() and exit(SIGTERM) ⚠️
Timers solve(const Block_Info &block_info, const SDPB_Parameters &parameters,
             const Environment &env,
             const std::chrono::time_point<std::chrono::high_resolution_clock>
               &start_time,
             El::Matrix<int32_t> &block_timings_ms)

// sdpb/save_solution.cxx:8 — out.txt (+ y.txt, z.txt, x_<j>.txt, X_matrix_*, Y_matrix_* per Write_Solution)
void save_solution(
  const SDP_Solver &solver,
  const SDP_Solver_Terminate_Reason &terminate_reason,
  const int64_t &solver_runtime, const fs::path &out_directory,
  const Write_Solution &write_solution,
  const std::vector<size_t> &block_indices,
  const std::optional<std::vector<El::BigFloat>> &normalization,
  const Verbosity &verbosity)

// sdpb/write_timing.cxx:34 — rank 0 writes <checkpoint_out>/block_timings (one int per block)
void write_block_timings(const fs::path &checkpoint_out,
                         const Block_Info &block_info,
                         const El::Matrix<int32_t> &block_timings_ms,
                         const Verbosity verbosity)

// sdpb/write_timing.cxx:7 — <checkpoint_out>.profiling/profiling.<rank> (rotates old dirs)
void write_profiling(const fs::path &checkpoint_out, const Timers &timers)
```

`main()` (`sdpb/main.cxx`) does the following:
1. Creates the `Environment` and `SDPB_Parameters`, then calls `set_precision`.
2. Builds `Block_Info(env, sdp_path, checkpoint_in, proc_granularity, verbosity)`.
3. Runs a timing solve if all of these hold: more than one rank, `block_timings_filename`
   is empty, and `checkpoint_in/checkpoint.0` doesn't exist. The timing solve uses 2
   iterations, all thresholds 0, and `no_final_checkpoint`. It then calls
   `write_block_timings`, rebuilds `Block_Info` from the timings, `swap`s it in, and
   subtracts the time spent from `max_runtime`.
4. Otherwise, copies the found `block_timings` file into `checkpoint_out`.
5. Calls `solve(...)`.
6. At debug verbosity, calls `write_profiling`.

---

## 10. Post-processing tools (executables only)

None of these are libraries. To embed one, compile its `.cxx` files (except `main.cxx`)
into your target and forward-declare the functions below. Every function here has no header.

### 10.1 spectrum — `src/spectrum/` (needs MPSolve)

Headers: `Zero.hxx`, `Zeros.hxx`, `compute_spectrum/compute_lambda.hxx`,
`compute_spectrum/interpolate.hxx`.

```cpp
struct Zero  { El::BigFloat zero; El::Matrix<El::BigFloat> lambda; Zero(const El::BigFloat &z); };
struct Zeros { std::vector<Zero> zeros; El::BigFloat error; std::filesystem::path block_path; };

inline void compute_lambda(const PVM_Info &pvm_info, const El::Matrix<El::BigFloat> &x,
                           const std::vector<El::BigFloat> &zero_values,
                           const std::optional<El::BigFloat> &min_eigenvalue_ratio,
                           Zeros &spectrum_block, Timers &timers);
inline std::vector<Boost_Polynomial> get_lagrange_basis(const std::vector<El::BigFloat> &sample_points);
inline Boost_Polynomial interpolate(const std::vector<Boost_Polynomial> &lagrange_basis,
                                    const std::vector<El::BigFloat> &y_values);
inline Boost_Polynomial interpolate(const std::vector<El::BigFloat> &x_values,
                                    const std::vector<El::BigFloat> &y_values);
```

🔒 Functions:

```cpp
// read_pmp_info.cxx:95 — pmp_info.json (plain or inside sdp.zip); rank r keeps blocks index % size == r
PMP_Info read_pmp_info(const std::filesystem::path &input_path, Timers &timers)
// read_c_minus_By.cxx:62 — local blocks of c_minus_By.json
std::vector<El::Matrix<El::BigFloat>> read_c_minus_By(const std::filesystem::path &input_path,
                                                      const PMP_Info &pmp_info, Timers &timers)
// read_x.cxx:7 — x_<block_index>.txt for local blocks
std::vector<El::Matrix<El::BigFloat>> read_x(const fs::path &solution_path, const PMP_Info &pmp,
                                             Timers &timers)
// set_default_parameters.cxx:45 — threshold / min_eigenvalue_ratio := sqrt(dualityGap from out.txt); collective
void set_default_parameters(const std::filesystem::path &solution_dir, const PMP_Info &pmp_info,
                            const bool &need_lambda, const Verbosity &verbosity, Timers &timers,
                            std::optional<Boost_Float> &threshold,
                            std::optional<El::BigFloat> &min_eigenvalue_ratio)
// compute_spectrum/compute_spectrum.cxx:21 — local, per block; per-block failures become warnings
std::vector<Zeros> compute_spectrum(const PMP_Info &pmp,
  const std::vector<El::Matrix<El::BigFloat>> &c_minus_By,
  const std::optional<std::vector<El::Matrix<El::BigFloat>>> &x,
  const Boost_Float &threshold, const El::BigFloat &max_zero,
  const El::BigFloat &min_zero_distance, const bool &need_lambda,
  const std::optional<El::BigFloat> &min_eigenvalue_ratio, const Verbosity &verbosity,
  const std::filesystem::path &output_path, Timers &timers)
// compute_spectrum/find_zeros.cxx:169
std::vector<El::BigFloat> find_zeros(const El::Matrix<El::BigFloat> &c_minus_By_block,
  const PVM_Info &pvm, const Boost_Float &threshold, const El::BigFloat &max_zero,
  const El::BigFloat &min_zero_distance, Timers &timers)
// compute_spectrum/mpsolve.cxx
std::vector<MPSolve_Root> find_polynomial_roots(const std::vector<El::BigFloat> &polynomial_coeffs,
                                                Timers &timers);                         // :25 (MPSolve_Root is file-local)
std::vector<El::BigFloat> find_real_positive_roots_sorted(const Boost_Polynomial &polynomial,
                                                          Timers &timers);               // :132
El::BigFloat arithmetic_mean(const std::vector<El::BigFloat> &values);                   // :167
std::vector<El::BigFloat> deduplicate_sorted_values(const std::vector<El::BigFloat> &sorted_values,
                                                    const El::BigFloat &min_distance);   // :176
std::vector<El::BigFloat> find_real_positive_minima_sorted(const Boost_Polynomial &polynomial,
  const El::BigFloat &min_zero_distance, Timers &timers);                                // :228
// write_spectrum/write_spectrum.cxx:106 — gather to rank 0 and write JSON; collective
void write_spectrum(const fs::path &output_path, const std::vector<Zeros> &zeros_blocks,
                    const PMP_Info &pmp_info, Timers &timers)
// write_spectrum/write_file.cxx:10 — rank 0 only
void write_file(const fs::path &output_path, const std::vector<Zeros> &zeros_blocks, Timers &timers)
// write_profiling.cxx:16, :31
void create_profiling_dir(const fs::path &spectrum_output_path)
void write_profiling(const fs::path &spectrum_output_path, Timers &timers)
// handle_arguments.cxx:14 — CLI parsing; ⚠️ calls Environment::set_precision internally
void handle_arguments(const int &argc, char **argv, std::optional<Boost_Float> &threshold,
  El::BigFloat &max_zero, El::BigFloat &min_zero_distance, fs::path &pmp_info_path,
  fs::path &solution_dir, fs::path &c_minus_By_path, fs::path &output_path, bool &need_lambda,
  std::optional<El::BigFloat> &min_eigenvalue_ratio, Verbosity &verbosity)
```

The call sequence is: `read_pmp_info` → `set_default_parameters` (or set `threshold`
yourself) → `read_x` (only if `need_lambda`) → `read_c_minus_By` → `compute_spectrum` →
`write_spectrum`.

⚠️ `spectrum` distributes blocks round-robin by `index % size`, not by `Block_Info`.
Passing `c−By` directly from an `SDP_Solver` therefore needs a redistribution step. The
simplest route is to write `save_c_minus_By(...)` and read it back with `read_c_minus_By`.

### 10.2 approx_objective — `src/approx_objective/`

```cpp
// Approx_Objective.hxx
struct Approx_Objective
{
  El::BigFloat objective, d_objective, dd_objective;
  Approx_Objective(const Block_Info &block_info, const SDP &sdp, const SDP &d_sdp,
                   const Block_Vector &x, const Block_Vector &y,
                   const Block_Diagonal_Matrix &schur_complement_cholesky,
                   const Block_Matrix &schur_off_diagonal,
                   const El::DistMatrix<El::BigFloat> &Q);   // linear + quadratic
  Approx_Objective(const SDP &sdp, const SDP &d_sdp, const Block_Vector &x,
                   const Block_Vector &y);                    // linear only
};
// Approx_Parameters.hxx — argv-only constructor
struct Approx_Parameters
{
  size_t proc_granularity = 1; size_t precision; size_t max_shared_memory_bytes;
  std::filesystem::path sdp_path, new_sdp_path, solution_dir, param_path;
  bool write_solver_state, linear_only; Verbosity verbosity;
  Approx_Parameters(int argc, char *argv[]);
  [[nodiscard]] bool is_valid() const;
};
```

🔒 Functions:

```cpp
// Axpy.cxx:3 — delta_sdp += alpha * new_sdp (B, c, b, const; bases assumed identical)
void Axpy(const El::BigFloat &alpha, const SDP &new_sdp, SDP &delta_sdp)
// Approx_Objective/compute_dx_dy.cxx:9
void compute_dx_dy(const Block_Info &block_info, const SDP &d_sdp, const Block_Vector &x,
                   const Block_Vector &y, const Block_Diagonal_Matrix &schur_complement_cholesky,
                   const Block_Matrix &schur_off_diagonal, const El::DistMatrix<El::BigFloat> &Q,
                   Block_Vector &dx, Block_Vector &dy)
// linear_approximate_objectives.cxx:11 — input_path: one SDP or an .nsv list
std::vector<std::pair<std::string, Approx_Objective>> linear_approximate_objectives(
  const Block_Info &block_info, const El::Grid &grid, const SDP &sdp, const Block_Vector &x,
  const Block_Vector &y, const fs::path &input_path)
// quadratic_approximate_objectives.cxx:11
std::vector<std::pair<std::string, Approx_Objective>> quadratic_approximate_objectives(
  const Block_Info &block_info, const El::Grid &grid, const SDP &sdp, const Block_Vector &x,
  const Block_Vector &y, const Block_Diagonal_Matrix &schur_complement_cholesky,
  const Block_Matrix &schur_off_diagonal, const El::DistMatrix<El::BigFloat> &Q,
  const fs::path &input_path)
// setup_solver.cxx:153 — reads cached state or rebuilds it from X_matrix_*/Y_matrix_* (§8.10)
void setup_solver(const Environment &env, const Block_Info &block_info, const El::Grid &grid,
                  const SDP &sdp, const Approx_Parameters &parameters,
                  Block_Diagonal_Matrix &schur_complement_cholesky,
                  Block_Matrix &schur_off_diagonal, El::DistMatrix<El::BigFloat> &Q)
// write_solver_state.cxx:8
void write_solver_state(const std::vector<size_t> &block_indices, const fs::path &solution_dir,
                        const Block_Diagonal_Matrix &schur_complement_cholesky,
                        const Block_Matrix &schur_off_diagonal,
                        const El::DistMatrix<El::BigFloat> &Q)
```

`setup_solver` only uses `parameters.solution_dir`, `max_shared_memory_bytes` and
`verbosity`. Since `Approx_Parameters` can only be built from argv, it is easier to
inline the body (quoted in §8.10).

### 10.3 outer_limits and pmp2functions

`outer_limits` is an experimental point-sampling solver. It is also the only in-repo
user of the in-memory `SDP` constructor.

```cpp
// outer_limits/Function.hxx — Chebyshev series on [0, max_delta]
struct Function
{
  El::BigFloat max_delta, epsilon_value, infinity_value;
  std::vector<El::BigFloat> chebyshev_coeffs;
  El::BigFloat eval(const El::BigFloat &epsilon, const El::BigFloat &infinity,
                    const El::BigFloat &x) const;
};
// outer_limits/Outer_Parameters.hxx — argv-only constructor (⚠️ truncates output_path on rank 0)
struct Outer_Parameters
{
  bool require_initial_checkpoint = false, use_svd = false; Write_Solution write_solution;
  El::BigFloat duality_gap_reduction, mesh_threshold; Solver_Parameters solver;
  Verbosity verbosity; std::filesystem::path functions_path, points_path, output_path, param_path;
  Outer_Parameters(int argc, char *argv[]);
  bool is_valid() const;
};
```

🔒 Functions:

```cpp
// compute_optimal/compute_optimal.cxx:55 — returns z (weights, length N+1)
std::vector<El::BigFloat> compute_optimal(
  const std::vector<std::vector<std::vector<std::vector<Function>>>> &function_blocks,
  const std::vector<std::vector<El::BigFloat>> &initial_points,
  const std::vector<El::BigFloat> &objectives, const std::vector<El::BigFloat> &normalization,
  const Environment &env, const Outer_Parameters &parameters_in,
  const std::chrono::time_point<std::chrono::high_resolution_clock> &start_time)
// read_function_blocks/read_function_blocks.cxx:15
void read_function_blocks(const fs::path &input_file, std::vector<El::BigFloat> &objectives,
  std::vector<El::BigFloat> &normalization,
  std::vector<std::vector<std::vector<std::vector<Function>>>> &functions)
// read_points/read_points.cxx:10
void read_points(const fs::path &input_path, std::vector<std::vector<El::BigFloat>> &points)
// compute_optimal/setup_constraints.cxx:7, compute_y_transform.cxx:8, find_new_points/*.cxx,
// load_checkpoint/load_checkpoint.cxx:21 (⚠️ declare it void, see §11), save_checkpoint.cxx:25
// pmp2functions/write_functions.cxx:9 — PMP → functions JSON (single rank; 1x1 and 2x2 only)
void write_functions(const fs::path &output_path, const Polynomial_Matrix_Program &pmp)
```

The in-memory SDP pattern `outer_limits` uses (`compute_optimal.cxx:168-219`):
1. Build `Block_Info(env, matrix_dimensions, verbosity)`.
2. Build `El::Grid grid(block_info.mpi_comm.value)`.
3. Construct `SDP sdp(objective_const, primal_objective_c, free_var_matrix, yp_to_y_star,
   dual_objective_b_star, normalization, primal_c_scale, block_info, grid)`.
4. Construct `SDP_Solver solver(params, verbosity, false, block_info, grid, N)`.
5. Warm-start `solver.y` from the previous iteration.
6. Call `solver.run(..., /*iterations_json_path*/ fs::path(), ...)`.

---

## 11. Known API defects

All of these were verified against the current tree. They matter if you write forward
declarations yourself.

| Location | Problem | What to do |
|---|---|---|
| `pmp2sdp/write_sdp.cxx:19` vs `write_control_json.cxx:21` | Declared returning `size_t`, defined `void` (UB if the result is used) | Declare it `void`. |
| `outer_limits/compute_optimal/compute_optimal.cxx:27` vs `load_checkpoint/load_checkpoint.cxx:21` | Declared returning `boost::optional<int64_t>`, defined `void` | Declare it `void`. |
| `pmp2sdp/Dual_Constraint_Group.cxx:11` | `write_block_json` declared, never defined | Don't use it. |
| `approx_objective/Approx_Parameters.hxx:27`, `pmp2functions/Pmp2functions_Parameters.hxx:24` | `to_property_tree(const Approx_Parameters&)` and `operator<<(…, const Pmp2functions_Parameters&)` declared, never defined | Link error if used. |
| `sdp_solve/Solver_Parameters.hxx` | Defaulted constructor leaves every field uninitialized | Use §3.6. |
| `SDP_Solver` constructor with empty `checkpoint_in` | Looks for `checkpoint.json` / `x_0.txt` in the CWD | Always set `checkpoint_in`. |
| `sdp_solve/SDP.hxx` | No default constructor. The `yp_to_y` in-memory constructor supports only `num_points == 1` (§3.4); general in-memory SDPs use the `Dual_Constraint_Group` constructor (§3.5) | — |
| `BigInt_Shared_Memory_Syrk_Context` | Stores `const std::vector<int> &group_comm_sizes` by reference; `initialize_bigint_syrk_context` passes a member of a local that dies on return, leaving it dangling | Harmless today because it's only read inside the constructor (see below); don't add code that reads it later. |
| `create_blas_job_schedule.cxx:121-124` | The `El::LOWER` branch skips the same half as `UPPER` | Only call with `El::UPPER` (what `compute_Q` does). |
| `step/frobenius_product_symmetric.cxx:4` | Defined, never used | — |

> Note on `group_comm_sizes`: `initialize_bigint_syrk_context` builds a local
> `Grouped_Block_Size_Info info` and passes `info.group_comm_sizes` to the context
> constructor, which stores it as a `const std::vector<int> &` member. `info` is
> destroyed when the function returns, leaving a dangling reference. In the current code
> the member is only read inside the constructor (`BigInt_Shared_Memory_Syrk_Context.cxx:113-323`),
> so this is latent rather than live. If you extend the class, first change the member to
> be stored by value.
