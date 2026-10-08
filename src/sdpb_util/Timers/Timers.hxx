//=======================================================================
// Copyright 2014-2015 David Simmons-Duffin.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "Timer.hxx"
#include "sdpb_util/Environment.hxx"
#include "sdpb_util/Verbosity.hxx"

#include <El.hpp>

#include <boost/core/noncopyable.hpp>

#include <chrono>
#include <filesystem>
#include <list>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Key/value annotations attached to a timer, e.g. {"kind","gemm"}, {"m","24"}.
// Values are written to the profile as JSON strings.
using Timer_Attrs = std::vector<std::pair<std::string, std::string>>;

// One node of the timer tree.
// The tree structure is given by `parent` ids (not by parsing names).
struct Timer_Entry
{
  // Leaf name, e.g. "cholesky"
  std::string name;
  // Legacy dot-separated name, e.g. "sdpb.solve.run.iter_1.step.cholesky".
  // Used by Timers::elapsed_milliseconds(name).
  std::string full_name;
  int64_t id = -1;
  int64_t parent = -1;
  int depth = 0;
  Timer timer;
  // Accumulators (see Accumulating_Timer): number of accumulated scopes,
  // total wall/cpu time of the scopes and the longest scope.
  // For ordinary scoped timers is_accumulator == false and count == 1.
  bool is_accumulator = false;
  int64_t count = 1;
  int64_t accumulated_ns = 0;
  int64_t accumulated_cpu_ns = 0;
  int64_t max_ns = 0;
  Timer_Attrs attrs;

  // Wall time to report: accumulated time for accumulators,
  // timer elapsed time otherwise.
  [[nodiscard]] int64_t elapsed_nanoseconds() const;
  [[nodiscard]] int64_t cpu_nanoseconds() const;
};

struct Timers
{
  friend struct Scoped_Timer; // can change private field prefix
  friend struct Accumulating_Timer;

private:
  // We use std::list instead of std::vector to avoid reallocation.
  // Scoped_Timer holds reference to timer, which would be invalidated after reallocation.
  std::list<Timer_Entry> named_timers;
  std::string prefix;
  // ids of currently running Scoped_Timers, innermost last
  std::vector<int64_t> open_ids;

  Verbosity verbosity = Verbosity::regular;
  std::string node_debug_prefix;

  bool can_read_meminfo = true;
  // Max MemUsed value
  size_t max_mem_used = 0;
  // name of the timer that had max MemUsed value
  std::string max_mem_used_name;

  // Creation time, all "start" offsets in the profile are relative to it.
  std::chrono::steady_clock::time_point t0_steady;
  int64_t t0_unix_ns = 0;

  int node_index = 0;
  int num_nodes = 1;
  int node_rank = 0;

  // Run-level metadata: key -> JSON value (already serialized)
  std::map<std::string, std::string> meta;

  std::filesystem::path autosave_path;

  // Profiling detail level, see --profileDetail in sdpb.
  // 0: coarse timers only, 1: default, 2: more attributes, 3: probe barriers.
  // detail_off (-1): no timer is recorded at all; Scoped_Timer still measures
  // its own elapsed time (for callers that use it), Accumulating_Timer counts
  // nothing. For solves nobody profiles (e.g. sdpb_python without
  // profile_path), where building the timer tree costs a few percent.
  int detail_level = 1;

public:
  Timers();
  Timers(const Environment &env, const Verbosity &verbosity);
  ~Timers() noexcept;

private:
  Timer_Entry &add_and_start(const std::string &name, Timer_Attrs attrs);
  Timer_Entry &add_accumulator(const std::string &name, Timer_Attrs attrs);
  Timer_Entry &add_entry(const std::string &name, Timer_Attrs attrs);

public:
  // Write the profile (valid JSON, see docs/Usage.md "Profiling")
  // for the current rank. The file is written atomically
  // (to <path>.tmp, then renamed).
  void write_profile(const std::filesystem::path &path) const;

  // Elapsed time of the last timer with the given legacy full name,
  // e.g. "sdpb.solve.run".
  [[nodiscard]] int64_t elapsed_milliseconds(const std::string &s) const;

  // Attach run-level metadata. `json_value` must be valid JSON
  // (e.g. "768", "\"text\"", "[1,2]", "{...}").
  void set_meta(const std::string &key, const std::string &json_value);
  [[nodiscard]] const std::map<std::string, std::string> &get_meta() const;

  // If set, flush() writes the profile to this path.
  void set_autosave_path(const std::filesystem::path &path);
  [[nodiscard]] const std::filesystem::path &get_autosave_path() const;
  // Write the current (possibly partial) profile to the autosave path.
  // Does nothing if no autosave path is set. Never throws.
  void flush() const noexcept;

  static constexpr int detail_off = -1;
  void set_detail(int level);
  [[nodiscard]] int detail() const;
  // false at detail_off: timers record nothing
  [[nodiscard]] bool enabled() const;

  [[nodiscard]] const std::list<Timer_Entry> &entries() const;
  [[nodiscard]] std::chrono::steady_clock::time_point t0() const;

private:
  void print_max_mem_used() const;
  void process_meminfo(const std::string &name, int depth);
};

// Simple RAII timer
// start() in constructor, stop() in destructor
// Temporarily appends name to timers.prefix
// NB: make sure that all Scoped_Timers are properly nested!
// Otherwise, prefixes will be wrong.
//
// Example:
//
// void f()
// {
//   Timers timers(false);
//   Scoped_Timer root_timer(timers, "root"); // "root"
//   Scoped_Timer foo_timer(timers, "foo"); // "root.foo"
//   foo_timer.stop();
//   Scoped_Timer bar_timer(timers, "bar"); // "root.bar"
// }
struct Scoped_Timer : boost::noncopyable
{
  Scoped_Timer(Timers &timers, const std::string &name);
  Scoped_Timer(Timers &timers, const std::string &name, Timer_Attrs attrs);
  virtual ~Scoped_Timer();

  [[nodiscard]] std::chrono::time_point<std::chrono::high_resolution_clock>
  start_time() const;

  void stop();
  [[nodiscard]] const Timer &timer() const;
  [[nodiscard]] int64_t elapsed_milliseconds() const;
  [[nodiscard]] int64_t elapsed_nanoseconds() const;
  // Add an attribute after construction (e.g. a size known only later)
  void add_attr(const std::string &key, const std::string &value);
  template <class T> void add_attr(const std::string &key, const T &value)
  {
    add_attr(key, El::BuildString(value));
  }

private:
  Timers &timers;
  // The recorded entry, or nullptr when the timers are off
  // (then own_timer measures this scope).
  Timer_Entry *entry;
  std::optional<Timer> own_timer;
  std::string old_prefix;
  std::string new_prefix;

  [[nodiscard]] bool is_running() const;
  [[nodiscard]] Timer &active_timer();
  [[nodiscard]] const Timer &active_timer() const;
};

// Timer for loops: one entry in the profile, many timed scopes.
//
//   Accumulating_Timer acc(timers, "gemm_tile", {{"kind", "gemm"}});
//   for(...)
//     {
//       auto scope = acc.scope();
//       El::Gemm(...);
//     }
//
// Unlike Scoped_Timer it does not push a prefix: scopes cannot have children.
// The entry's timer runs from construction to destruction (envelope);
// the reported elapsed time is the sum of the scopes.
struct Accumulating_Timer : boost::noncopyable
{
  struct Scope : boost::noncopyable
  {
    explicit Scope(Accumulating_Timer &owner);
    ~Scope();

  private:
    Accumulating_Timer &owner;
    std::chrono::steady_clock::time_point start;
    int64_t cpu_start = -1;
  };

  Accumulating_Timer(Timers &timers, const std::string &name,
                     Timer_Attrs attrs = {});
  virtual ~Accumulating_Timer();

  [[nodiscard]] Scope scope();
  void stop();
  void add_attr(const std::string &key, const std::string &value);
  template <class T> void add_attr(const std::string &key, const T &value)
  {
    add_attr(key, El::BuildString(value));
  }
  [[nodiscard]] int64_t count() const;
  [[nodiscard]] int64_t elapsed_nanoseconds() const;

private:
  // nullptr when the timers are off: scopes then measure nothing
  Timer_Entry *entry;
  void add(int64_t wall_ns, int64_t cpu_ns);
};
