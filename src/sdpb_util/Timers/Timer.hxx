#pragma once

#include <chrono>
#include <cstdint>
#include <iostream>

struct Timer
{
  // Kept for API compatibility (e.g. Scoped_Timer::start_time()).
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time,
    stop_time;
  Timer();
  void stop();

  [[nodiscard]] int64_t elapsed_milliseconds() const;
  [[nodiscard]] int64_t elapsed_seconds() const;
  // Wall-clock time (monotonic clock) between start and stop (or now).
  [[nodiscard]] int64_t elapsed_nanoseconds() const;
  // CPU time consumed by the calling thread between start and stop (or now).
  // Returns -1 if thread CPU time is unavailable on this platform.
  // NB: with MPI implementations that busy-poll (e.g. Open MPI),
  // CPU time inside MPI waits is close to wall time.
  [[nodiscard]] int64_t cpu_nanoseconds() const;
  [[nodiscard]] std::chrono::steady_clock::time_point steady_start_time() const;

  [[nodiscard]] bool is_running() const;

  // Current thread CPU time in nanoseconds, or -1 if unavailable.
  static int64_t thread_cpu_now_ns();

private:
  bool running = true;
  std::chrono::steady_clock::time_point steady_start, steady_stop;
  int64_t cpu_start = -1, cpu_stop = -1;

  static std::chrono::time_point<std::chrono::high_resolution_clock> now();
};

std::ostream &operator<<(std::ostream &os, const Timer &timer);
