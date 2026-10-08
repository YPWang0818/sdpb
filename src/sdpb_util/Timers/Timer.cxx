#include "Timer.hxx"

#include "sdpb_util/assert.hxx"

#include <ctime>

Timer::Timer()
    : start_time(now()), steady_start(std::chrono::steady_clock::now()),
      cpu_start(thread_cpu_now_ns())
{}
void Timer::stop()
{
  ASSERT(is_running());
  stop_time = now();
  steady_stop = std::chrono::steady_clock::now();
  cpu_stop = thread_cpu_now_ns();
  running = false;
}
int64_t Timer::elapsed_milliseconds() const
{
  return elapsed_nanoseconds() / 1000000;
}
int64_t Timer::elapsed_seconds() const
{
  return elapsed_nanoseconds() / 1000000000;
}
int64_t Timer::elapsed_nanoseconds() const
{
  const auto end
    = is_running() ? std::chrono::steady_clock::now() : steady_stop;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end
                                                              - steady_start)
    .count();
}
int64_t Timer::cpu_nanoseconds() const
{
  if(cpu_start < 0)
    return -1;
  const int64_t end = is_running() ? thread_cpu_now_ns() : cpu_stop;
  if(end < 0)
    return -1;
  return end - cpu_start;
}
std::chrono::steady_clock::time_point Timer::steady_start_time() const
{
  return steady_start;
}
bool Timer::is_running() const
{
  return running;
}
int64_t Timer::thread_cpu_now_ns()
{
#ifdef CLOCK_THREAD_CPUTIME_ID
  timespec ts{};
  if(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0)
    return -1;
  return static_cast<int64_t>(ts.tv_sec) * 1000000000 + ts.tv_nsec;
#else
  return -1;
#endif
}
std::chrono::time_point<std::chrono::high_resolution_clock> Timer::now()
{
  return std::chrono::high_resolution_clock::now();
}
std::ostream &operator<<(std::ostream &os, const Timer &timer)
{
  os << (timer.elapsed_milliseconds() / 1000.0);
  return os;
}
