#include "Timers.hxx"

Accumulating_Timer::Accumulating_Timer(Timers &timers, const std::string &name,
                                       Timer_Attrs attrs)
    : entry(timers.add_accumulator(name, std::move(attrs)))
{}
Accumulating_Timer::~Accumulating_Timer()
{
  if(entry.timer.is_running())
    stop();
}
Accumulating_Timer::Scope Accumulating_Timer::scope()
{
  return Scope(*this);
}
void Accumulating_Timer::stop()
{
  entry.timer.stop();
}
void Accumulating_Timer::add_attr(const std::string &key,
                                  const std::string &value)
{
  entry.attrs.emplace_back(key, value);
}
int64_t Accumulating_Timer::count() const
{
  return entry.count;
}
int64_t Accumulating_Timer::elapsed_nanoseconds() const
{
  return entry.accumulated_ns;
}
void Accumulating_Timer::add(const int64_t wall_ns, const int64_t cpu_ns)
{
  ++entry.count;
  entry.accumulated_ns += wall_ns;
  if(cpu_ns >= 0)
    entry.accumulated_cpu_ns += cpu_ns;
  entry.max_ns = std::max(entry.max_ns, wall_ns);
}

Accumulating_Timer::Scope::Scope(Accumulating_Timer &owner)
    : owner(owner), start(std::chrono::steady_clock::now()),
      cpu_start(Timer::thread_cpu_now_ns())
{}
Accumulating_Timer::Scope::~Scope()
{
  const int64_t wall_ns
    = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start)
        .count();
  int64_t cpu_ns = -1;
  if(cpu_start >= 0)
    {
      const int64_t cpu_now = Timer::thread_cpu_now_ns();
      if(cpu_now >= 0)
        cpu_ns = cpu_now - cpu_start;
    }
  owner.add(wall_ns, cpu_ns);
}
