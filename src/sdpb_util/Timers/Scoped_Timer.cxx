#include "Timers.hxx"
#include "sdpb_util/assert.hxx"

Scoped_Timer::Scoped_Timer(Timers &timers, const std::string &name)
    : Scoped_Timer(timers, name, Timer_Attrs{})
{}
Scoped_Timer::Scoped_Timer(Timers &timers, const std::string &name,
                           Timer_Attrs attrs)
    : timers(timers),
      entry(timers.enabled() ? &timers.add_and_start(name, std::move(attrs))
                             : nullptr)
{
  if(entry == nullptr)
    {
      own_timer.emplace();
      return;
    }
  old_prefix = timers.prefix;
  new_prefix = timers.prefix + name + ".";
  timers.prefix = new_prefix;
}
Scoped_Timer::~Scoped_Timer()
{
  if(is_running())
    stop();
}
Timer &Scoped_Timer::active_timer()
{
  return entry != nullptr ? entry->timer : *own_timer;
}
const Timer &Scoped_Timer::active_timer() const
{
  return entry != nullptr ? entry->timer : *own_timer;
}
std::chrono::time_point<std::chrono::high_resolution_clock>
Scoped_Timer::start_time() const
{
  return active_timer().start_time;
}
void Scoped_Timer::stop()
{
  ASSERT(is_running(), "Timer '" + new_prefix + "' already stopped!");
  if(entry == nullptr)
    {
      own_timer->stop();
      return;
    }
  // These assertions fail if some of the nested timers is still running.
  // They come before stopping the timer, so that a failed stop() leaves
  // the timer in a consistent (still running) state.
  ASSERT_EQUAL(timers.prefix, new_prefix);
  ASSERT(!timers.open_ids.empty() && timers.open_ids.back() == entry->id,
         "Timer '", new_prefix,
         "' stopped while a nested timer is still running");
  entry->timer.stop();
  timers.open_ids.pop_back();
  timers.prefix = old_prefix;
}
const Timer &Scoped_Timer::timer() const
{
  return active_timer();
}
bool Scoped_Timer::is_running() const
{
  return active_timer().is_running();
}
int64_t Scoped_Timer::elapsed_milliseconds() const
{
  return active_timer().elapsed_milliseconds();
}
int64_t Scoped_Timer::elapsed_nanoseconds() const
{
  return active_timer().elapsed_nanoseconds();
}
void Scoped_Timer::add_attr(const std::string &key, const std::string &value)
{
  if(entry != nullptr)
    entry->attrs.emplace_back(key, value);
}
