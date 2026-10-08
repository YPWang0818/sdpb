#include "Timers.hxx"

#include "sdpb_util/Proc_Meminfo.hxx"
#include "sdpb_util/assert.hxx"

#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

#include <fstream>
#include <unistd.h>

namespace fs = std::filesystem;

namespace
{
  // Convert bytes to gigabytes
  double to_GB(const size_t bytes)
  {
    return static_cast<double>(bytes) / 1024 / 1024 / 1024;
  }

  int64_t unix_now_ns()
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
  }

  int64_t steady_ns(const std::chrono::steady_clock::time_point &t)
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
             t.time_since_epoch())
      .count();
  }

  std::string get_hostname()
  {
    char buf[256] = {0};
    if(gethostname(buf, sizeof(buf) - 1) != 0)
      return "";
    return buf;
  }

  template <class TWriter>
  void write_string(TWriter &writer, const std::string &s)
  {
    writer.String(s.c_str(), s.size());
  }
  template <class TWriter>
  void write_key(TWriter &writer, const std::string &s)
  {
    writer.Key(s.c_str(), s.size());
  }
}

int64_t Timer_Entry::elapsed_nanoseconds() const
{
  return is_accumulator ? accumulated_ns : timer.elapsed_nanoseconds();
}
int64_t Timer_Entry::cpu_nanoseconds() const
{
  return is_accumulator ? accumulated_cpu_ns : timer.cpu_nanoseconds();
}

Timers::Timers()
    : t0_steady(std::chrono::steady_clock::now()), t0_unix_ns(unix_now_ns())
{}
Timers::Timers(const Environment &env, const Verbosity &verbosity) : Timers()
{
  this->verbosity = verbosity;
  node_index = env.node_index();
  num_nodes = env.num_nodes();
  node_rank = env.comm_shared_mem.Rank();
  if(node_rank != 0)
    {
      // Print info only from the first rank on a node
      this->verbosity = Verbosity::none;
      if(num_nodes != 1)
        node_debug_prefix = El::BuildString("node=", node_index, " ");
    }
}

Timers::~Timers() noexcept
{
  try
    {
      if(verbosity >= Verbosity::debug)
        print_max_mem_used();
    }
  catch(...)
    {
      // destructors should never throw exceptions
    }
}

Timer_Entry &Timers::add_entry(const std::string &name, Timer_Attrs attrs)
{
  const std::string full_name = prefix + name;
  const int depth = static_cast<int>(open_ids.size());

  // Read /proc/meminfo before starting the timer,
  // so that it is not counted.
  process_meminfo(full_name, depth);

  auto &entry = named_timers.emplace_back();
  entry.name = name;
  entry.full_name = full_name;
  entry.id = static_cast<int64_t>(named_timers.size()) - 1;
  entry.parent = open_ids.empty() ? -1 : open_ids.back();
  entry.depth = depth;
  entry.attrs = std::move(attrs);
  return entry;
}
Timer_Entry &Timers::add_and_start(const std::string &name, Timer_Attrs attrs)
{
  auto &entry = add_entry(name, std::move(attrs));
  open_ids.push_back(entry.id);
  return entry;
}
Timer_Entry &Timers::add_accumulator(const std::string &name,
                                     Timer_Attrs attrs)
{
  auto &entry = add_entry(name, std::move(attrs));
  entry.is_accumulator = true;
  entry.count = 0;
  return entry;
}

void Timers::write_profile(const fs::path &path) const
{
  if(path.has_parent_path())
    fs::create_directories(path.parent_path());
  const fs::path tmp_path = path.string() + ".tmp";
  {
    std::ofstream f(tmp_path);
    ASSERT(f.good(), "Cannot open for writing: ", tmp_path);
    rapidjson::OStreamWrapper os(f);
    rapidjson::Writer<rapidjson::OStreamWrapper> writer(os);

    writer.StartObject();
    writer.Key("schema_version");
    writer.Int(1);
    writer.Key("sdpb_version");
#ifdef SDPB_VERSION_STRING
    writer.String(SDPB_VERSION_STRING);
#else
    writer.String("");
#endif
    writer.Key("rank");
    writer.Int(El::mpi::Rank());
    writer.Key("num_ranks");
    writer.Int(El::mpi::Size());
    writer.Key("node");
    writer.Int(node_index);
    writer.Key("node_rank");
    writer.Int(node_rank);
    writer.Key("num_nodes");
    writer.Int(num_nodes);
    writer.Key("hostname");
    write_string(writer, get_hostname());
    writer.Key("precision_bits");
    writer.Uint64(El::gmp::Precision());
    writer.Key("t0_unix_ns");
    writer.Int64(t0_unix_ns);
    writer.Key("t0_monotonic_ns");
    writer.Int64(steady_ns(t0_steady));
    writer.Key("cpu_time");
    writer.Bool(Timer::thread_cpu_now_ns() >= 0);

    writer.Key("meta");
    writer.StartObject();
    for(const auto &[key, json_value] : meta)
      {
        write_key(writer, key);
        writer.RawValue(json_value.c_str(), json_value.size(),
                        rapidjson::kObjectType);
      }
    writer.EndObject();

    writer.Key("timers");
    writer.StartArray();
    for(const auto &entry : named_timers)
      {
        writer.StartObject();
        writer.Key("id");
        writer.Int64(entry.id);
        writer.Key("parent");
        writer.Int64(entry.parent);
        writer.Key("depth");
        writer.Int(entry.depth);
        writer.Key("name");
        write_string(writer, entry.name);
        writer.Key("start");
        writer.Int64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                       entry.timer.steady_start_time() - t0_steady)
                       .count());
        writer.Key("elapsed");
        writer.Int64(entry.elapsed_nanoseconds());
        writer.Key("cpu");
        writer.Int64(entry.cpu_nanoseconds());
        writer.Key("count");
        writer.Int64(entry.count);
        if(entry.is_accumulator)
          {
            writer.Key("max");
            writer.Int64(entry.max_ns);
          }
        writer.Key("attrs");
        writer.StartObject();
        for(const auto &[key, value] : entry.attrs)
          {
            write_key(writer, key);
            write_string(writer, value);
          }
        writer.EndObject();
        writer.EndObject();
      }
    writer.EndArray();
    writer.EndObject();
    os.Flush();
    ASSERT(f.good(), "Error when writing to: ", tmp_path);
  }
  fs::rename(tmp_path, path);
}

int64_t Timers::elapsed_milliseconds(const std::string &s) const
{
  auto iter(std::find_if(
    named_timers.rbegin(), named_timers.rend(),
    [&s](const Timer_Entry &entry) { return entry.full_name == s; }));
  ASSERT(iter != named_timers.rend(), "Could not find timing for ", s);
  return iter->elapsed_nanoseconds() / 1000000;
}

void Timers::set_meta(const std::string &key, const std::string &json_value)
{
  meta[key] = json_value;
}
const std::map<std::string, std::string> &Timers::get_meta() const
{
  return meta;
}
void Timers::set_autosave_path(const fs::path &path)
{
  autosave_path = path;
}
const fs::path &Timers::get_autosave_path() const
{
  return autosave_path;
}
void Timers::flush() const noexcept
{
  if(autosave_path.empty())
    return;
  try
    {
      write_profile(autosave_path);
    }
  catch(std::exception &e)
    {
      try
        {
          El::Output("Warning: cannot write profile to ", autosave_path, ": ",
                     e.what());
        }
      catch(...)
        {}
    }
  catch(...)
    {}
}
void Timers::set_detail(const int level)
{
  detail_level = level;
}
int Timers::detail() const
{
  return detail_level;
}
bool Timers::enabled() const
{
  return detail_level != detail_off;
}
const std::list<Timer_Entry> &Timers::entries() const
{
  return named_timers;
}
std::chrono::steady_clock::time_point Timers::t0() const
{
  return t0_steady;
}

void Timers::print_max_mem_used() const
{
  if(max_mem_used > 0 && !max_mem_used_name.empty())
    {
      El::Output(node_debug_prefix, "max MemUsed: ", to_GB(max_mem_used),
                 " GB at \"", max_mem_used_name, "\"");
    }
}

// For --verbosity=trace:
// Print memory usage for the current node (from the first rank).
// If we cannot parse /proc/meminfo, then simply print timer name.
//
// In addition, for --verbosity=debug we update max MemUsed (will be printed in the end).
// To keep the overhead of fine-grained timers low, at --verbosity=debug
// only timers up to depth 4 (e.g. sdpb.solve.run.iter.step.<phase>) are sampled.
void Timers::process_meminfo(const std::string &name, const int depth)
{
  // Do not collect memory info for lower verbosity levels, it will not be used
  if(verbosity < Verbosity::debug)
    return;
  if(verbosity < Verbosity::trace && depth > 4)
    return;

  const auto prefix = El::BuildString(node_debug_prefix, "start ", name, " ");
  if(!can_read_meminfo)
    {
      // Print "start timer" without MemUsed
      if(verbosity >= Verbosity::trace)
        El::Output(prefix);
      return;
    }

  // Read /proc/meminfo
  constexpr bool print_error_msg = true;
  const auto meminfo
    = Proc_Meminfo::try_read(can_read_meminfo, print_error_msg);

  // Update max MemUsed info, it will be printed for --verbosity=debug
  if(meminfo.mem_used() > max_mem_used)
    {
      max_mem_used = meminfo.mem_used();
      max_mem_used_name = name;
    }

  // Print "start timer", only for --verbosity=trace
  if(verbosity < Verbosity::trace)
    return;

  // MemTotal is constant, thus we print it only once, when adding first timer
  if(named_timers.empty())
    {
      El::Output(prefix, "--- MemTotal: ", to_GB(meminfo.mem_total), " GB");
    }

  //Print MemUsed each time
  El::Output(prefix, "--- MemUsed: ", to_GB(meminfo.mem_used()), " GB");
}
