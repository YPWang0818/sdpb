#include "sdpb_util/Timers/Timers.hxx"
#include "sdpb_util/Environment.hxx"

#include <catch2/catch_amalgamated.hpp>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/error/en.h>

#include <filesystem>
#include <fstream>
#include <thread>

namespace fs = std::filesystem;

namespace
{
  rapidjson::Document read_json(const fs::path &path)
  {
    std::ifstream is(path);
    REQUIRE(is.good());
    rapidjson::IStreamWrapper wrapper(is);
    rapidjson::Document document;
    document.ParseStream(wrapper);
    INFO("JSON parse error: " << rapidjson::GetParseError_En(
           document.GetParseError()));
    REQUIRE(!document.HasParseError());
    return document;
  }

  void sleep_ms(const int ms)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  }
}

TEST_CASE("timers")
{
  Environment env;
  Timers timers(env, Verbosity::none);
  timers.set_meta("precision", "768");
  timers.set_meta("blocks", R"([{"index":0,"dim":1}])");

  // Build a small tree:
  // root
  // ├─ foo {kind:gemm, m:3}
  // │   └─ inner (accumulator, 3 scopes)
  // └─ bar
  {
    Scoped_Timer root(timers, "root");
    {
      Scoped_Timer foo(timers, "foo", {{"kind", "gemm"}, {"m", "3"}});
      foo.add_attr("n", 4);
      Accumulating_Timer inner(timers, "inner", {{"kind", "trsm"}});
      for(int i = 0; i < 3; ++i)
        {
          auto scope = inner.scope();
          sleep_ms(2);
        }
      REQUIRE(inner.count() == 3);
      REQUIRE(inner.elapsed_nanoseconds() >= 3 * 2000000);
    }
    Scoped_Timer bar(timers, "bar");
    sleep_ms(1);
  }

  SECTION("legacy lookup by full name")
  {
    REQUIRE(timers.elapsed_milliseconds("root") >= 7);
    REQUIRE(timers.elapsed_milliseconds("root.foo") >= 6);
    REQUIRE(timers.elapsed_milliseconds("root.bar") >= 1);
    REQUIRE_THROWS(timers.elapsed_milliseconds("does.not.exist"));
  }

  SECTION("tree structure")
  {
    const auto &entries = timers.entries();
    REQUIRE(entries.size() == 4);
    auto it = entries.begin();
    const auto &root = *it++;
    const auto &foo = *it++;
    const auto &inner = *it++;
    const auto &bar = *it++;
    REQUIRE(root.name == "root");
    REQUIRE(root.parent == -1);
    REQUIRE(root.depth == 0);
    REQUIRE(foo.parent == root.id);
    REQUIRE(foo.depth == 1);
    REQUIRE(inner.parent == foo.id);
    REQUIRE(inner.depth == 2);
    REQUIRE(inner.is_accumulator);
    REQUIRE(inner.count == 3);
    REQUIRE(bar.parent == root.id);
    REQUIRE(bar.depth == 1);
    // children do not exceed parent
    REQUIRE(foo.elapsed_nanoseconds() + bar.elapsed_nanoseconds()
            <= root.elapsed_nanoseconds());
    REQUIRE(inner.elapsed_nanoseconds() <= foo.elapsed_nanoseconds());
  }

  SECTION("write_profile produces valid JSON")
  {
    const fs::path dir = fs::temp_directory_path() / "sdpb_unit_tests"
                         / "timers" / std::to_string(El::mpi::Rank());
    fs::remove_all(dir);
    const fs::path path = dir / "profiling.json";
    timers.write_profile(path);
    REQUIRE(fs::exists(path));
    REQUIRE(!fs::exists(path.string() + ".tmp"));

    auto document = read_json(path);
    REQUIRE(document.IsObject());
    REQUIRE(document["schema_version"].GetInt() == 1);
    REQUIRE(document["rank"].GetInt() == El::mpi::Rank());
    REQUIRE(document["num_ranks"].GetInt() == El::mpi::Size());
    REQUIRE(document["meta"]["precision"].GetInt() == 768);
    REQUIRE(document["meta"]["blocks"][0]["dim"].GetInt() == 1);

    const auto &json_timers = document["timers"];
    REQUIRE(json_timers.IsArray());
    REQUIRE(json_timers.Size() == 4);

    const auto &root = json_timers[0];
    REQUIRE(std::string(root["name"].GetString()) == "root");
    REQUIRE(root["parent"].GetInt64() == -1);
    REQUIRE(root["start"].GetInt64() >= 0);
    REQUIRE(root["count"].GetInt64() == 1);

    const auto &foo = json_timers[1];
    REQUIRE(foo["parent"].GetInt64() == root["id"].GetInt64());
    REQUIRE(std::string(foo["attrs"]["kind"].GetString()) == "gemm");
    REQUIRE(std::string(foo["attrs"]["m"].GetString()) == "3");
    REQUIRE(std::string(foo["attrs"]["n"].GetString()) == "4");

    const auto &inner = json_timers[2];
    REQUIRE(inner["count"].GetInt64() == 3);
    REQUIRE(inner.HasMember("max"));
    REQUIRE(inner["max"].GetInt64() <= inner["elapsed"].GetInt64());
    REQUIRE(inner["elapsed"].GetInt64() <= foo["elapsed"].GetInt64());

    if(document["cpu_time"].GetBool())
      {
        REQUIRE(root["cpu"].GetInt64() >= 0);
      }
  }

  SECTION("flush writes to the autosave path, also for running timers")
  {
    const fs::path dir = fs::temp_directory_path() / "sdpb_unit_tests"
                         / "timers_flush" / std::to_string(El::mpi::Rank());
    fs::remove_all(dir);
    const fs::path path = dir / "profiling.0";
    // No autosave path: flush() is a no-op
    timers.flush();
    REQUIRE(!fs::exists(path));

    timers.set_autosave_path(path);
    Scoped_Timer running(timers, "running");
    sleep_ms(1);
    timers.flush();
    REQUIRE(fs::exists(path));
    auto document = read_json(path);
    const auto &json_timers = document["timers"];
    REQUIRE(json_timers.Size() == 5);
    REQUIRE(std::string(json_timers[4]["name"].GetString()) == "running");
    REQUIRE(json_timers[4]["elapsed"].GetInt64() >= 1000000);
  }

  SECTION("stopping a timer while a nested one runs is an error")
  {
    Scoped_Timer outer(timers, "outer");
    Scoped_Timer nested(timers, "nested");
    REQUIRE_THROWS(outer.stop());
    nested.stop();
    outer.stop();
  }
}

TEST_CASE("timers off")
{
  Environment env;
  Timers timers(env, Verbosity::none);
  timers.set_detail(Timers::detail_off);
  REQUIRE(!timers.enabled());
  {
    Scoped_Timer root(timers, "root", {{"kind", "gemm"}});
    root.add_attr("m", 3);
    Accumulating_Timer inner(timers, "inner");
    for(int i = 0; i < 3; ++i)
      {
        auto scope = inner.scope();
        sleep_ms(1);
      }
    REQUIRE(inner.count() == 0);
    REQUIRE(inner.elapsed_nanoseconds() == 0);
    Scoped_Timer nested(timers, "nested");
    sleep_ms(2);
    nested.stop();
    // Scoped timers still measure their own scope
    REQUIRE(nested.elapsed_milliseconds() >= 2);
    REQUIRE(root.elapsed_milliseconds() >= 5);
  }
  // ... but nothing is recorded
  REQUIRE(timers.entries().empty());
  REQUIRE_THROWS(timers.elapsed_milliseconds("root"));
}
