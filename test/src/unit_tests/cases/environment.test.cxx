#include "sdpb_util/Environment.hxx"

#include <catch2/catch_amalgamated.hpp>

TEST_CASE("environment")
{
  SECTION("request_termination")
  {
    INFO("request_termination() should behave as receiving SIGTERM");
    // Nested Environment is fine: Elemental counts initializations.
    Environment env;
    REQUIRE(!env.sigterm_received());
    Environment::request_termination();
    REQUIRE(env.sigterm_received());
    Environment::clear_termination_request();
    REQUIRE(!env.sigterm_received());
  }
}
