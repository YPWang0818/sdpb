// Compiled instead of the SDP file readers when SDPB is configured with
// `./waf configure --libs-only` (see wscript: archive_sources).
//
// SDP.cxx and Block_Info.cxx define the constructors that read an SDP
// directory or sdp.zip next to the in-memory ones, so the reading entry points
// must exist at link time. In a --libs-only build they throw: that build has
// no libarchive, and an embedding program constructs Block_Info and SDP in
// memory (see api-doc.md).
//
// The signatures must match the definitions in Block_Info/read_block_info.cxx,
// SDP/read_objectives.cxx, SDP/read_normalization.cxx and
// SDP/read_block_data/read_block_data.cxx; a mismatch shows up as an undefined
// symbol when the embedding program is linked.

#include "sdp_solve/Block_Info.hxx"
#include "sdp_solve/SDP.hxx"
#include "sdpb_util/assert.hxx"

#include <filesystem>
#include <optional>
#include <vector>

namespace fs = std::filesystem;

namespace
{
  [[noreturn]] void file_input_disabled(const fs::path &sdp_path)
  {
    RUNTIME_ERROR("SDPB was built with --libs-only: reading an SDP from files "
                  "is not available. sdp_path=",
                  sdp_path);
  }
}

void Block_Info::read_block_info(const fs::path &sdp_path)
{
  file_input_disabled(sdp_path);
}

void read_objectives(const fs::path &sdp_path, const El::Grid &,
                     El::BigFloat &, El::DistMatrix<El::BigFloat> &, Timers &)
{
  file_input_disabled(sdp_path);
}

std::optional<std::vector<El::BigFloat>>
read_normalization(const fs::path &sdp_path, Timers &)
{
  file_input_disabled(sdp_path);
}

void read_block_data(const fs::path &sdp_path, const El::Grid &,
                     const Block_Info &, SDP &, Timers &)
{
  file_input_disabled(sdp_path);
}
