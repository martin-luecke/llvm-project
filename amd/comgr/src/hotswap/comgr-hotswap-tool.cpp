// Optional HSA_TOOLS_LIB hotswap tool.
//
// Enabled only by pointing HSA_TOOLS_LIB at this library, e.g.:
//   HSA_TOOLS_LIB=/path/libamd_comgr_hotswap_tool.so ./my_app
//
// When loaded, libhsa-runtime hands each code object to this tool before
// dispatch. For each object the tool asks comgr to adapt it for the running
// device, or passes it through unchanged:
//   co ISA != device ISA  -> transpile (cross-ISA)
//   co ISA == device ISA  -> in-place rewrite, if the device is treated as the
//                            rewrite target (see below)
//   otherwise             -> pass through, untouched
//
// Choosing the rewrite target is intentionally simple and is NOT robust A0/B0
// stepping detection (that is separate work). The tool reads
// HSA_AMD_AGENT_INFO_ASIC_REVISION and rewrites only when it equals
// HSA_HOTSWAP_A0_REVISION (default 0), or when HSA_HOTSWAP_FORCE_STEPPING_REWRITE
// is set.
//
// Other optional env: HSA_HOTSWAP_CACHE_DIR (comgr cache),
// HSA_HOTSWAP_TARGET_OVERRIDE=gfxNNNN, HSA_HOTSWAP_TOOL_VERBOSE=1.
//
// comgr runs in-process here.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "inc/hsa.h"
#include "inc/hsa_ext_amd.h"
#include "inc/hsa_api_trace.h"
#include <amd_comgr.h>

namespace {

// Loader entry we wrap + agent/isa queries we need, captured from the table.
decltype(hsa_code_object_reader_create_from_memory)* g_real_reader_create = nullptr;
decltype(hsa_iterate_agents)* g_iterate_agents = nullptr;
decltype(hsa_agent_get_info)* g_agent_get_info = nullptr;
decltype(hsa_isa_get_info_alt)* g_isa_get_info_alt = nullptr;

// Device facts, resolved once lazily on first load (HSA fully up by then).
std::once_flag g_detect_once;
std::string g_device_gfx;     // e.g. "gfx950" or "gfx1250"
uint32_t g_device_rev = 0;
bool g_device_is_a0 = false;

// Config.
std::string g_cache_dir;
uint32_t g_a0_revision = 0;
bool g_force_stepping_rewrite = false;
std::string g_target_override;
bool g_verbose = false;

// reader_create REFERENCES the bytes for the module's lifetime, so transpiled /
// rewritten buffers must outlive the wrapper call. Retain for process lifetime.
std::mutex g_retain_mu;
std::vector<std::vector<uint8_t>> g_retained;

#define LOGF(...)                                                             \
  do {                                                                        \
    if (g_verbose) {                                                          \
      std::fprintf(stderr, "hotswap_tool: " __VA_ARGS__);                     \
      std::fprintf(stderr, "\n");                                             \
    }                                                                         \
  } while (0)

const char* GfxFromMach(uint8_t mach) {
  // Only the arches HotSwap acts on; any other mach returns nullptr and the
  // code object is passed through untouched.
  switch (mach) {
    case 0x49: return "gfx1250";  // transpile source / rewrite target
    case 0x4f: return "gfx950";   // transpile target
    default: return nullptr;
  }
}

std::string ExtractGfx(const char* isa_name) {
  if (!isa_name) return {};
  std::string s(isa_name);
  size_t g = s.find("gfx");
  if (g == std::string::npos) return {};
  size_t e = g;
  while (e < s.size() && s[e] != ':' && s[e] != '\0') ++e;
  return s.substr(g, e - g);
}

// ELF fields below are read at fixed offsets, not via llvm::object::ELFFile:
// this standalone tool links amd_comgr (whose static LLVM is symbol-hidden), so
// it cannot reuse LLVM, and bundling its own LLVMObject would risk a second
// in-process LLVM. We only touch stable ELF64 header fields (magic, EI_CLASS,
// e_flags mach) plus a custom e_ident marker that ELFFile cannot read anyway.
bool IsElf(const void* p, size_t n) {
  if (n < 64) return false;
  auto* b = static_cast<const uint8_t*>(p);
  return b[0] == 0x7f && b[1] == 'E' && b[2] == 'L' && b[3] == 'F' && b[4] == 2;
}

hsa_status_t FindGpuCb(hsa_agent_t agent, void* data) {
  hsa_device_type_t dt;
  if (g_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dt) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if (dt != HSA_DEVICE_TYPE_GPU) return HSA_STATUS_SUCCESS;
  *static_cast<hsa_agent_t*>(data) = agent;
  return HSA_STATUS_INFO_BREAK;
}

void DetectDevice() {
  if (!g_iterate_agents || !g_agent_get_info || !g_isa_get_info_alt) return;
  hsa_agent_t gpu = {0};
  g_iterate_agents(&FindGpuCb, &gpu);
  if (gpu.handle == 0) {
    std::fprintf(stderr, "hotswap_tool: no GPU agent found; tool inert\n");
    return;
  }
  hsa_isa_t isa = {0};
  char name[128] = {0};
  if (g_agent_get_info(gpu, HSA_AGENT_INFO_ISA, &isa) == HSA_STATUS_SUCCESS &&
      g_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, name) == HSA_STATUS_SUCCESS) {
    g_device_gfx = ExtractGfx(name);
  }
  g_agent_get_info(gpu,
                   static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_ASIC_REVISION),
                   &g_device_rev);
  if (!g_target_override.empty()) g_device_gfx = g_target_override;

  // Stepping rewrite is gfx1250-specific; only arm on a gfx1250 board so a
  // non-gfx1250 board reporting revision 0 doesn't rewrite its native COs.
  g_device_is_a0 =
      (g_device_gfx == "gfx1250") &&
      (g_force_stepping_rewrite || (g_device_rev == g_a0_revision));

  std::fprintf(stderr,
               "hotswap_tool: device=%s asic_revision=%u -> %s%s\n",
               g_device_gfx.empty() ? "(unknown)" : g_device_gfx.c_str(),
               g_device_rev,
               g_device_is_a0 ? "treat as A0 (stepping rewrite armed)"
                              : "treat as B0/native",
               g_force_stepping_rewrite ? " [FORCED]" : "");
}

enum Action { PASS, REWRITE, TRANSPILE };

// Run comgr to produce target bytes. `rewrite` selects the byte-level stepping
// patcher (identity ISA) vs the full cross-ISA transpiler.
bool RunComgr(const std::vector<uint8_t>& src, const std::string& src_gfx,
              const std::string& tgt_gfx, bool rewrite,
              std::vector<uint8_t>* out) {
  const std::string src_isa = "amdgcn-amd-amdhsa--" + src_gfx;
  const std::string tgt_isa = "amdgcn-amd-amdhsa--" + tgt_gfx;
  amd_comgr_data_t input = {0};
  if (amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &input) !=
      AMD_COMGR_STATUS_SUCCESS)
    return false;
  if (amd_comgr_set_data(input, src.size(),
                         reinterpret_cast<const char*>(src.data())) !=
      AMD_COMGR_STATUS_SUCCESS) {
    amd_comgr_release_data(input);
    return false;
  }
  amd_comgr_data_t output = {0};
  amd_comgr_status_t st;
  if (rewrite) {
    st = amd_comgr_hotswap_rewrite(input, src_isa.c_str(), tgt_isa.c_str(),
                                   &output);
  } else {
    amd_comgr_hotswap_transpile_options_t opts = {};
    opts.size = sizeof(opts);
    opts.cache_directory = g_cache_dir.empty() ? nullptr : g_cache_dir.c_str();
    amd_comgr_hotswap_transpile_result_t result = {0};
    st = amd_comgr_hotswap_transpile_with_options(
        input, src_isa.c_str(), tgt_isa.c_str(), &opts, &output, &result);
    if (result.handle && st != AMD_COMGR_STATUS_SUCCESS) {
      auto dump = [&](amd_comgr_hotswap_transpile_result_string_t f,
                      const char* label) {
        size_t n = 0;
        if (amd_comgr_hotswap_transpile_result_get_string(result, f, &n,
                                                          nullptr) ==
                AMD_COMGR_STATUS_SUCCESS && n > 1) {
          std::string buf(n, '\0');
          if (amd_comgr_hotswap_transpile_result_get_string(result, f, &n,
                                                            buf.data()) ==
              AMD_COMGR_STATUS_SUCCESS)
            LOGF("comgr transpile %s: %s", label, buf.c_str());
        }
      };
      dump(AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_FAIL_REASON, "FAIL_REASON");
      dump(AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_FAIL_DETAIL, "FAIL_DETAIL");
    }
    if (result.handle) amd_comgr_destroy_hotswap_transpile_result(result);
  }
  amd_comgr_release_data(input);
  if (st != AMD_COMGR_STATUS_SUCCESS) {
    LOGF("comgr %s %s->%s FAILED (status=%d)", rewrite ? "rewrite" : "transpile",
         src_gfx.c_str(), tgt_gfx.c_str(), (int)st);
    if (output.handle) amd_comgr_release_data(output);
    return false;
  }
  size_t osz = 0;
  if (amd_comgr_get_data(output, &osz, nullptr) != AMD_COMGR_STATUS_SUCCESS ||
      osz == 0) {
    amd_comgr_release_data(output);
    return false;
  }
  out->resize(osz);
  st = amd_comgr_get_data(output, &osz, reinterpret_cast<char*>(out->data()));
  amd_comgr_release_data(output);
  if (st != AMD_COMGR_STATUS_SUCCESS) return false;
  LOGF("comgr %s %s->%s ok (%zu->%zu)", rewrite ? "rewrite" : "transpile",
       src_gfx.c_str(), tgt_gfx.c_str(), src.size(), osz);
  return true;
}

hsa_status_t ReaderCreateWrapper(const void* code_object, size_t size,
                                 hsa_code_object_reader_t* reader) {
  std::call_once(g_detect_once, DetectDevice);

  if (!IsElf(code_object, size) || g_device_gfx.empty())
    return g_real_reader_create(code_object, size, reader);

  auto* b = static_cast<const uint8_t*>(code_object);
  const bool marked = (b[9] == 'S' && b[10] == 'L');
  uint8_t src_mach = marked ? b[11] : b[48];
  const char* src_gfx_c = GfxFromMach(src_mach);
  if (!src_gfx_c) {
    LOGF("reader_create: unhandled mach=0x%02x -> pass-through (size=%zu)",
         src_mach, size);
    return g_real_reader_create(code_object, size, reader);
  }
  std::string src_gfx = src_gfx_c;

  // Decide the action from (code-object ISA, device ISA, device stepping).
  Action action;
  bool rewrite = false;
  if (src_gfx != g_device_gfx) {
    action = TRANSPILE;                       // cross-family
  } else if (g_device_is_a0) {
    action = REWRITE;                         // same ISA, A0 board: B0->A0
    rewrite = true;
  } else {
    action = PASS;                            // same ISA, B0/native
  }
  LOGF("reader_create: co_isa=%s device=%s -> %s (size=%zu, marked=%d)",
       src_gfx.c_str(), g_device_gfx.c_str(),
       action == PASS ? "PASS" : (action == REWRITE ? "REWRITE" : "TRANSPILE"),
       size, (int)marked);
  if (action == PASS) return g_real_reader_create(code_object, size, reader);

  // Build comgr input carrying the SOURCE mach in e_flags (a marker-patched
  // object may have byte 48 set to the device mach).
  std::vector<uint8_t> src_bytes(b, b + size);
  src_bytes[48] = src_mach;

  std::vector<uint8_t> out;
  if (!RunComgr(src_bytes, src_gfx, g_device_gfx, rewrite, &out)) {
    std::fprintf(stderr,
                 "hotswap_tool: %s %s->%s failed; forwarding original\n",
                 rewrite ? "rewrite" : "transpile", src_gfx.c_str(),
                 g_device_gfx.c_str());
    return g_real_reader_create(code_object, size, reader);
  }

  const uint8_t* persist;
  size_t persist_size;
  {
    std::lock_guard<std::mutex> lk(g_retain_mu);
    g_retained.emplace_back(std::move(out));
    persist = g_retained.back().data();
    persist_size = g_retained.back().size();
  }
  return g_real_reader_create(persist, persist_size, reader);
}

}  // namespace

extern "C" bool OnLoad(void* table, uint64_t, uint64_t, const char* const*) {
  auto* api = static_cast<HsaApiTable*>(table);
  if (!api || !api->core_) {
    std::fprintf(stderr, "hotswap_tool: no core API table\n");
    return false;
  }

  if (const char* c = std::getenv("HSA_HOTSWAP_CACHE_DIR")) g_cache_dir = c;
  if (const char* r = std::getenv("HSA_HOTSWAP_A0_REVISION"))
    g_a0_revision = static_cast<uint32_t>(std::strtoul(r, nullptr, 0));
  if (const char* f = std::getenv("HSA_HOTSWAP_FORCE_STEPPING_REWRITE"))
    g_force_stepping_rewrite = f[0] && f[0] != '0';
  if (const char* t = std::getenv("HSA_HOTSWAP_TARGET_OVERRIDE"))
    g_target_override = ExtractGfx(t);
  if (const char* v = std::getenv("HSA_HOTSWAP_TOOL_VERBOSE"))
    g_verbose = v[0] && v[0] != '0';

  g_iterate_agents = api->core_->hsa_iterate_agents_fn;
  g_agent_get_info = api->core_->hsa_agent_get_info_fn;
  g_isa_get_info_alt = api->core_->hsa_isa_get_info_alt_fn;
  g_real_reader_create = api->core_->hsa_code_object_reader_create_from_memory_fn;
  api->core_->hsa_code_object_reader_create_from_memory_fn = &ReaderCreateWrapper;

  std::fprintf(stderr,
               "hotswap_tool: loaded (device facts resolved on first load)\n");
  return true;
}

extern "C" void OnUnload() {}
