# hotswap-interposer

A single injected library that presents a spoofed gfx device to the ROCm stack,
captures every emitted code object at the HSA load layer, transpiles it to the
real device ISA via COMGR, and runs the transpiled result on the real hardware.
See `DESIGN.md` for the architecture and `INTEGRATION.md` for how it converges
with the in-runtime HotSwap integration.

## Build

The library consumes a prebuilt COMGR (with the hotswap transpile API) and the
HSA headers; both are supplied explicitly (no defaults, so the build is
portable):

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DHOTSWAP_HSA_INCLUDE_DIR=<rocr-runtime>/runtime/hsa-runtime/inc \
  -DHOTSWAP_COMGR_INCLUDE_DIR=<comgr-build>/include \
  -DHOTSWAP_COMGR_LIBRARY=<comgr-build>/libamd_comgr.so
ninja -C build
```

`HOTSWAP_HSA_INCLUDE_DIR` must be the rocr-runtime source `inc/` layout (its
`hsa_api_trace.h` includes `inc/hsa_ext_image.h`), not the flattened install tree.

## Run

The interposer is inert unless `HOTSWAP_INTERPOSER_SPOOF` is set. It must run
against a gfx-aware ROCr and HIP (see `DESIGN.md`):

```
LD_LIBRARY_PATH=<hip>/lib:<rocr-install>/lib \
HOTSWAP_INTERPOSER_SPOOF=gfx1250 \
LD_PRELOAD=build/libhotswap-interposer.so \
HSA_TOOLS_LIB=build/libhotswap-interposer.so \
  <program>
```

## Environment

- `HOTSWAP_INTERPOSER_SPOOF` -- gfx target to present upward (`gfx1250` or an
  encoded `gfx_target_version`). Unset => fully inert.
- `HSA_HOTSWAP_TARGET` -- real device ISA to transpile toward. Auto-published by
  the spoof half from the detected real device; user-overridable.
- `HSA_HOTSWAP_DISABLE` -- forward every code object untouched (no transpile).
- `HSA_HOTSWAP_VERBOSE` / `HOTSWAP_INTERPOSER_LOG` -- one-line diagnostics.
- `HOTSWAP_INTERPOSER_DUMP_DIR` -- dump captured/transpiled code objects (debug).
