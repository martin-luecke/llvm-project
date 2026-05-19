// COM: Null-safety contract: `--preserve-debug-info` on an input without
// COM: `.debug_info` must not crash and must not inject synthetic DI
// COM: metadata -- the flag is meant to be safe to enable unconditionally
// COM: even when the caller doesn't know whether the input had -g.

// REQUIRES: comgr-hotswap-transpile

// COM: Same kernel as hotswap-debuginfo.hip but without -g. -O2 avoids
// COM: the SRC_PRIVATE_BASE prologue the raiser refuses.
// RUN: %clang --offload-arch=gfx942 --offload-device-only \
// RUN:     --no-gpu-bundle-output -nogpulib -nogpuinc -O2 \
// RUN:     %s -o %t.nodbg.hsaco

// COM: Sanity: pin that the input has no DWARF. NOT on section names
// COM: rather than globally, since some configs emit unrelated
// COM: debug-strings sections.
// RUN: %llvm-readelf --section-headers %t.nodbg.hsaco \
// RUN:   | %FileCheck --check-prefix=NO-DWARF %s
// NO-DWARF-NOT: .debug_info
// NO-DWARF-NOT: .debug_line

// COM: Raise with the flag: must succeed and emit no DI nodes.
// RUN: %raise_cli %t.nodbg.hsaco --emit-ir=_Z3addPfS_S_ --preserve-debug-info \
// RUN:   | %FileCheck --check-prefix=CLEAN %s
// CLEAN:     define
// CLEAN-NOT: !DICompileUnit
// CLEAN-NOT: !DISubprogram
// CLEAN-NOT: !DILocation
// CLEAN-NOT: !dbg

__attribute__((global))
void add(float *A, float *B, float *C) {
  *C = *A + *B;
}
