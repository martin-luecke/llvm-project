// COM: End-to-end DWARF preservation: compile a tiny HIP kernel with -g,
// COM: raise it through raise_cli, and verify the IR carries DI metadata
// COM: mirroring the input's CU and line table. Also pins the opt-in
// COM: contract: without --preserve-debug-info the IR has zero DI nodes
// COM: (a default-on regression would collide cache keys, see
// COM: translation-cache.cpp).

// REQUIRES: comgr-hotswap-transpile

// COM: -O2 elides the SRC_PRIVATE_BASE prologue the raiser refuses;
// COM: -nogpulib -nogpuinc drops the HIP install dependency.
// RUN: %clang --offload-arch=gfx942 --offload-device-only \
// RUN:     --no-gpu-bundle-output -nogpulib -nogpuinc -g -O2 \
// RUN:     %s -o %t.dbg.hsaco

// COM: Sanity: pin that the input really has DWARF, else the assertions
// COM: below pass vacuously.
// RUN: %llvm-readelf --section-headers %t.dbg.hsaco \
// RUN:   | %FileCheck --check-prefix=HAS-DWARF %s
// HAS-DWARF: .debug_info
// HAS-DWARF: .debug_line

// COM: With the flag, the IR carries DI metadata. Pin only structural
// COM: anchors (CU, subprogram-by-name, a DILocation, a `!dbg`), not line
// COM: numbers, which vary with optimisation level.
// RUN: %raise_cli %t.dbg.hsaco --emit-ir=_Z3addPfS_S_ --preserve-debug-info \
// RUN:   | %FileCheck --check-prefix=WITH-DI %s
// WITH-DI:      !dbg !
// WITH-DI:      !llvm.dbg.cu
// WITH-DI:      !DICompileUnit(language:
// WITH-DI:      !DISubprogram(name: "add"
// WITH-DI-SAME: linkageName: "_Z3addPfS_S_"
// WITH-DI:      !DILocation(line: {{[0-9]+}}

// COM: Opt-in guard: without the flag the IR has zero DI metadata even
// COM: though the input has DWARF.
// RUN: %raise_cli %t.dbg.hsaco --emit-ir=_Z3addPfS_S_ \
// RUN:   | %FileCheck --check-prefix=NO-DI %s
// NO-DI-NOT: !DICompileUnit
// NO-DI-NOT: !DISubprogram
// NO-DI-NOT: !DILocation
// NO-DI-NOT: !dbg

__attribute__((global))
void add(float *A, float *B, float *C) {
  *C = *A + *B;
}
