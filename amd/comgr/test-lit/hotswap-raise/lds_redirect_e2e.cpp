// E2e test: LDS->global redirect for lds_redirect_shape_kernel.
//
// Loads a pre-translated gfx1151 binary (produced by hotswap-transpile with
// HSA_HOTSWAP_LDS_TO_GLOBAL=1 FORCE=1) and launches it via the low-level HIP
// module API.  The kernel does a 32-lane ring-shift through redirected LDS:
//   out[i] = (i + 1) % 32
// Kernel: lds_redirect_shape_kernel
// Kernarg layout (16 bytes total after translation):
//   offset 0: uint64_t out_ptr    (original arg)
//   offset 8: uint64_t wg_lds_base (injected by redirect)
// LDS scratch: 4096 bytes for 1 workgroup * group_segment_fixed_size=4096.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(x)                                                        \
    do {                                                                \
        hipError_t _e = (x);                                            \
        if (_e != hipSuccess) {                                         \
            fprintf(stderr, "HIP error %s:%d: %s\n",                   \
                    __FILE__, __LINE__, hipGetErrorString(_e));          \
            exit(1);                                                    \
        }                                                               \
    } while (0)

int main(int argc, char **argv) {
    const char *hsaco = (argc > 1) ? argv[1] : "/tmp/shape_gfx1151.hsaco";

    // Read the translated binary.
    FILE *f = fopen(hsaco, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", hsaco); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *img = malloc((size_t)sz);
    if (!img || (long)fread(img, 1, (size_t)sz, f) != sz) {
        fprintf(stderr, "read error\n"); return 1;
    }
    fclose(f);

    // Load module from in-memory binary.
    hipModule_t mod;
    CHECK(hipModuleLoadData(&mod, img));

    hipFunction_t kern;
    CHECK(hipModuleGetFunction(&kern, mod, "lds_redirect_shape_kernel"));

    // Allocate device buffers.
    int32_t *d_out = nullptr;
    void    *d_lds = nullptr;
    CHECK(hipMalloc((void **)&d_out, 32 * sizeof(int32_t)));
    CHECK(hipMalloc(&d_lds, 4096));
    CHECK(hipMemset(d_out, 0, 32 * sizeof(int32_t)));
    CHECK(hipMemset(d_lds, 0, 4096));

    // Kernarg: {out_ptr (8), wg_lds_base (8)} = 16 bytes.
    struct { uint64_t out; uint64_t lds_base; } kargs;
    kargs.out      = (uint64_t)(uintptr_t)d_out;
    kargs.lds_base = (uint64_t)(uintptr_t)d_lds;

    // Launch: 1 workgroup, 32 threads (wave32).
    // HIP_LAUNCH_PARAM_BUFFER_SIZE must follow with a pointer to size_t,
    // not the size value itself cast to void*.
    size_t kargs_size = sizeof(kargs);
    void *config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, (void *)&kargs,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    (void *)&kargs_size,
        HIP_LAUNCH_PARAM_END
    };
    CHECK(hipModuleLaunchKernel(kern,
        /*gridX=*/1,  /*gridY=*/1,  /*gridZ=*/1,
        /*blkX=*/32, /*blkY=*/1,  /*blkZ=*/1,
        /*sharedMem=*/0, /*stream=*/nullptr,
        /*kernelParams=*/nullptr, /*extra=*/config));
    CHECK(hipDeviceSynchronize());

    // Read back.
    int32_t h_out[32];
    CHECK(hipMemcpy(h_out, d_out, 32 * sizeof(int32_t), hipMemcpyDeviceToHost));

    // Verify: out[i] == (i + 1) % 32
    int fail = 0;
    for (int i = 0; i < 32; i++) {
        int expected = (i + 1) % 32;
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL: out[%d] = %d, expected %d\n",
                    i, h_out[i], expected);
            fail = 1;
        }
    }

    if (!fail)
        printf("PASS: LDS->global redirect e2e: 32-lane ring-shift verified\n");

    (void)hipFree(d_out);
    (void)hipFree(d_lds);
    (void)hipModuleUnload(mod);
    free(img);
    return fail;
}
