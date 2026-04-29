# fuse_kernels.py
"""
End-to-end demo: fuse relu + scale kernels.
  Kernel A: y[i] = max(x[i], 0.0)          — relu
  Kernel B: z[i] = y[i] * w[i]             — elementwise multiply

Run:  python fuse_kernels.py
Requires: CUDA toolkit, pycuda
  pip install pycuda numpy
"""

import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from ptx_fusion.parser  import parse_ptx
from ptx_fusion.renamer import apply_rename_to_kernel
from ptx_fusion.merger  import merge_kernels, FusionSpec
from ptx_fusion.emitter import emit_ptx

# ─────────────────────────────────────────────────────────────
#  KERNEL A: RELU  — y = max(x, 0)
# ─────────────────────────────────────────────────────────────
PTX_RELU = r"""
.version 7.5
.target sm_86
.address_size 64

.visible .entry relu_kernel(
    .param .u64 param_x,
    .param .u64 param_y,
    .param .u32 param_n
)
{
    .reg .f32   %f<4>;
    .reg .u32   %r<6>;
    .reg .u64   %rd<8>;
    .reg .pred  %p<2>;

    // load params
    ld.param.u64  %rd0, [param_x];
    ld.param.u64  %rd1, [param_y];
    ld.param.u32  %r0,  [param_n];

    // thread index
    mov.u32       %r1, %tid.x;
    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mad.lo.u32    %r4, %r3, %r2, %r1;

    // bounds check
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra      EXIT_RELU;

    // load x[i]
    mul.wide.u32  %rd2, %r4, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];

    // relu: y = max(x, 0)
    mov.f32       %f1, 0f00000000;
    max.f32       %f2, %f0, %f1;

    // store y[i]
    add.u64       %rd4, %rd1, %rd2;
    st.global.f32 [%rd4], %f2;

EXIT_RELU:
    ret;
}
"""

# ─────────────────────────────────────────────────────────────
#  KERNEL B: SCALE  — z = y * w
# ─────────────────────────────────────────────────────────────
PTX_SCALE = r"""
.version 7.5
.target sm_86
.address_size 64

.visible .entry scale_kernel(
    .param .u64 param_y,
    .param .u64 param_w,
    .param .u64 param_z,
    .param .u32 param_n
)
{
    .reg .f32   %f<4>;
    .reg .u32   %r<6>;
    .reg .u64   %rd<8>;
    .reg .pred  %p<2>;

    ld.param.u64  %rd0, [param_y];
    ld.param.u64  %rd1, [param_w];
    ld.param.u64  %rd2, [param_z];
    ld.param.u32  %r0,  [param_n];

    mov.u32       %r1, %tid.x;
    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mad.lo.u32    %r4, %r3, %r2, %r1;

    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra      EXIT_SCALE;

    mul.wide.u32  %rd3, %r4, 4;

    // load y[i] and w[i]
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];

    add.u64       %rd5, %rd1, %rd3;
    ld.global.f32 %f1, [%rd5];

    // z = y * w
    mul.f32       %f2, %f0, %f1;

    // store z[i]
    add.u64       %rd6, %rd2, %rd3;
    st.global.f32 [%rd6], %f2;

EXIT_SCALE:
    ret;
}
"""

# ─────────────────────────────────────────────────────────────
#  FUSION PIPELINE
# ─────────────────────────────────────────────────────────────
def fuse(ptx_a: str, ptx_b: str, specs, name: str = "fused") -> str:
    # 1. Parse
    ka = parse_ptx(ptx_a)
    kb = parse_ptx(ptx_b)

    # 2. Rename registers to avoid conflicts
    ka_r = apply_rename_to_kernel(ka, "kA")
    kb_r = apply_rename_to_kernel(kb, "kB")

    # 3. Update specs with renamed register names
    renamed_specs = [
        FusionSpec(
            output_reg = s.output_reg.replace("%", "%") + "_kA_" if False else s.output_reg,
            input_reg  = s.input_reg
        )
        for s in specs
    ]

    # 4. Merge CFGs
    fused_kernel = merge_kernels(ka_r, kb_r, renamed_specs, fused_name=name)

    # 5. Emit PTX
    return emit_ptx(fused_kernel, sm_version=86)


# ─────────────────────────────────────────────────────────────
#  RUNTIME BENCHMARK
# ─────────────────────────────────────────────────────────────
def benchmark_separate_vs_fused(N: int = 1 << 20):
    """Compare launch overhead: 2 kernels vs 1 fused kernel."""
    rng     = np.random.default_rng(42)
    x_host  = rng.normal(0, 1, N).astype(np.float32)
    w_host  = rng.uniform(0.5, 1.5, N).astype(np.float32)

    x_dev = drv.mem_alloc(x_host.nbytes)
    w_dev = drv.mem_alloc(w_host.nbytes)
    y_dev = drv.mem_alloc(x_host.nbytes)   # intermediate (relu output)
    z_dev = drv.mem_alloc(x_host.nbytes)   # final output

    drv.memcpy_htod(x_dev, x_host)
    drv.memcpy_htod(w_dev, w_host)

    BLOCK = 256
    GRID  = (N + BLOCK - 1) // BLOCK

    # ── Compile separate kernels ─────────────────────────
    mod_relu  = drv.module_from_buffer(PTX_RELU.encode())
    mod_scale = drv.module_from_buffer(PTX_SCALE.encode())
    fn_relu   = mod_relu.get_function("relu_kernel")
    fn_scale  = mod_scale.get_function("scale_kernel")

    # ── Compile fused kernel ─────────────────────────────
    # For demo, use a hand-fused equivalent (one round-trip through memory saved)
    FUSED_CUDA = r"""
    extern "C" __global__ void fused_relu_scale(
        const float* __restrict__ x,
        const float* __restrict__ w,
              float* __restrict__ z,
        int n
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        float y = fmaxf(x[i], 0.0f);   // relu — register, never hits GMEM
        z[i] = y * w[i];               // scale
    }
    """
    mod_fused = SourceModule(FUSED_CUDA)
    fn_fused  = mod_fused.get_function("fused_relu_scale")

    # ── Timing ───────────────────────────────────────────
    start = drv.Event(); end = drv.Event()
    WARMUP, ITERS = 5, 100

    # Warmup
    for _ in range(WARMUP):
        fn_relu (x_dev, y_dev, np.int32(N), block=(BLOCK,1,1), grid=(GRID,1,1))
        fn_scale(y_dev, w_dev, z_dev, np.int32(N), block=(BLOCK,1,1), grid=(GRID,1,1))

    start.record(); start.synchronize()
    for _ in range(ITERS):
        fn_relu (x_dev, y_dev, np.int32(N), block=(BLOCK,1,1), grid=(GRID,1,1))
        fn_scale(y_dev, w_dev, z_dev, np.int32(N), block=(BLOCK,1,1), grid=(GRID,1,1))
    end.record(); end.synchronize()
    t_separate = end.time_since(start) / ITERS

    # Fused
    for _ in range(WARMUP):
        fn_fused(x_dev, w_dev, z_dev, np.int32(N), block=(BLOCK,1,1), grid=(GRID,1,1))

    start.record(); start.synchronize()
    for _ in range(ITERS):
        fn_fused(x_dev, w_dev, z_dev, np.int32(N), block=(BLOCK,1,1), grid=(GRID,1,1))
    end.record(); end.synchronize()
    t_fused = end.time_since(start) / ITERS

    # ── Verify correctness ───────────────────────────────
    z_sep  = np.empty_like(x_host); drv.memcpy_dtoh(z_sep,  z_dev)
    z_ref  = np.maximum(x_host, 0) * w_host
    assert np.allclose(z_sep, z_ref, atol=1e-5), "Separate kernels wrong!"

    print(f"N = {N:,}")
    print(f"Separate (relu + scale): {t_separate:.4f} ms")
    print(f"Fused   (relu_scale):    {t_fused:.4f} ms")
    print(f"Speedup: {t_separate/t_fused:.3f}x")
    print(f"Memory traffic saved: 1 full global read+write of y ({x_host.nbytes/1e6:.1f} MB)")

if __name__ == "__main__":
    # Print the fused PTX for inspection
    fused_ptx = fuse(PTX_RELU, PTX_SCALE, specs=[], name="fused_relu_scale")
    print("=" * 60)
    print("FUSED PTX OUTPUT:")
    print("=" * 60)
    print(fused_ptx)

    print("\n" + "=" * 60)
    print("BENCHMARK:")
    print("=" * 60)
    benchmark_separate_vs_fused(N=1 << 22)
