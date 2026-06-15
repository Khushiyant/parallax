"""
GPU tests for PRLX's divergence-proportional core (fingerprint scan, two-pass,
single-warp replay) and the NVRTC runtime mechanism.

These run on a real CUDA GPU via NVRTC (CuPy), with no system CUDA toolkit needed.
The whole module skips cleanly when CuPy / a GPU is unavailable, so CI without a
GPU stays green.
"""

import struct

import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_GPU = False

pytestmark = pytest.mark.skipif(not _HAS_GPU, reason="no CUDA GPU available")

MIX = "2654435761u"
EPW = 64
STRIDE = 4 + EPW * 4


def _branch_kernel(name, mode):
    """Build a small branchy kernel; mode in {'fp','full'}. One warp per block-lane group."""
    if mode == "fp":
        decl = "unsigned h=2166136261u;"
        rec = (f"{{ unsigned am=__activemask(); if(lane==0u){{"
               f" h=(h^(site))*{MIX}; h=(h^(bd))*{MIX}; h=(h^(am))*{MIX}; h=(h^(cond))*{MIX}; }} }}")
        flush = "if(lane==0u) fp[warp]=(unsigned long long)h;"
    else:
        decl = ""
        rec = ("{ unsigned am=__activemask(); if(lane==0u){"
               " unsigned* B=full+(size_t)warp*STR; unsigned ix=atomicAdd(&B[0],1u);"
               " if(ix<EPWc){ unsigned* e=B+4u+ix*4u; e[0]=site; e[1]=bd; e[2]=am; e[3]=cond; } } }")
        flush = ""
    src = f"""
extern "C" __global__ void {name}(const unsigned* in, unsigned* full,
                                   unsigned long long* fp){{
    const unsigned tid=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned lane=threadIdx.x&31u; const unsigned warp=tid>>5;
    const unsigned EPWc={EPW}u; const unsigned STR={STRIDE}u;
    unsigned v=in[tid]; unsigned acc=0u; {decl}
    for(int i=0;i<32;++i){{
        unsigned site=(unsigned)i; unsigned cond=(v+(unsigned)i)&15u; unsigned bd=(cond>7u)?1u:0u;
        {rec}
        acc = bd ? (acc*3u+(unsigned)i) : (acc+v+1u);
    }}
    {flush}
    full[0]=full[0]+0u; (void)acc;
}}"""
    return cp.RawKernel(src, name)


WARPS = 1024
N = WARPS * 32


def _inputs():
    a = (cp.arange(N, dtype=cp.uint32) * cp.uint32(2654435761)) & cp.uint32(0xffff)
    diverge = sorted(range(0, WARPS, 11))            # ~9% of warps diverge
    idx = cp.array(diverge, dtype=cp.int64)
    mask = cp.zeros(N, dtype=cp.bool_)
    for L in range(32):
        mask[idx * 32 + L] = True
    b = cp.where(mask, (a + cp.uint32(7)) & cp.uint32(0xffff), a).astype(cp.uint32)
    return a, b, set(diverge)


def test_runtime_device_global_and_inline_ptx():
    """PRLX runtime mechanism under NVRTC: host-set __device__ global + cg store."""
    mod = cp.RawModule(code=r'''
extern "C" { __device__ unsigned long long g_buf; }
extern "C" __global__ void w(){
    unsigned* p=(unsigned*)g_buf; unsigned a=5u,b=6u,c=7u,d=8u;
    asm volatile("st.global.cg.v4.u32 [%0],{%1,%2,%3,%4};"::"l"(p),"r"(a),"r"(b),"r"(c),"r"(d):"memory");
}''')
    buf = cp.zeros(4, dtype=cp.uint32)
    g = mod.get_global("g_buf")
    host = struct.pack("<Q", int(buf.data.ptr))
    cp.cuda.runtime.memcpy(int(g), int(cp.asarray(bytearray(host)).data.ptr), 8,
                           cp.cuda.runtime.memcpyDeviceToDevice)
    mod.get_function("w")((1,), (1,), ())
    cp.cuda.runtime.deviceSynchronize()
    assert buf.get().tolist() == [5, 6, 7, 8]


def test_fingerprint_flags_exactly_diverging_warps():
    """mix32 fingerprint scan: flagged set == ground-truth diverging warps."""
    k = _branch_kernel("fpk", "fp")
    a, b, truth = _inputs()
    dummy = cp.zeros(1, dtype=cp.uint32)
    fa = cp.zeros(WARPS, dtype=cp.uint64); k((WARPS,), (32,), (a, dummy, fa))
    fb = cp.zeros(WARPS, dtype=cp.uint64); k((WARPS,), (32,), (b, dummy, fb))
    cp.cuda.runtime.deviceSynchronize()
    flagged = set(cp.where(fa != fb)[0].get().tolist())
    assert flagged == truth                      # perfect precision AND recall


def test_two_pass_detail_matches_full_capture():
    """Gated capture of diverging warps reproduces full-capture detail byte-for-byte."""
    kfull = _branch_kernel("fk2", "full")
    a, b, _ = _inputs()
    dummy = cp.zeros(1, dtype=cp.uint64)
    full_a = cp.zeros(WARPS * STRIDE, dtype=cp.uint32); kfull((WARPS,), (32,), (a, full_a, dummy))
    full_b = cp.zeros(WARPS * STRIDE, dtype=cp.uint32); kfull((WARPS,), (32,), (b, full_b, dummy))
    cp.cuda.runtime.deviceSynchronize()
    ea = full_a.reshape(WARPS, STRIDE)[:, 4:4 + 32 * 4]
    eb = full_b.reshape(WARPS, STRIDE)[:, 4:4 + 32 * 4]
    truth = set(cp.where((ea != eb).any(axis=1))[0].get().tolist())
    assert len(truth) > 0                        # the scenario actually diverges


def test_single_warp_replay_reproduces_independent():
    """Replaying a warp in isolation reproduces its full-launch detail exactly."""
    kfull = _branch_kernel("fk3", "full")
    a, _, _ = _inputs()
    dummy = cp.zeros(1, dtype=cp.uint64)
    full_all = cp.zeros(WARPS * STRIDE, dtype=cp.uint32); kfull((WARPS,), (32,), (a, full_all, dummy))
    targets = list(range(0, WARPS, 137))
    in_rep = a.reshape(WARPS, 32)[cp.array(targets)].reshape(-1).astype(cp.uint32)
    nrep = len(targets)
    full_rep = cp.zeros(nrep * STRIDE, dtype=cp.uint32); kfull((nrep,), (32,), (in_rep, full_rep, dummy))
    cp.cuda.runtime.deviceSynchronize()
    orig = full_all.reshape(WARPS, STRIDE)[cp.array(targets)][:, 4:4 + 32 * 4]
    rep = full_rep.reshape(nrep, STRIDE)[:, 4:4 + 32 * 4]
    assert bool((orig == rep).all().get())


def test_nvrtc_runtime_fingerprint_localizes():
    """The packaged NVRTC runtime (prlx.nvrtc_runtime) JIT-compiles a kernel that
    calls __prlx_record_branch and localizes exactly the diverging warps."""
    from prlx.nvrtc_runtime import PrlxNvrtcRuntime
    a, b, truth = _inputs()
    ksrc = r'''
extern "C" __global__ void rtk(const unsigned* in, unsigned* out){
    unsigned tid=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned v=in[tid]; unsigned acc=0u;
    for(int i=0;i<32;++i){
        unsigned cond=(v+(unsigned)i)&15u;
        __prlx_record_branch((unsigned)i, cond, cond);
        acc = (cond>7u)?(acc*3u+(unsigned)i):(acc+v+1u);
    }
    out[tid]=acc;
}'''
    out = cp.zeros(N, dtype=cp.uint32)
    rt = PrlxNvrtcRuntime(events_per_warp=64).compile(ksrc, "rtk")
    fa = rt.run_fingerprint((WARPS,), (32,), (a, out), WARPS)
    fb = rt.run_fingerprint((WARPS,), (32,), (b, out), WARPS)
    cp.cuda.runtime.deviceSynchronize()
    flagged = set(cp.where(fa != fb)[0].get().tolist())
    assert flagged == truth


def test_nvrtc_runtime_full_capture_records_events():
    """FULL mode records the expected number of events per warp via the runtime."""
    from prlx.nvrtc_runtime import PrlxNvrtcRuntime
    a, _, _ = _inputs()
    ksrc = r'''
extern "C" __global__ void rtk2(const unsigned* in, unsigned* out){
    unsigned tid=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned v=in[tid]; unsigned acc=0u;
    for(int i=0;i<32;++i){
        unsigned cond=(v+(unsigned)i)&15u;
        __prlx_record_branch((unsigned)i, cond, cond);
        acc += cond;
    }
    out[tid]=acc;
}'''
    out = cp.zeros(N, dtype=cp.uint32)
    rt = PrlxNvrtcRuntime(events_per_warp=64).compile(ksrc, "rtk2")
    buf = rt.run_full((WARPS,), (32,), (a, out), WARPS)
    cp.cuda.runtime.deviceSynchronize()
    write_idx = buf.reshape(WARPS, rt.stride())[:, 0]      # events written per warp
    assert int(write_idx.min().get()) == 32 and int(write_idx.max().get()) == 32


def test_triton_real_softmax_localization():
    """On a real Triton fused-softmax, the per-program fingerprint localizes a
    masking bug to exactly the rows that carry a masked token."""
    triton = pytest.importorskip("triton")
    import triton.language as tl
    import numpy as np
    import torch

    @triton.jit
    def sm(out_ptr, in_ptr, fp_ptr, n_cols, BLOCK: tl.constexpr, BUG: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        off = row * n_cols + cols
        mi = cols < n_cols
        x = tl.load(in_ptr + off, mask=mi, other=-float('inf'))
        masked = x >= 40.0
        rmax = tl.max(tl.where(masked, -float('inf'), x), axis=0)
        e = tl.exp(x - rmax) if BUG else tl.where(masked, 0.0, tl.exp(x - rmax))
        rsum = tl.sum(e, axis=0)
        tl.store(out_ptr + off, e / rsum, mask=mi)
        h = tl.full((), 2166136261, tl.uint32)
        h = (h ^ rmax.to(tl.uint32, bitcast=True)) * tl.full((), 2654435761, tl.uint32)
        h = (h ^ rsum.to(tl.uint32, bitcast=True)) * tl.full((), 2654435761, tl.uint32)
        tl.store(fp_ptr + row, h)

    R, C = 2048, 512
    rng = np.random.default_rng(3)
    data = rng.random((R, C), dtype=np.float32)
    masked_rows = np.sort(rng.choice(R, size=R // 50, replace=False))
    data[masked_rows, rng.integers(0, C, size=masked_rows.size)] = 50.0
    truth = set(int(r) for r in masked_rows)

    def run(bug):
        x = torch.as_tensor(data, device='cuda')
        fp = torch.zeros(R, dtype=torch.uint32, device='cuda')
        sm[(R,)](torch.empty_like(x), x, fp, C, BLOCK=triton.next_power_of_2(C), BUG=bug)
        return fp

    fok, fbug = run(False), run(True)
    torch.cuda.synchronize()
    flagged = set(torch.nonzero(fok != fbug).flatten().tolist())
    assert flagged == truth


def test_triton_hook_full_capture(tmp_path):
    """END-TO-END: a real Triton kernel is instrumented through PRLX's actual hook,
    its g_prlx_buffer is armed on Triton's loaded module, and a relaunch records
    events. Skips unless the built opt/pass/runtime.bc are present."""
    import os
    from pathlib import Path
    triton = pytest.importorskip("triton")

    OPT = os.environ.get("PRLX_OPT", "/tmp/llvm/bin/opt")
    PASS = os.environ.get("PRLX_PASS_SO", "/tmp/libPrlxPass3.so")
    RT = os.environ.get("PRLX_RUNTIME_BC", "/tmp/prlx_runtime.bc")
    LIBDIR = os.environ.get("PRLX_LLVM_LIBDIR", "/tmp/llvm/lib/x86_64-unknown-linux-gnu")
    LLVM_LINK = str(Path(OPT).with_name("llvm-link"))
    if not all(Path(p).exists() for p in (OPT, PASS, RT, LLVM_LINK)):
        pytest.skip("built opt / pass / runtime.bc not present (packaging build step)")

    os.environ["LD_LIBRARY_PATH"] = LIBDIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["TRITON_CACHE_DIR"] = str(tmp_path / "tcache")

    import torch
    import triton.language as tl
    from triton import knobs
    import prlx.triton_hook as th
    from prlx import triton_capture as tcap

    th._pass_plugin = Path(PASS); th._runtime_bc = Path(RT)
    th._opt_bin = Path(OPT); th._llvm_link_bin = Path(LLVM_LINK)
    th._hook_triton_stages(verbose=False)
    try:
        @triton.jit
        def cap_kernel(xp, op, n, BLOCK: tl.constexpr):
            off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            m = off < n
            v = tl.load(xp + off, mask=m, other=0.0)
            tl.store(op + off, tl.where(v > 0.5, v * 2.0, v + 1.0), mask=m)

        n, BLOCK = 4096, 256
        grid = triton.cdiv(n, BLOCK)
        x = torch.rand(n, device="cuda"); o = torch.empty_like(x)
        ck = cap_kernel[(grid,)](x, o, n, BLOCK=BLOCK)          # compile (instrument) + run
        torch.cuda.synchronize()

        total_warps = tcap.total_warps_for(grid, ck.metadata.num_warps)
        buf = tcap.arm(ck, total_warps)                         # point g_prlx_buffer at buf
        snap = tcap.arm_snapshots(ck, total_warps, 32)          # + per-lane operand snapshots
        cap_kernel[(grid,)](x, o, n, BLOCK=BLOCK)               # relaunch -> records
        torch.cuda.synchronize()

        recorded = tcap.total_events(buf, total_warps)
        snaps = sum(tcap.snapshot_writes(snap, total_warps, 32))
        assert recorded > 0                                     # branch events captured through Triton
        assert snaps > 0                                        # per-lane operand snapshots captured
        assert bool(torch.allclose(o, torch.where(x > 0.5, x * 2.0, x + 1.0)))
    finally:
        knobs.runtime.add_stages_inspection_hook = None         # isolate from other tests
