#!/bin/bash
# Single source of truth for which CUDA architectures and which NATTEN kernel
# families this build targets. Sourced by BOTH build_kernels.sh and
# build_python.sh, so that the two halves cannot drift apart. Note that a
# mismatch does NOT fail the build: csrc/natten.cpp registers the hopper and
# blackwell entry points unconditionally (it never tests NATTEN_WITH_*_FNA),
# and the wrappers in csrc/src define them either way -- the real kernels when
# the family is on, stubs that raise "not compiled with ..." when it is off.
# So a disagreement links cleanly and only surfaces the first time a user
# actually reaches that code path on a datacenter GPU.
#
# Three separate lists, because the families do not target the same GPUs:
#
#   NATTEN_CUDA_ARCHS      the portable kernels (reference/fna/fmha and the
#                          ATen wrappers), built for every supported GPU
#   NATTEN_HOPPER_ARCHS    sm_90a only
#   NATTEN_BLACKWELL_ARCHS sm_100a only
#
# The "a" suffix is not cosmetic. CUTLASS's Hopper and Blackwell kernels use
# arch-conditional instructions, and nvcc compiles them to *nothing* for a
# plain sm_90/sm_100 without emitting any diagnostic: `nvcc -ptx` on
# hopper_fmha/source_0.cu gives 51662 lines with 584 wgmma ops for sm_90a,
# and 395 lines with zero for sm_90. Upstream's setup.py maps 90/100/103 to
# 90a/100a/103a for the same reason.
#
# Datacenter parts are CUDA 13 only. Enabling Hopper for both CUDA versions
# was measured on PR #4 (run 35351538869): linux_64 cuda 13.4 finished in
# 4h44m, while the identical commit on cuda 12.9 ran past the 360 minute CI
# timeout and was cancelled at 6h01m. That run predates both the per-family
# arch split below and the per-family -j in build_kernels.sh, which between
# them cut the datacenter cost a long way, so treat those numbers as a worst
# case rather than a current estimate. 12.9 stays consumer-only until someone
# actually re-measures it.
#
# sm_110 is still left out: it would add a seventh -real target to the
# portable kernels, and that budget is needed for sm_100 coverage instead.
case "${cuda_compiler_version}" in
    13.*)
        export NATTEN_CUDA_ARCHS="75-real;80-real;86-real;90-real;100-real;120-real;120-virtual"
        export NATTEN_HOPPER_ARCHS="90a-real"
        export NATTEN_BLACKWELL_ARCHS="100a-real"
        export NATTEN_WITH_HOPPER_FNA="1"
        export NATTEN_WITH_BLACKWELL_FNA="1"
        ;;
    *)
        export NATTEN_CUDA_ARCHS="75-real;80-real;86-real;120-real;120-virtual"
        export NATTEN_HOPPER_ARCHS=""
        export NATTEN_BLACKWELL_ARCHS=""
        export NATTEN_WITH_HOPPER_FNA="0"
        export NATTEN_WITH_BLACKWELL_FNA="0"
        ;;
esac

echo "cuda_compiler_version=${cuda_compiler_version}"
echo "NATTEN_CUDA_ARCHS=${NATTEN_CUDA_ARCHS}"
echo "NATTEN_HOPPER_ARCHS=${NATTEN_HOPPER_ARCHS}"
echo "NATTEN_BLACKWELL_ARCHS=${NATTEN_BLACKWELL_ARCHS}"
echo "NATTEN_WITH_HOPPER_FNA=${NATTEN_WITH_HOPPER_FNA}"
echo "NATTEN_WITH_BLACKWELL_FNA=${NATTEN_WITH_BLACKWELL_FNA}"
