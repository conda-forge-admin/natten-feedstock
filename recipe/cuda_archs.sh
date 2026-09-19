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
# Datacenter parts are CUDA 13 only; 12.9 keeps the consumer selection.
#
# Blackwell FNA is off, and the reason is CI time, not correctness. The
# 0003 patch does port the blackwell helpers to the CUTLASS 4.5+ API and all
# 53 of its translation units compile clean against conda-forge's 4.7.1 --
# it is turned off purely because the job does not fit. Measured on
# run 35439135977, linux_64 cuda 13.4, against the 360 minute GitHub cap:
#
#   portable    80 TUs   -j4   2h18m
#   hopper      25 TUs   -j1   1h22m
#   blackwell   53 TUs   -j1   3h07m  (cancelled at 38/53)
#                              ~7h05m total
#
# Hopper and blackwell cannot share the machine: one translation unit peaks
# at 10.4 GB and 11.7 GB respectively, so on a 16 GB agent they are strictly
# -j1 and together they are 4h29m of unavoidable serial work. Dropping
# blackwell brings the job to roughly 3h58m.
#
# To turn it back on, flip NATTEN_WITH_BLACKWELL_FNA to 1 below. It needs an
# agent that can either run the datacenter families at -j2 (so, more than
# 16 GB of RAM) or allow more than six hours; nothing in this recipe will
# buy back the missing hour.
#
# sm_100-real stays in the portable list even with blackwell off -- it is
# what datacenter Blackwell falls back to, and 120-virtual PTX cannot JIT
# down to sm_100. sm_110 is left out: it would add an eighth target to the
# portable kernels for no coverage we do not already have.
case "${cuda_compiler_version}" in
    13.*)
        export NATTEN_CUDA_ARCHS="75-real;80-real;86-real;90-real;100-real;120-real;120-virtual"
        export NATTEN_HOPPER_ARCHS="90a-real"
        export NATTEN_BLACKWELL_ARCHS="100a-real"
        export NATTEN_WITH_HOPPER_FNA="1"
        # See the CI budget note above before flipping this to 1
        export NATTEN_WITH_BLACKWELL_FNA="0"
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
