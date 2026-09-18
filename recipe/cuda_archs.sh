#!/bin/bash
# Single source of truth for which CUDA architectures and which NATTEN kernel
# families this build targets. Sourced by BOTH build_kernels.sh and
# build_python.sh: the kernel sources and csrc/natten.cpp guard the
# hopper/blackwell families with "#if defined(NATTEN_WITH_*_FNA)", so if the
# two halves disagree the module loses symbols instead of failing loudly.
#
# The datacenter parts are CUDA 13 only, on purpose. Enabling Hopper for both
# CUDA versions was measured on PR #4 (run 35351538869): linux_64 cuda 13.4
# finished in 4h44m, while the identical commit on cuda 12.9 ran past the 360
# minute CI timeout and was cancelled at 6h01m. CUDA 12.9 therefore keeps the
# conservative consumer selection.
#
# Two things pytorch covers that we still do not, both deliberate:
#
#   * Datacenter Blackwell (10.0). NATTEN 0.21.7 vendors CUTLASS 4.3.5, but we
#     delete third_party/cutlass and build against conda-forge's, which is
#     4.5.3 or newer. cute::SM100_MMA_F8F6F4_SS became a class template in
#     CUTLASS 4.5, so natten/cuda/fmha_blackwell/collective/fmha_common.hpp
#     fails to compile:
#         error: argument list for class template
#                "cute::SM100_MMA_F8F6F4_SS" is missing
#     Only the blackwell FMHA sources use the changed API; every other family
#     builds fine against 4.5+. Upstream has no fix as of v0.21.7. Revisit when
#     NATTEN moves to CUTLASS 4.5+, then turn the family back on.
#
#   * sm_100/sm_110 as plain binary targets. They compile, but every extra
#     -real arch multiplies codegen across all sources, and the CUDA 13 job
#     already takes 4h44m of its 6 h budget with the list below. GitHub's
#     hosted-runner ceiling is 6 h, so there is no room without raising the
#     compile worker count.
case "${cuda_compiler_version}" in
    13.*)
        export NATTEN_CUDA_ARCHS="75-real;80-real;86-real;90-real;120-real;120-virtual"
        export NATTEN_WITH_HOPPER_FNA="1"
        export NATTEN_WITH_BLACKWELL_FNA="0"
        ;;
    *)
        export NATTEN_CUDA_ARCHS="75-real;80-real;86-real;120-real;120-virtual"
        export NATTEN_WITH_HOPPER_FNA="0"
        export NATTEN_WITH_BLACKWELL_FNA="0"
        ;;
esac

echo "cuda_compiler_version=${cuda_compiler_version}"
echo "NATTEN_CUDA_ARCHS=${NATTEN_CUDA_ARCHS}"
echo "NATTEN_WITH_HOPPER_FNA=${NATTEN_WITH_HOPPER_FNA}"
echo "NATTEN_WITH_BLACKWELL_FNA=${NATTEN_WITH_BLACKWELL_FNA}"
