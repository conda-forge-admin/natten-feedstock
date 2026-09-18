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
# The CUDA 13 list mirrors pytorch-cpu-feedstock's TORCH_CUDA_ARCH_LIST for
# CUDA 13 on linux-64 ("7.5;8.0;8.6;9.0;10.0;11.0;12.0+PTX"), so a natten build
# covers the same GPUs as the pytorch it links against. sm_110 exists only from
# CUDA 13 on, which is a second reason this cannot be shared with 12.9.
case "${cuda_compiler_version}" in
    13.*)
        export NATTEN_CUDA_ARCHS="75-real;80-real;86-real;90-real;100-real;110-real;120-real;120-virtual"
        export NATTEN_WITH_HOPPER_FNA="1"
        export NATTEN_WITH_BLACKWELL_FNA="1"
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
