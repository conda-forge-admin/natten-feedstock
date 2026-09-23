#!/bin/bash
set -ex

# Same architectures as conda-forge's pytorch for each CUDA version. The hopper
# and blackwell kernel families get their own arch lists in kernels/CMakeLists.txt.
case "${cuda_compiler_version}" in
    12.*)
        NATTEN_CUDA_ARCHS="50-real;60-real;70-real;75-real;80-real;86-real;90-real;100-real;120-real;120-virtual"
        ;;
    *)
        NATTEN_CUDA_ARCHS="75-real;80-real;86-real;90-real;100-real;110-real;120-real;120-virtual"
        ;;
esac

rm -rf third_party/cutlass/include

# Generate the kernel instantiations with setup.py's "default" split policy.
#
# Resplitting is not a lever worth pulling here: measured on blackwell_fna,
# 56 splits costs 39% more CPU than the default 28 and 14 splits raises peak
# memory, so the default stays.
AUTOGEN_SPECS="reference_fna:2 fna:64 fmha:6"
AUTOGEN_SPECS+=" hopper_fna:8 hopper_fna_bwd:4 hopper_fmha:5 hopper_fmha_bwd:5"
AUTOGEN_SPECS+=" blackwell_fna:28 blackwell_fna_bwd:14 blackwell_fmha:4 blackwell_fmha_bwd:4"
for spec in ${AUTOGEN_SPECS}; do
    "${BUILD_PREFIX}/bin/python" "scripts/autogen_${spec%%:*}.py" \
        --num-splits "${spec##*:}" -o csrc
done

# The kernels and csrc/src wrappers only use the C++ API (ATen/c10), so include
# torch/all.h instead of torch/extension.h, which also pulls in the Python and
# pybind11 headers. natten.cpp keeps torch/extension.h; it is built per Python.
grep -rl 'torch/extension.h' csrc/src csrc/autogen | xargs sed -i 's#torch/extension.h#torch/all.h#'

cmake -S "${RECIPE_DIR}/kernels" -B build-kernels ${CMAKE_ARGS} \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DNATTEN_CSRC="${SRC_DIR}/csrc" \
    -DNATTEN_CUDA_ARCHS="${NATTEN_CUDA_ARCHS}" \
    -DCUTLASS_INCLUDE_DIR="${PREFIX}/include" \
    -DTORCH_INCLUDE_DIRS="${PREFIX}/include;${PREFIX}/include/torch/csrc/api/include" \
    -DTORCH_LIBRARY_DIRS="${PREFIX}/lib"
# A hopper or blackwell translation unit peaks at 10-12 GB, so six at once
# could in principle need ~72 GB of the large runner's 64 GB. In practice -j6
# has not been OOM-killed; drop to -j5 (~60 GB worst case) if it ever is.
cmake --build build-kernels -j6
cmake --install build-kernels
