#!/bin/bash
set -ex

# Resolves NATTEN_CUDA_ARCHS and the NATTEN_WITH_*_FNA flags from cuda_compiler_version
source "${RECIPE_DIR}/cuda_archs.sh"

rm -rf third_party/cutlass/include

# Generate the kernel instantiations with setup.py's "default" split policy.
# The hopper/blackwell families are only built for sm_90/sm_100, like upstream.
#
# Resplitting is not a lever worth pulling here: measured on blackwell_fna,
# 56 splits costs 39% more CPU than the default 28 and 14 splits raises peak
# memory, so the default stays.
AUTOGEN_SPECS="reference_fna:2 fna:64 fmha:6"
if [[ "${NATTEN_WITH_HOPPER_FNA}" == "1" ]]; then
    AUTOGEN_SPECS+=" hopper_fna:8 hopper_fna_bwd:4 hopper_fmha:5 hopper_fmha_bwd:5"
fi
if [[ "${NATTEN_WITH_BLACKWELL_FNA}" == "1" ]]; then
    AUTOGEN_SPECS+=" blackwell_fna:28 blackwell_fna_bwd:14 blackwell_fmha:4 blackwell_fmha_bwd:4"
fi
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
    -DNATTEN_HOPPER_ARCHS="${NATTEN_HOPPER_ARCHS}" \
    -DNATTEN_BLACKWELL_ARCHS="${NATTEN_BLACKWELL_ARCHS}" \
    -DNATTEN_WITH_HOPPER_FNA="${NATTEN_WITH_HOPPER_FNA}" \
    -DNATTEN_WITH_BLACKWELL_FNA="${NATTEN_WITH_BLACKWELL_FNA}" \
    -DCUTLASS_INCLUDE_DIR="${PREFIX}/include" \
    -DTORCH_INCLUDE_DIRS="${PREFIX}/include;${PREFIX}/include/torch/csrc/api/include" \
    -DTORCH_LIBRARY_DIRS="${PREFIX}/lib"
# One compile job at a time. The CUTLASS translation units are huge -- peak
# RSS for a single one is ~3 GB for the portable kernels and ~10-12 GB for
# hopper and blackwell -- and the CI agents only have 16 GB, so anything above
# -j1 risks the agent being OOM-killed part way through a multi-hour build.
cmake --build build-kernels -j1
cmake --install build-kernels
