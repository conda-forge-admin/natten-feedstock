#!/bin/bash
set -ex

# Resolves NATTEN_CUDA_ARCHS and the NATTEN_WITH_*_FNA flags from cuda_compiler_version
source "${RECIPE_DIR}/cuda_archs.sh"

rm -rf third_party/cutlass/include

# Generate the kernel instantiations with setup.py's "default" split policy.
# The hopper/blackwell families are only built for sm_90/sm_100, like upstream.
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
# nvcc's peak memory differs by almost an order of magnitude between the
# families, so each one gets its own job count instead of a single global -j.
# Measured peak RSS of one translation unit with CUDA 13.4 and CUTLASS 4.7.1:
# a portable kernel needs ~1.1 GB, a blackwell one ~8.5 GB. The CI agents have
# 4 cores and 16 GB, so blackwell has to be built one at a time while the
# portable kernels can use every core. A single hardcoded -j either wastes
# most of the build or runs the agent out of memory part way through.
mem_gb=$(awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo)
max_workers="${NATTEN_N_WORKERS:-${CPU_COUNT:-1}}"

# $1 = GB one nvcc process of this family needs at its peak
build_family() {
    local target="$1" per_job_gb="$2" jobs
    jobs=$(( mem_gb / per_job_gb ))
    # Spelled out rather than as "(( ... )) && jobs=N": a false (( )) returns 1,
    # which under "set -e" is a trap waiting for the next person to reorder this.
    if (( jobs < 1 )); then
        jobs=1
    fi
    if (( jobs > max_workers )); then
        jobs="${max_workers}"
    fi
    echo "building ${target} with -j${jobs} (${mem_gb} GB / ${per_job_gb} GB per job)"
    cmake --build build-kernels --target "${target}" -j"${jobs}"
}

build_family natten_kernels_generic 2
if [[ "${NATTEN_WITH_HOPPER_FNA}" == "1" ]]; then
    build_family natten_kernels_hopper 6
fi
if [[ "${NATTEN_WITH_BLACKWELL_FNA}" == "1" ]]; then
    build_family natten_kernels_blackwell 10
fi

# Everything above is compiled; all that is left is linking the shared library.
cmake --build build-kernels -j1
cmake --install build-kernels
