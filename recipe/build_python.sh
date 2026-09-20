#!/bin/bash
set -ex

# Must match the kernels: same NATTEN_WITH_*_FNA flags, resolved the same way
source "${RECIPE_DIR}/cuda_archs.sh"

# Install the pure-python part. Hide any GPU and leave NATTEN_CUDA_ARCH unset so
# setup.py does not build libnatten itself; the pybind11 module is built below
# against the kernels already compiled into the libnatten package.
CUDA_VISIBLE_DEVICES="" NATTEN_CUDA_ARCH="" \
    ${PYTHON} -m pip install . -vv --no-deps --no-build-isolation

cmake -S "${RECIPE_DIR}/extension" -B build-extension ${CMAKE_ARGS} \
    -DNATTEN_CSRC="${SRC_DIR}/csrc" \
    -DNATTEN_KERNELS_LIB="${PREFIX}/lib/libnatten_kernels.so" \
    -DNATTEN_WITH_HOPPER_FNA="${NATTEN_WITH_HOPPER_FNA}" \
    -DNATTEN_WITH_BLACKWELL_FNA="${NATTEN_WITH_BLACKWELL_FNA}" \
    -DNATTEN_EXT_SUFFIX="$(${PYTHON} -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')" \
    -DTORCH_INCLUDE_DIRS="$(${PYTHON} -c 'from torch.utils.cpp_extension import include_paths; print(";".join(include_paths()))')" \
    -DTORCH_LIBRARY_DIRS="${PREFIX}/lib;${SP_DIR}/torch/lib" \
    -DPYTHON_INCLUDE_DIR="$(${PYTHON} -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
cmake --build build-extension -j"${CPU_COUNT}"
cp build-extension/libnatten*.so "${SP_DIR}/natten/"
