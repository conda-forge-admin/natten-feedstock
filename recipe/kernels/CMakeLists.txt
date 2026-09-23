# Python-independent part of libnatten: every CUDA kernel plus the ATen-level
# wrappers in csrc/src, built as a shared library. Compile flags and
# definitions mirror upstream's csrc/CMakeLists.txt, minus the Python headers,
# so the result does not depend on the Python version. The pybind11 module in
# csrc/natten.cpp links against this and is the only per-Python artifact.
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(natten_kernels LANGUAGES CXX CUDA)

find_package(CUDAToolkit 12.0 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "-Xcompiler -Wall -ldl")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wconversion")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fno-strict-aliasing")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ftemplate-backtrace-limit=0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNDEBUG")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin -compress-all")

# conda-forge's libtorch is built with the new C++11 ABI
add_compile_definitions(
  _GLIBCXX_USE_CXX11_ABI=1
  WITH_CUDA
  NATTEN_WITH_CUTLASS
  CUTLASS_ENABLE_TENSOR_CORE_MMA=1
  NATTEN_WITH_HOPPER_FNA=1
  NATTEN_WITH_BLACKWELL_FNA=1
)

# Split the sources by the architecture they actually target.
#
# The hopper and blackwell families use arch-conditional CUTLASS instructions
# (WGMMA/TMA on sm_90a, tcgen05 on sm_100a/sm_103a) and NATTEN only ever
# dispatches them on those GPUs. Compiling them for a plain "90" instead of
# "90a" silently yields an empty kernel: nvcc -ptx on hopper_fmha/source_0.cu
# emits 584 wgmma ops for sm_90a and 0 for sm_90, with no diagnostic either
# way. An "a" cubin also only runs on exactly that compute capability, so
# blackwell needs both 100a (B200) and 103a (B300). Upstream's setup.py has the
# same mapping (90/100/103 -> 90a/100a/103a).
set(NATTEN_HOPPER_ARCHS "90a-real")
set(NATTEN_BLACKWELL_ARCHS "100a-real;103a-real")

# csrc/src/{hopper,blackwell}_*.cu are the ATen-level wrappers that instantiate
# those kernels, so they go with their family too. Everything else is portable;
# glob the whole tree and subtract, so that a family added upstream lands in
# the portable set instead of being silently dropped from the build.
file(GLOB HOPPER_SOURCES
  ${NATTEN_CSRC}/src/hopper_*.cu
  ${NATTEN_CSRC}/autogen/src/cuda/hopper_*/*.cu
)
file(GLOB BLACKWELL_SOURCES
  ${NATTEN_CSRC}/src/blackwell_*.cu
  ${NATTEN_CSRC}/autogen/src/cuda/blackwell_*/*.cu
)
file(GLOB GENERIC_SOURCES
  ${NATTEN_CSRC}/src/*.cpp
  ${NATTEN_CSRC}/src/*.cu
  ${NATTEN_CSRC}/autogen/src/cuda/*/*.cu
)
list(REMOVE_ITEM GENERIC_SOURCES ${HOPPER_SOURCES} ${BLACKWELL_SOURCES})

macro(natten_configure_kernels tgt)
  set_target_properties(${tgt} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(${tgt} PRIVATE
    ${NATTEN_CSRC}/include
    ${NATTEN_CSRC}/autogen/include
  )
  target_include_directories(${tgt} SYSTEM PRIVATE ${CUTLASS_INCLUDE_DIR})
  target_include_directories(${tgt} SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
endmacro()

add_library(natten_kernels_generic OBJECT ${GENERIC_SOURCES})
set_target_properties(natten_kernels_generic PROPERTIES
  CUDA_ARCHITECTURES "${NATTEN_CUDA_ARCHS}")
natten_configure_kernels(natten_kernels_generic)

add_library(natten_kernels_hopper OBJECT ${HOPPER_SOURCES})
set_target_properties(natten_kernels_hopper PROPERTIES
  CUDA_ARCHITECTURES "${NATTEN_HOPPER_ARCHS}")
natten_configure_kernels(natten_kernels_hopper)

add_library(natten_kernels_blackwell OBJECT ${BLACKWELL_SOURCES})
set_target_properties(natten_kernels_blackwell PROPERTIES
  CUDA_ARCHITECTURES "${NATTEN_BLACKWELL_ARCHS}")
natten_configure_kernels(natten_kernels_blackwell)

add_library(natten_kernels SHARED
  $<TARGET_OBJECTS:natten_kernels_generic>
  $<TARGET_OBJECTS:natten_kernels_hopper>
  $<TARGET_OBJECTS:natten_kernels_blackwell>
)
set_target_properties(natten_kernels PROPERTIES
  LINKER_LANGUAGE CUDA
  POSITION_INDEPENDENT_CODE ON
  # The kernels and the pybind11 module are always built and pinned together,
  # so there is no independent ABI to version with a SOVERSION.
  INSTALL_RPATH "$ORIGIN"
)

# As a static library these stayed unresolved until the extension module linked
# them; a shared library has to resolve them itself. torch_python is absent on
# purpose -- nothing here touches the CPython API, which is what keeps this
# object independent of the Python version.
target_link_directories(natten_kernels PRIVATE ${TORCH_LIBRARY_DIRS})
target_link_libraries(natten_kernels PRIVATE
  c10 torch torch_cpu c10_cuda torch_cuda CUDA::cudart
)

install(TARGETS natten_kernels LIBRARY DESTINATION lib)
