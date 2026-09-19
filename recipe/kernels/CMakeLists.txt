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
)

# The sources test these with #if defined(), so only define them when enabled
if(NATTEN_WITH_HOPPER_FNA)
  add_compile_definitions(NATTEN_WITH_HOPPER_FNA=1)
endif()
if(NATTEN_WITH_BLACKWELL_FNA)
  add_compile_definitions(NATTEN_WITH_BLACKWELL_FNA=1)
endif()

# Split the sources by the architecture they actually target.
#
# The hopper and blackwell families use arch-conditional CUTLASS instructions
# (WGMMA/TMA on sm_90a, tcgen05 on sm_100a) and NATTEN only ever dispatches
# them on those GPUs. Compiling them for the whole arch list is pure waste --
# and worse, compiling them for a plain "90" instead of "90a" silently yields
# an empty kernel: nvcc -ptx on hopper_fmha/source_0.cu emits 584 wgmma ops
# for sm_90a and 0 for sm_90, with no diagnostic either way. Upstream's
# setup.py has the same mapping (90/100/103 -> 90a/100a/103a).
#
# So each family gets its own object library with its own CUDA_ARCHITECTURES,
# and they are combined into one shared library at the end.
file(GLOB SRC_SOURCES ${NATTEN_CSRC}/src/*.cpp ${NATTEN_CSRC}/src/*.cu)
file(GLOB HOPPER_SRC ${NATTEN_CSRC}/src/hopper_*.cu)
file(GLOB BLACKWELL_SRC ${NATTEN_CSRC}/src/blackwell_*.cu)
file(GLOB HOPPER_AUTOGEN ${NATTEN_CSRC}/autogen/src/cuda/hopper_*/*.cu)
file(GLOB BLACKWELL_AUTOGEN ${NATTEN_CSRC}/autogen/src/cuda/blackwell_*/*.cu)

# Everything that is not a datacenter autogen directory is portable. Glob the
# whole autogen tree and subtract, rather than listing the portable families,
# so that a family added upstream lands in the portable set instead of being
# silently dropped from the build.
file(GLOB GENERIC_SOURCES
  ${NATTEN_CSRC}/autogen/src/cuda/*/*.cu
)
list(APPEND GENERIC_SOURCES ${SRC_SOURCES})
if(HOPPER_AUTOGEN)
  list(REMOVE_ITEM GENERIC_SOURCES ${HOPPER_AUTOGEN})
endif()
if(BLACKWELL_AUTOGEN)
  list(REMOVE_ITEM GENERIC_SOURCES ${BLACKWELL_AUTOGEN})
endif()

# csrc/src/{hopper,blackwell}_*.cu are the ATen-level wrappers, and natten.cpp
# registers their entry points unconditionally -- there is no #if around the
# pybind11 defs. With the family enabled they instantiate its kernels, so they
# have to move to that family's architecture. With it disabled they still have
# to be built, as the stubs that raise "not compiled with ..." at runtime;
# leaving them out costs the shared library those symbols and the extension
# module then fails to link.
set(HOPPER_SOURCES ${HOPPER_AUTOGEN})
set(BLACKWELL_SOURCES ${BLACKWELL_AUTOGEN})
if(NATTEN_WITH_HOPPER_FNA AND HOPPER_SRC)
  list(APPEND HOPPER_SOURCES ${HOPPER_SRC})
  list(REMOVE_ITEM GENERIC_SOURCES ${HOPPER_SRC})
endif()
if(NATTEN_WITH_BLACKWELL_FNA AND BLACKWELL_SRC)
  list(APPEND BLACKWELL_SOURCES ${BLACKWELL_SRC})
  list(REMOVE_ITEM GENERIC_SOURCES ${BLACKWELL_SRC})
endif()

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
set(NATTEN_OBJECTS $<TARGET_OBJECTS:natten_kernels_generic>)

if(NATTEN_WITH_HOPPER_FNA AND HOPPER_SOURCES)
  add_library(natten_kernels_hopper OBJECT ${HOPPER_SOURCES})
  set_target_properties(natten_kernels_hopper PROPERTIES
    CUDA_ARCHITECTURES "${NATTEN_HOPPER_ARCHS}")
  natten_configure_kernels(natten_kernels_hopper)
  list(APPEND NATTEN_OBJECTS $<TARGET_OBJECTS:natten_kernels_hopper>)
endif()

if(NATTEN_WITH_BLACKWELL_FNA AND BLACKWELL_SOURCES)
  add_library(natten_kernels_blackwell OBJECT ${BLACKWELL_SOURCES})
  set_target_properties(natten_kernels_blackwell PROPERTIES
    CUDA_ARCHITECTURES "${NATTEN_BLACKWELL_ARCHS}")
  natten_configure_kernels(natten_kernels_blackwell)
  list(APPEND NATTEN_OBJECTS $<TARGET_OBJECTS:natten_kernels_blackwell>)
endif()

add_library(natten_kernels SHARED ${NATTEN_OBJECTS})
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
