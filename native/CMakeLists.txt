cmake_minimum_required(VERSION 3.15)
project(donksolver_core CXX)

# Builds `donksolver_core`, a shared library with a plain C ABI, loaded by the
# Python package through ctypes. Driven by scikit-build-core when building a
# wheel (pip / python -m build / cibuildwheel), or by hand:
#
#   cmake -S native -B native/build && cmake --build native/build --config Release
#
# then point DONKSOLVER_LIB at the library, or copy it into
# src/donksolver/_native/.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# AVX2 is opt-in and off in every published wheel: a wheel must run on any CPU
# of its architecture, and it measured as no solve speedup (see README).
option(DONK_AVX2 "compile with AVX2 (x86-64 only)" OFF)

add_library(donksolver_core SHARED src/capi.cpp)
target_include_directories(donksolver_core PRIVATE include)

# The version reported by ds_version() is the package version.
if(DEFINED SKBUILD_PROJECT_VERSION)
  target_compile_definitions(donksolver_core PRIVATE DONK_VERSION="${SKBUILD_PROJECT_VERSION}")
endif()

find_package(Threads REQUIRED)
target_link_libraries(donksolver_core PRIVATE Threads::Threads)

# Load it by one fixed name on every platform: no "lib" prefix, so ctypes
# looks for donksolver_core.{dll,so,dylib}.
set_target_properties(donksolver_core PROPERTIES PREFIX "")

if(MSVC)
  # Static CRT: the DLL then depends only on system DLLs.
  set_property(TARGET donksolver_core PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_compile_definitions(donksolver_core PRIVATE _CRT_SECURE_NO_WARNINGS)
  target_compile_options(donksolver_core PRIVATE /EHsc /W3 $<$<CONFIG:Release>:/O2 /Ob3 /GL /Gw>)
  target_link_options(donksolver_core PRIVATE $<$<CONFIG:Release>:/LTCG /OPT:REF /OPT:ICF>)
  if(DONK_AVX2)
    target_compile_options(donksolver_core PRIVATE /arch:AVX2)
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(donksolver_core PRIVATE $<$<CONFIG:Release>:-O3>)
  if(DONK_AVX2)
    target_compile_options(donksolver_core PRIVATE -march=x86-64-v3)
  endif()
  if(MINGW)
    # MinGW's C++ runtime and winpthread are not on a user's PATH.
    target_link_options(donksolver_core PRIVATE -static)
  endif()
  # Linux links libstdc++ dynamically. A static libstdc++ inside a dlopen'ed
  # library crashed in its iostream locale code (Ubuntu 24.04, gcc 13.3; see
  # docs/RESULTS.md), and the manylinux images already keep the wheel within
  # the libstdc++ every supported distribution has.
endif()

# scikit-build-core installs relative to the wheel's platlib root.
install(TARGETS donksolver_core
  LIBRARY DESTINATION donksolver/_native
  RUNTIME DESTINATION donksolver/_native)
