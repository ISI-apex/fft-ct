cmake_minimum_required(VERSION 2.8.12)

project(fft-ct)
set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)
set(VERSION_PATCH 0)
set(PROJECT_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -std=c11")

option(ENABLE_AVX "Enable Skylake AVX-512 binaries" ON)


# Dependencies

find_package(Threads)

find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(FFTW fftw3)
  if(FFTW_FOUND)
    include_directories(${FFTW_INCLUDE_DIRS})
  endif(FFTW_FOUND)

  pkg_check_modules(MKL mkl-static-ilp64-seq)
  if(MKL_FOUND)
    include_directories(${MKL_INCLUDE_DIRS})
  endif(MKL_FOUND)
endif(PKG_CONFIG_FOUND)


# Binaries

# Name format: ${prog}-${datatype}-{impl}
# datatype is probably one of:
# [float, double, fftw_complex, MKL_Complex8, MKL_Complex16]
# impl is probably one or more of:
# [naive, blocked, mkl, mkl-fftw, pthread]

function(add_exec_prim name main definitions)
  add_executable(${name} ${main} transpose.c util.c)
  target_compile_definitions(${name} PRIVATE ${definitions})
endfunction(add_exec_prim)

add_exec_prim(transp-float-naive transp.c "-DUSE_FLOAT_NAIVE")
add_exec_prim(transp-double-naive transp.c "-DUSE_DOUBLE_NAIVE")
add_exec_prim(transp-float-blocked transp.c "-DUSE_FLOAT_BLOCKED")
add_exec_prim(transp-double-blocked transp.c "-DUSE_DOUBLE_BLOCKED")

# Use threads
if(Threads_FOUND)
  function(add_exec_threads name main definitions)
    add_executable(${name} ${main} transpose-threads.c util.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${CMAKE_THREAD_LIBS_INIT})
  endfunction(add_exec_threads)

  add_exec_threads(transp-float-threads-row transp.c "-DUSE_FLOAT_THREADS_ROW")
  add_exec_threads(transp-double-threads-row transp.c "-DUSE_DOUBLE_THREADS_ROW")
  add_exec_threads(transp-float-threads-col transp.c "-DUSE_FLOAT_THREADS_COL")
  add_exec_threads(transp-double-threads-col transp.c "-DUSE_DOUBLE_THREADS_COL")
endif(Threads_FOUND)

# Use FFTW library
if(FFTW_FOUND)
  function(add_exec_fftw name main definitions)
    add_executable(${name} ${main} transpose-fftw.c util.c util-fftw.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${FFTW_LDFLAGS} ${FFTW_STATIC_LIBRARIES})
  endfunction(add_exec_fftw)

  add_exec_fftw(transp-fftw_complex-naive transp.c "-DUSE_FFTW_NAIVE")
  add_exec_fftw(fft-ct-fftw_complex-naive fft-ct.c "")
  add_exec_fftw(fft-2d-fftw_complex-fftw fft-2d.c "")
endif(FFTW_FOUND)

# Use MKL library
if(MKL_FOUND)
  function(add_exec_mkl name main definitions)
    add_executable(${name} ${main} transpose-mkl.c util.c util-mkl.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_compile_options(${name} PRIVATE ${MKL_CFLAGS} ${MKL_CFLAGS_OTHER})
    target_link_libraries(${name} ${MKL_LDFLAGS})
  endfunction(add_exec_mkl)

  add_exec_mkl(transp-float-mkl transp.c "-DUSE_MKL_FLOAT")
  add_exec_mkl(transp-double-mkl transp.c "-DUSE_MKL_DOUBLE")
  add_exec_mkl(transp-cmplx8-mkl transp.c "-DUSE_MKL_CMPLX8")
  add_exec_mkl(transp-cmplx16-mkl transp.c "-DUSE_MKL_CMPLX16")
endif(MKL_FOUND)

# Use MKL library implementations of the FFTW interface
if(MKL_FOUND)
  function(add_exec_mkl_fftw name main definitions)
    add_executable(${name} ${main} transpose-fftw.c util.c util-fftw.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${MKL_LDFLAGS})
  endfunction(add_exec_mkl_fftw)

  add_exec_mkl_fftw(transp-fftw_complex-naive-mkl-fftw transp.c "-DUSE_FFTW_NAIVE")
  add_exec_mkl_fftw(fft-ct-fftw_complex-naive-mkl-fftw fft-ct.c "")
  add_exec_mkl_fftw(fft-2d-fftw_complex-mkl-fftw fft-2d.c "")
endif(MKL_FOUND)

# A primitive approach for setting user-specified or default AVX compile flags
# Note: Complete auto-detection for architectures and compilers would require
#       checking target CPUIDs; the list would also get outdated with new CPUs.
#       Maybe another project maintains such a CMake capability we can use.
set(C_FLAGS_AVX CACHE STRING "C compiler AVX flags (implies ENABLE_AVX)")
if(C_FLAGS_AVX)
  message("-- Checking for C_FLAGS_AVX - found")
  set(ENABLE_AVX ON)
else(C_FLAGS_AVX)
  message("-- Checking for C_FLAGS_AVX - not found")
endif(C_FLAGS_AVX)
if(ENABLE_AVX AND NOT C_FLAGS_AVX)
  message("--   Defaulting to AVX-512 for Skylake...")
  # TODO: Does specifying optimization level conflict with CMAKE_BUILD_TYPE?
  #       Certain per-compiler levels are needed for automatic vectorization.
  if("${CMAKE_C_COMPILER_ID}" MATCHES "GNU")
    if(CMAKE_C_COMPILER_VERSION VERSION_LESS 5.0)
      message(FATAL_ERROR "GNU C compiler version is too old for AVX-512: "
                          "${CMAKE_C_COMPILER_VERSION} < 5.0\n"
                          "Set option: ENABLE_AVX=OFF")
    endif(CMAKE_C_COMPILER_VERSION VERSION_LESS 5.0)
    set(C_FLAGS_AVX "-march=skylake-avx512 -mtune=skylake-avx512 -ffast-math -O3")
  elseif("${CMAKE_C_COMPILER_ID}" MATCHES "Intel")
    if(CMAKE_C_COMPILER_VERSION VERSION_LESS 15.0.1)
      message(FATAL_ERROR "Intel C compiler version is too old for AVX-512: "
                          "${CMAKE_C_COMPILER_VERSION} < 15.0.1\n"
                          "Set option: ENABLE_AVX=OFF")
    endif(CMAKE_C_COMPILER_VERSION VERSION_LESS 15.0.1)
    set(C_FLAGS_AVX "-xCORE-AVX512 -qopt-zmm-usage=high -O3")
  else()
    message(FATAL_ERROR "No default AVX-512 flags for C compiler: "
                        "${CMAKE_C_COMPILER_ID}\n"
                        "Set option: ENABLE_AVX=OFF")
  endif()
endif(ENABLE_AVX AND NOT C_FLAGS_AVX)

# Use automatic and intrinsic AVX
if(ENABLE_AVX)
  message("--   C_FLAGS_AVX: ${C_FLAGS_AVX}")
  function(add_exec_avx name main definitions)
    add_executable(${name} ${main} transpose.c transpose-avx.c util.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    string(REPLACE " " ";" C_FLAGS_AVX_LIST ${C_FLAGS_AVX}) # string->list
    target_compile_options(${name} PRIVATE ${C_FLAGS_AVX_LIST})
  endfunction(add_exec_avx)

  add_exec_avx(transp-float-avx-auto transp.c "-DUSE_FLOAT_NAIVE")
  add_exec_avx(transp-double-avx-auto transp.c "-DUSE_DOUBLE_NAIVE")
  add_exec_avx(transp-float-blocked-avx-auto transp.c "-DUSE_FLOAT_BLOCKED")
  add_exec_avx(transp-double-blocked-avx-auto transp.c "-DUSE_DOUBLE_BLOCKED")

  # add_exec_avx(transp-float-avx-intr-8x8 transp.c "-DUSE_FLOAT_AVX_INTR_8X8")
  add_exec_avx(transp-double-avx-intr-8x8 transp.c "-DUSE_DOUBLE_AVX_INTR_8X8")
endif(ENABLE_AVX)
