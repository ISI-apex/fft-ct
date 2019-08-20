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
  # FFTW3 single precision
  pkg_check_modules(FFTWF fftw3f)
  if(FFTWF_FOUND)
    include_directories(${FFTWF_INCLUDE_DIRS})
  endif(FFTWF_FOUND)

  # FFTW3 double precision
  pkg_check_modules(FFTW fftw3)
  if(FFTW_FOUND)
    include_directories(${FFTW_INCLUDE_DIRS})
  endif(FFTW_FOUND)

  # Intel MKL
  pkg_check_modules(MKL mkl-static-ilp64-seq)
  if(MKL_FOUND)
    include_directories(${MKL_INCLUDE_DIRS})
  endif(MKL_FOUND)
endif(PKG_CONFIG_FOUND)

include(CheckSymbolExists)

# Some C11 implementations are incomplete, e.g., on OSX with clang
CHECK_SYMBOL_EXISTS(aligned_alloc stdlib.h HAVE_ALIGNED_ALLOC)
if(HAVE_ALIGNED_ALLOC)
  add_definitions(-DHAVE_ALIGNED_ALLOC)
endif(HAVE_ALIGNED_ALLOC)


# Binaries

# Name format: ${prog}-${datatype}-${algo}[-${lib}]
# 'prog' is probably one of:
#   transp, fft-ct, fft-2d
# 'datatype' is probably one of:
#   flt (float), dbl (double), fcmplx (float complex), dcmplx (double complex),
#   fftw (fftw_complex), fftwf (fftwf_complex),
#   cmplx8 (MKL_Complex8), cmplx16 (MKL_Complex16)
# 'algo' is probably one of:
#   naive, blocked,
#   thr{row,col}[-blocked] (thread-by-{row,column} [and blocked]),
#   lib (library-defined),
#   [blocked-]avx-auto ([blocked] AVX-512 automatic),
#   avx-intr[-ss] (AVX-512 intrinsics [with streaming stores]),
#   thr{row,col}-avx-intr (threaded-by-{row,column} AVX-512 intrinsics)
# 'lib' is probably one of:
#   lfftw, lmkl

function(add_exec_prim name main definitions)
  add_executable(${name} ${main} ptime.c transpose.c util.c)
  target_compile_definitions(${name} PRIVATE ${definitions})
endfunction(add_exec_prim)

add_exec_prim(transp-flt-naive transp.c "-DUSE_FLT_NAIVE")
add_exec_prim(transp-dbl-naive transp.c "-DUSE_DBL_NAIVE")
add_exec_prim(transp-fcmplx-naive transp.c "-DUSE_FCMPLX_NAIVE")
add_exec_prim(transp-dcmplx-naive transp.c "-DUSE_DCMPLX_NAIVE")

add_exec_prim(transp-flt-blocked transp.c "-DUSE_FLT_BLOCKED")
add_exec_prim(transp-dbl-blocked transp.c "-DUSE_DBL_BLOCKED")
add_exec_prim(transp-fcmplx-blocked transp.c "-DUSE_FCMPLX_BLOCKED")
add_exec_prim(transp-dcmplx-blocked transp.c "-DUSE_DCMPLX_BLOCKED")

# Use threads
if(Threads_FOUND)
  function(add_exec_threads name main definitions)
    add_executable(${name} ${main} ptime.c transpose-threads.c util.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${CMAKE_THREAD_LIBS_INIT})
  endfunction(add_exec_threads)

  add_exec_threads(transp-flt-thrrow transp.c "-DUSE_FLT_THRROW")
  add_exec_threads(transp-dbl-thrrow transp.c "-DUSE_DBL_THRROW")
  add_exec_threads(transp-flt-thrcol transp.c "-DUSE_FLT_THRCOL")
  add_exec_threads(transp-dbl-thrcol transp.c "-DUSE_DBL_THRCOL")
  add_exec_threads(transp-fcmplx-thrrow transp.c "-DUSE_FCMPLX_THRROW")
  add_exec_threads(transp-dcmplx-thrrow transp.c "-DUSE_DCMPLX_THRROW")
  add_exec_threads(transp-fcmplx-thrcol transp.c "-DUSE_FCMPLX_THRCOL")
  add_exec_threads(transp-dcmplx-thrcol transp.c "-DUSE_DCMPLX_THRCOL")

  add_exec_threads(transp-flt-thrrow-blocked transp.c
                   "-DUSE_FLT_THRROW_BLOCKED")
  add_exec_threads(transp-dbl-thrrow-blocked transp.c
                   "-DUSE_DBL_THRROW_BLOCKED")
  add_exec_threads(transp-flt-thrcol-blocked transp.c
                   "-DUSE_FLT_THRCOL_BLOCKED")
  add_exec_threads(transp-dbl-thrcol-blocked transp.c
                   "-DUSE_DBL_THRCOL_BLOCKED")
  add_exec_threads(transp-fcmplx-thrrow-blocked transp.c
                   "-DUSE_FCMPLX_THRROW_BLOCKED")
  add_exec_threads(transp-dcmplx-thrrow-blocked transp.c
                   "-DUSE_DCMPLX_THRROW_BLOCKED")
  add_exec_threads(transp-fcmplx-thrcol-blocked transp.c
                   "-DUSE_FCMPLX_THRCOL_BLOCKED")
  add_exec_threads(transp-dcmplx-thrcol-blocked transp.c
                   "-DUSE_DCMPLX_THRCOL_BLOCKED")
endif(Threads_FOUND)

# Use FFTWF library
if(FFTWF_FOUND)
  function(add_exec_fftwf name main definitions)
    add_executable(${name} ${main} ptime.c transpose.c transpose-fftwf.c
                                   util.c util-fftwf.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${FFTWF_LDFLAGS} ${FFTWF_STATIC_LIBRARIES})
  endfunction(add_exec_fftwf)

  add_exec_fftwf(transp-fftwf-naive transp.c "-DUSE_FFTWF_NAIVE")
  add_exec_fftwf(transp-fftwf-blocked transp.c "-DUSE_FFTWF_BLOCKED")

  add_exec_fftwf(fft-ct-fftwf-naive fft-ct.c "-DUSE_FFTWF_NAIVE")
  add_exec_fftwf(fft-ct-fftwf-blocked fft-ct.c "-DUSE_FFTWF_BLOCKED")

  add_exec_fftwf(fft-2d-fftwf-lib-lfftw fft-2d.c "-DUSE_FFTWF")
endif(FFTWF_FOUND)

if(Threads_FOUND AND FFTWF_FOUND)
  function(add_exec_threads_fftwf name main definitions)
    add_executable(${name} ${main} ptime.c
                                   transpose-threads.c transpose-threads-fftwf.c
                                   util.c util-fftwf.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${CMAKE_THREAD_LIBS_INIT}
                          ${FFTWF_LDFLAGS} ${FFTWF_STATIC_LIBRARIES})
  endfunction(add_exec_threads_fftwf)

  add_exec_threads_fftwf(transp-fftwf-thrrow transp.c "-DUSE_FFTWF_THRROW")
  add_exec_threads_fftwf(transp-fftwf-thrcol transp.c "-DUSE_FFTWF_THRCOL")
  add_exec_threads_fftwf(transp-fftwf-thrrow-blocked transp.c
                         "-DUSE_FFTWF_THRROW_BLOCKED")
  add_exec_threads_fftwf(transp-fftwf-thrcol-blocked transp.c
                         "-DUSE_FFTWF_THRCOL_BLOCKED")

  add_exec_threads_fftwf(fft-ct-fftwf-thrrow fft-ct.c "-DUSE_FFTWF_THRROW")
  add_exec_threads_fftwf(fft-ct-fftwf-thrcol fft-ct.c "-DUSE_FFTWF_THRCOL")
  add_exec_threads_fftwf(fft-ct-fftwf-thrrow-blocked fft-ct.c
                         "-DUSE_FFTWF_THRROW_BLOCKED")
  add_exec_threads_fftwf(fft-ct-fftwf-thrcol-blocked fft-ct.c
                         "-DUSE_FFTWF_THRCOL_BLOCKED")
endif(Threads_FOUND AND FFTWF_FOUND)

# Use FFTW library
if(FFTW_FOUND)
  function(add_exec_fftw name main definitions)
    add_executable(${name} ${main} ptime.c transpose.c transpose-fftw.c
                                   util.c util-fftw.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${FFTW_LDFLAGS} ${FFTW_STATIC_LIBRARIES})
  endfunction(add_exec_fftw)

  add_exec_fftw(transp-fftw-naive transp.c "-DUSE_FFTW_NAIVE")
  add_exec_fftw(transp-fftw-blocked transp.c "-DUSE_FFTW_BLOCKED")

  add_exec_fftw(fft-ct-fftw-naive fft-ct.c "-DUSE_FFTW_NAIVE")
  add_exec_fftw(fft-ct-fftw-blocked fft-ct.c "-DUSE_FFTW_BLOCKED")

  add_exec_fftw(fft-2d-fftw-lib-lfftw fft-2d.c "")
endif(FFTW_FOUND)

if(Threads_FOUND AND FFTW_FOUND)
  function(add_exec_threads_fftw name main definitions)
    add_executable(${name} ${main} ptime.c
                                   transpose-threads.c transpose-threads-fftw.c
                                   util.c util-fftw.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${CMAKE_THREAD_LIBS_INIT}
                          ${FFTW_LDFLAGS} ${FFTW_STATIC_LIBRARIES})
  endfunction(add_exec_threads_fftw)

  add_exec_threads_fftw(transp-fftw-thrrow transp.c "-DUSE_FFTW_THRROW")
  add_exec_threads_fftw(transp-fftw-thrcol transp.c "-DUSE_FFTW_THRCOL")
  add_exec_threads_fftw(transp-fftw-thrrow-blocked transp.c
                        "-DUSE_FFTW_THRROW_BLOCKED")
  add_exec_threads_fftw(transp-fftw-thrcol-blocked transp.c
                        "-DUSE_FFTW_THRCOL_BLOCKED")

  add_exec_threads_fftw(fft-ct-fftw-thrrow fft-ct.c "-DUSE_FFTW_THRROW")
  add_exec_threads_fftw(fft-ct-fftw-thrcol fft-ct.c "-DUSE_FFTW_THRCOL")
  add_exec_threads_fftw(fft-ct-fftw-thrrow-blocked fft-ct.c
                        "-DUSE_FFTW_THRROW_BLOCKED")
  add_exec_threads_fftw(fft-ct-fftw-thrcol-blocked fft-ct.c
                        "-DUSE_FFTW_THRCOL_BLOCKED")
endif(Threads_FOUND AND FFTW_FOUND)

# Use MKL library
if(MKL_FOUND)
  function(add_exec_mkl name main definitions)
    add_executable(${name} ${main} ptime.c transpose-mkl.c util.c util-mkl.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_compile_options(${name} PRIVATE ${MKL_CFLAGS} ${MKL_CFLAGS_OTHER})
    target_link_libraries(${name} ${MKL_LDFLAGS})
  endfunction(add_exec_mkl)

  add_exec_mkl(transp-flt-lib-lmkl transp.c "-DUSE_MKL_FLOAT")
  add_exec_mkl(transp-dbl-lib-lmkl transp.c "-DUSE_MKL_DOUBLE")
  add_exec_mkl(transp-cmplx8-lib-lmkl transp.c "-DUSE_MKL_CMPLX8")
  add_exec_mkl(transp-cmplx16-lib-lmkl transp.c "-DUSE_MKL_CMPLX16")
endif(MKL_FOUND)

# Use MKL library implementations of the FFTWF interface
if(MKL_FOUND)
  function(add_exec_mkl_fftwf name main definitions)
    add_executable(${name} ${main} ptime.c transpose.c transpose-fftwf.c
                                   util.c util-fftwf.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${MKL_LDFLAGS})
  endfunction(add_exec_mkl_fftwf)

  add_exec_mkl_fftwf(transp-fftwf-naive-lmkl transp.c "-DUSE_FFTWF_NAIVE")
  add_exec_mkl_fftwf(transp-fftwf-blocked-lmkl transp.c "-DUSE_FFTWF_BLOCKED")

  add_exec_mkl_fftwf(fft-ct-fftwf-naive-lmkl fft-ct.c "-DUSE_FFTWF_NAIVE")
  add_exec_mkl_fftwf(fft-ct-fftwf-blocked-lmkl fft-ct.c "-DUSE_FFTWF_BLOCKED")

  add_exec_mkl_fftwf(fft-2d-fftwf-lib-lmkl fft-2d.c "-DUSE_FFTWF")
endif(MKL_FOUND)

# Use MKL library implementations of the FFTW interface
if(MKL_FOUND)
  function(add_exec_mkl_fftw name main definitions)
    add_executable(${name} ${main} ptime.c transpose.c transpose-fftw.c
                                   util.c util-fftw.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${MKL_LDFLAGS})
  endfunction(add_exec_mkl_fftw)

  add_exec_mkl_fftw(transp-fftw-naive-lmkl transp.c "-DUSE_FFTW_NAIVE")
  add_exec_mkl_fftw(transp-fftw-blocked-lmkl transp.c "-DUSE_FFTW_BLOCKED")

  add_exec_mkl_fftw(fft-ct-fftw-naive-lmkl fft-ct.c "-DUSE_FFTW_NAIVE")
  add_exec_mkl_fftw(fft-ct-fftw-blocked-lmkl fft-ct.c "-DUSE_FFTW_BLOCKED")

  add_exec_mkl_fftw(fft-2d-fftw-lib-lmkl fft-2d.c "")
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
  elseif("${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
    if(CMAKE_C_COMPILER_VERSION VERSION_LESS 3.9.1)
      message(FATAL_ERROR "Clang compiler version is too old for AVX-512: "
                          "${CMAKE_C_COMPILER_VERSION} < 3.9.1\n"
                          "Set option: ENABLE_AVX=OFF")
    endif(CMAKE_C_COMPILER_VERSION VERSION_LESS 3.9.1)
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
                        "Set C_FLAGS_AVX explicitly or set ENABLE_AVX=OFF")
  endif()
endif(ENABLE_AVX AND NOT C_FLAGS_AVX)

# Use automatic and intrinsic AVX
if(ENABLE_AVX)
  message("--   C_FLAGS_AVX: ${C_FLAGS_AVX}")
  function(add_exec_avx name main definitions)
    add_executable(${name} ${main} ptime.c transpose.c transpose-avx.c util.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    string(REPLACE " " ";" C_FLAGS_AVX_LIST ${C_FLAGS_AVX}) # string->list
    target_compile_options(${name} PRIVATE ${C_FLAGS_AVX_LIST})
  endfunction(add_exec_avx)

  add_exec_avx(transp-flt-avx-auto transp.c "-DUSE_FLT_NAIVE")
  add_exec_avx(transp-dbl-avx-auto transp.c "-DUSE_DBL_NAIVE")
  add_exec_avx(transp-flt-blocked-avx-auto transp.c "-DUSE_FLT_BLOCKED")
  add_exec_avx(transp-dbl-blocked-avx-auto transp.c "-DUSE_DBL_BLOCKED")

  add_exec_avx(transp-dbl-avx-intr transp.c "-DUSE_DBL_AVX_INTR_8X8")
  add_exec_avx(transp-dbl-avx-intr-ss transp.c
               "-DUSE_DBL_AVX_INTR_8X8;-DUSE_AVX_STREAMING_STORES")
endif(ENABLE_AVX)

# Use threads with intrinsic AVX
if(Threads_FOUND AND ENABLE_AVX)
  function(add_exec_threads_avx name main definitions)
    add_executable(${name} ${main} ptime.c transpose.c transpose-threads-avx.c
                                   util.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    string(REPLACE " " ";" C_FLAGS_AVX_LIST ${C_FLAGS_AVX}) # string->list
    target_compile_options(${name} PRIVATE ${C_FLAGS_AVX_LIST})
    target_link_libraries(${name} ${CMAKE_THREAD_LIBS_INIT})
  endfunction(add_exec_threads_avx)

  add_exec_threads_avx(transp-dbl-thrrow-avx-intr transp.c
                       "-DUSE_DBL_THREADS_AVX_INTR_8X8_ROW")
  add_exec_threads_avx(transp-dbl-thrcol-avx-intr transp.c
                       "-DUSE_DBL_THREADS_AVX_INTR_8X8_COL")
endif(Threads_FOUND AND ENABLE_AVX)
