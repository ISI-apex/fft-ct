cmake_minimum_required(VERSION 2.8.12)

project(fft-ct)
set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)
set(VERSION_PATCH 0)
set(PROJECT_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -std=c11")

option(CFLAGS_AVX_VERBOSE "Use verbose AVX compile flags" OFF)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()


# Dependencies

if(CMAKE_HOST_UNIX)
  include(GNUInstallDirs)
else()
  set(CMAKE_INSTALL_LIBDIR lib)
  set(CMAKE_INSTALL_INCLUDEDIR include)
  set(CMAKE_INSTALL_BINDIR bin)
  set(CMAKE_INSTALL_MANDIR share/man)
endif()

include(CheckFunctionExists)
include(CheckSymbolExists)

# Determine if we should link with librt
if(UNIX AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  CHECK_FUNCTION_EXISTS(clock_gettime HAVE_CLOCK_GETTIME)
  if(NOT HAVE_CLOCK_GETTIME)
    find_library(LIBRT NAMES rt)
  endif(NOT HAVE_CLOCK_GETTIME)
endif(UNIX AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

# Determine if we should link with libm
if(UNIX)
  set(LIBM m)
endif(UNIX)

find_package(Threads)

find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
  # FFTW3 single precision
  pkg_check_modules(FFTWF fftw3f)
  if(FFTWF_FOUND)
    link_directories(${FFTWF_LIBRARY_DIRS})
  endif(FFTWF_FOUND)

  # FFTW3 double precision
  pkg_check_modules(FFTW fftw3)
  if(FFTW_FOUND)
    link_directories(${FFTW_LIBRARY_DIRS})
  endif(FFTW_FOUND)

  # Intel MKL
  pkg_check_modules(MKL mkl-static-ilp64-seq)
endif(PKG_CONFIG_FOUND)

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
#   avx512-intr[-ss] (AVX-512 intrinsics [with streaming stores]),
#   thr{row,col}-avx512-intr (threaded-by-{row,column} AVX-512 intrinsics)
# 'lib' is probably one of:
#   lfftwf, lfftw, lmkl

function(add_exec_prim name main definitions)
  add_executable(${name} ${main} ptime.c transpose.c util.c)
  target_compile_definitions(${name} PRIVATE ${definitions})
  target_link_libraries(${name} ${LIBRT} ${LIBM})
  install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction(add_exec_prim)

add_exec_prim(transp-flt-naive transp.c "-DUSE_FLT_NAIVE")
add_exec_prim(transp-flt-blocked transp.c "-DUSE_FLT_BLOCKED")
add_exec_prim(transp-dbl-naive transp.c "-DUSE_DBL_NAIVE")
add_exec_prim(transp-dbl-blocked transp.c "-DUSE_DBL_BLOCKED")
add_exec_prim(transp-fcmplx-naive transp.c "-DUSE_FCMPLX_NAIVE")
add_exec_prim(transp-fcmplx-blocked transp.c "-DUSE_FCMPLX_BLOCKED")
add_exec_prim(transp-dcmplx-naive transp.c "-DUSE_DCMPLX_NAIVE")
add_exec_prim(transp-dcmplx-blocked transp.c "-DUSE_DCMPLX_BLOCKED")

# Use threads
if(Threads_FOUND)
  function(add_exec_threads name main definitions)
    add_executable(${name} ${main} ptime.c transpose-threads.c util.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${CMAKE_THREAD_LIBS_INIT} ${LIBRT} ${LIBM})
    install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})
  endfunction(add_exec_threads)

  add_exec_threads(transp-flt-thrrow transp.c "-DUSE_FLT_THRROW")
  add_exec_threads(transp-flt-thrcol transp.c "-DUSE_FLT_THRCOL")
  add_exec_threads(transp-flt-thrrow-blocked transp.c
                   "-DUSE_FLT_THRROW_BLOCKED")
  add_exec_threads(transp-flt-thrcol-blocked transp.c
                   "-DUSE_FLT_THRCOL_BLOCKED")
  add_exec_threads(transp-dbl-thrrow transp.c "-DUSE_DBL_THRROW")
  add_exec_threads(transp-dbl-thrcol transp.c "-DUSE_DBL_THRCOL")
  add_exec_threads(transp-dbl-thrrow-blocked transp.c
                   "-DUSE_DBL_THRROW_BLOCKED")
  add_exec_threads(transp-dbl-thrcol-blocked transp.c
                   "-DUSE_DBL_THRCOL_BLOCKED")
  add_exec_threads(transp-fcmplx-thrrow transp.c "-DUSE_FCMPLX_THRROW")
  add_exec_threads(transp-fcmplx-thrcol transp.c "-DUSE_FCMPLX_THRCOL")
  add_exec_threads(transp-fcmplx-thrrow-blocked transp.c
                   "-DUSE_FCMPLX_THRROW_BLOCKED")
  add_exec_threads(transp-fcmplx-thrcol-blocked transp.c
                   "-DUSE_FCMPLX_THRCOL_BLOCKED")
  add_exec_threads(transp-dcmplx-thrrow transp.c "-DUSE_DCMPLX_THRROW")
  add_exec_threads(transp-dcmplx-thrcol transp.c "-DUSE_DCMPLX_THRCOL")
  add_exec_threads(transp-dcmplx-thrrow-blocked transp.c
                   "-DUSE_DCMPLX_THRROW_BLOCKED")
  add_exec_threads(transp-dcmplx-thrcol-blocked transp.c
                   "-DUSE_DCMPLX_THRCOL_BLOCKED")
endif(Threads_FOUND)

# Use FFTWF library
if(FFTWF_FOUND)
  function(add_exec_fftwf name main definitions)
    add_executable(${name} ${main} ptime.c transpose.c transpose-fftwf.c
                                   util.c util-fftwf.c)
    target_compile_options(${name} PRIVATE ${FFTWF_CFLAGS}
                                           ${FFTWF_CFLAGS_OTHER})
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${FFTWF_STATIC_LIBRARIES}
                                  ${LIBRT} ${LIBM})
    install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})
  endfunction(add_exec_fftwf)

  add_exec_fftwf(transp-fftwf-naive transp.c "-DUSE_FFTWF_NAIVE")
  add_exec_fftwf(transp-fftwf-blocked transp.c "-DUSE_FFTWF_BLOCKED")

  add_exec_fftwf(fft-ct-fftwf-naive fft-ct.c "-DUSE_FFTWF_NAIVE")
  add_exec_fftwf(fft-ct-fftwf-blocked fft-ct.c "-DUSE_FFTWF_BLOCKED")

  add_exec_fftwf(fft-2d-fftwf-lib-lfftwf fft-2d.c "-DUSE_FFTWF")
endif(FFTWF_FOUND)

# Use FFTWF library with threads
if(FFTWF_FOUND AND Threads_FOUND)
  function(add_exec_fftwf_threads name main definitions)
    add_executable(${name} ${main} ptime.c fft-threads-fftwf.c
                                   transpose-threads.c transpose-fftwf-threads.c
                                   util.c util-fftwf.c)
    target_compile_options(${name} PRIVATE ${FFTWF_CFLAGS}
                                           ${FFTWF_CFLAGS_OTHER})
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${FFTWF_STATIC_LIBRARIES}
                                  ${CMAKE_THREAD_LIBS_INIT}
                                  ${LIBRT} ${LIBM})
    install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})
  endfunction(add_exec_fftwf_threads)

  add_exec_fftwf_threads(transp-fftwf-thrrow transp.c "-DUSE_FFTWF_THRROW")
  add_exec_fftwf_threads(transp-fftwf-thrcol transp.c "-DUSE_FFTWF_THRCOL")
  add_exec_fftwf_threads(transp-fftwf-thrrow-blocked transp.c
                         "-DUSE_FFTWF_THRROW_BLOCKED")
  add_exec_fftwf_threads(transp-fftwf-thrcol-blocked transp.c
                         "-DUSE_FFTWF_THRCOL_BLOCKED")

  add_exec_fftwf_threads(fft-ct-fftwf-thrrow fft-ct.c "-DUSE_FFTWF_THRROW")
  add_exec_fftwf_threads(fft-ct-fftwf-thrcol fft-ct.c "-DUSE_FFTWF_THRCOL")
  add_exec_fftwf_threads(fft-ct-fftwf-thrrow-blocked fft-ct.c
                         "-DUSE_FFTWF_THRROW_BLOCKED")
  add_exec_fftwf_threads(fft-ct-fftwf-thrcol-blocked fft-ct.c
                         "-DUSE_FFTWF_THRCOL_BLOCKED")
endif(FFTWF_FOUND AND Threads_FOUND)

# Use FFTW library
if(FFTW_FOUND)
  function(add_exec_fftw name main definitions)
    add_executable(${name} ${main} ptime.c transpose.c transpose-fftw.c
                                   util.c util-fftw.c)
    target_compile_options(${name} PRIVATE ${FFTW_CFLAGS} ${FFTW_CFLAGS_OTHER})
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${FFTW_STATIC_LIBRARIES}
                                  ${LIBRT} ${LIBM})
    install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})
  endfunction(add_exec_fftw)

  add_exec_fftw(transp-fftw-naive transp.c "-DUSE_FFTW_NAIVE")
  add_exec_fftw(transp-fftw-blocked transp.c "-DUSE_FFTW_BLOCKED")

  add_exec_fftw(fft-ct-fftw-naive fft-ct.c "-DUSE_FFTW_NAIVE")
  add_exec_fftw(fft-ct-fftw-blocked fft-ct.c "-DUSE_FFTW_BLOCKED")

  add_exec_fftw(fft-2d-fftw-lib-lfftw fft-2d.c "")
endif(FFTW_FOUND)

# Use FFTW library with threads
if(FFTW_FOUND AND Threads_FOUND)
  function(add_exec_fftw_threads name main definitions)
    add_executable(${name} ${main} ptime.c fft-threads-fftw.c
                                   transpose-threads.c transpose-fftw-threads.c
                                   util.c util-fftw.c)
    target_compile_options(${name} PRIVATE ${FFTW_CFLAGS} ${FFTW_CFLAGS_OTHER})
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${FFTW_STATIC_LIBRARIES}
                                  ${CMAKE_THREAD_LIBS_INIT}
                                  ${LIBRT} ${LIBM})
    install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})
  endfunction(add_exec_fftw_threads)

  add_exec_fftw_threads(transp-fftw-thrrow transp.c "-DUSE_FFTW_THRROW")
  add_exec_fftw_threads(transp-fftw-thrcol transp.c "-DUSE_FFTW_THRCOL")
  add_exec_fftw_threads(transp-fftw-thrrow-blocked transp.c
                        "-DUSE_FFTW_THRROW_BLOCKED")
  add_exec_fftw_threads(transp-fftw-thrcol-blocked transp.c
                        "-DUSE_FFTW_THRCOL_BLOCKED")

  add_exec_fftw_threads(fft-ct-fftw-thrrow fft-ct.c "-DUSE_FFTW_THRROW")
  add_exec_fftw_threads(fft-ct-fftw-thrcol fft-ct.c "-DUSE_FFTW_THRCOL")
  add_exec_fftw_threads(fft-ct-fftw-thrrow-blocked fft-ct.c
                        "-DUSE_FFTW_THRROW_BLOCKED")
  add_exec_fftw_threads(fft-ct-fftw-thrcol-blocked fft-ct.c
                        "-DUSE_FFTW_THRCOL_BLOCKED")
endif(FFTW_FOUND AND Threads_FOUND)

# Use MKL library
if(MKL_FOUND)
  function(add_exec_mkl name main definitions)
    add_executable(${name} ${main} ptime.c transpose-mkl.c util.c util-mkl.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_compile_definitions(${name} PRIVATE "MKL_Complex8=float _Complex")
    target_compile_definitions(${name} PRIVATE "MKL_Complex16=double _Complex")
    target_compile_options(${name} PRIVATE ${MKL_CFLAGS} ${MKL_CFLAGS_OTHER})
    target_link_libraries(${name} ${MKL_LDFLAGS} ${LIBRT})
    install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})
  endfunction(add_exec_mkl)

  add_exec_mkl(transp-flt-lib-lmkl transp.c "-DUSE_FLT_MKL")
  add_exec_mkl(transp-dbl-lib-lmkl transp.c "-DUSE_DBL_MKL")
  add_exec_mkl(transp-cmplx8-lib-lmkl transp.c "-DUSE_CMPLX8_MKL")
  add_exec_mkl(transp-cmplx16-lib-lmkl transp.c "-DUSE_CMPLX16_MKL")
endif(MKL_FOUND)

# Use MKL library implementations of the FFTWF interface
if(MKL_FOUND)
  function(add_exec_mkl_fftwf name main definitions)
    add_executable(${name} ${main} ptime.c transpose-mkl.c transpose-fftwf-mkl.c
                                   util.c util-fftwf.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_compile_definitions(${name} PRIVATE "MKL_Complex8=float _Complex")
    target_compile_definitions(${name} PRIVATE "MKL_Complex16=double _Complex")
    target_compile_options(${name} PRIVATE ${MKL_CFLAGS} ${MKL_CFLAGS_OTHER})
    target_link_libraries(${name} ${MKL_LDFLAGS} ${LIBRT})
    install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})
  endfunction(add_exec_mkl_fftwf)

  add_exec_mkl_fftwf(transp-fftwf-lib-lmkl transp.c "-DUSE_FFTWF_MKL")
  add_exec_mkl_fftwf(fft-ct-fftwf-lib-lmkl fft-ct.c "-DUSE_FFTWF_MKL")
  add_exec_mkl_fftwf(fft-2d-fftwf-lib-lmkl fft-2d.c "-DUSE_FFTWF")
endif(MKL_FOUND)

# Use MKL library implementations of the FFTW interface
if(MKL_FOUND)
  function(add_exec_mkl_fftw name main definitions)
    add_executable(${name} ${main} ptime.c transpose-mkl.c transpose-fftw-mkl.c
                                   util.c util-fftw.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_compile_definitions(${name} PRIVATE "MKL_Complex8=float _Complex")
    target_compile_definitions(${name} PRIVATE "MKL_Complex16=double _Complex")
    target_compile_options(${name} PRIVATE ${MKL_CFLAGS} ${MKL_CFLAGS_OTHER})
    target_link_libraries(${name} ${MKL_LDFLAGS} ${LIBRT})
    install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})
  endfunction(add_exec_mkl_fftw)

  add_exec_mkl_fftw(transp-fftw-lib-lmkl transp.c "-DUSE_FFTW_MKL")
  add_exec_mkl_fftw(fft-ct-fftw-lib-lmkl fft-ct.c "-DUSE_FFTW_MKL")
  add_exec_mkl_fftw(fft-2d-fftw-lib-lmkl fft-2d.c "")
endif(MKL_FOUND)

# A primitive approach for setting user-specified or default AVX compile flags
# Note: Complete auto-detection for architectures and compilers would require
#       checking target CPUIDs; the list would also get outdated with new CPUs.
#       Maybe another project maintains such a CMake capability we can use.
set(C_FLAGS_AVX CACHE STRING "C compiler AVX flags")
if(C_FLAGS_AVX)
  message("-- Checking for C_FLAGS_AVX - found")
  set(ENABLE_AVX ON)
else(C_FLAGS_AVX)
  message("-- Checking for C_FLAGS_AVX - not found")
  message("--   Defaulting to AVX-512 for Skylake...")
  # TODO: Does specifying optimization level conflict with CMAKE_BUILD_TYPE?
  #       Certain per-compiler levels are needed for vectorization.
  if("${CMAKE_C_COMPILER_ID}" MATCHES "GNU")
    set(CMAKE_C_COMPILER_VERSION_MIN 5.0)
    set(C_FLAGS_AVX "-march=skylake-avx512 -mtune=skylake-avx512 -ffast-math -O3")
    if (CFLAGS_AVX_VERBOSE)
      set(C_FLAGS_AVX "${C_FLAGS_AVX} -fopt-info-vec-all")
    endif (CFLAGS_AVX_VERBOSE)
  elseif("${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
    set(CMAKE_C_COMPILER_VERSION_MIN 3.9.1)
    set(C_FLAGS_AVX "-march=skylake-avx512 -mtune=skylake-avx512 -ffast-math -O3")
    # Note: regardless of whether we set C_FLAGS_AVX as a string or list, cmake
    # turns params of form `-Rpass="loop|vect"` into `"-Rpass=\"loop|vect\""`,
    # and clang seems to then ignore the param, so we don't get verbose output.
    # For now, just set `loop` and `vect` options separately.
    if (CFLAGS_AVX_VERBOSE)
      set(C_FLAGS_AVX "${C_FLAGS_AVX} -Rpass=loop -Rpass-missed=loop -Rpass-analysis=loop -Rpass=vect -Rpass-missed=vect -Rpass-analysis=vect")
    endif (CFLAGS_AVX_VERBOSE)
  elseif("${CMAKE_C_COMPILER_ID}" MATCHES "Intel")
    set(CMAKE_C_COMPILER_VERSION_MIN 15.0.1)
    set(C_FLAGS_AVX "-xCORE-AVX512 -qopt-zmm-usage=high -O3 -vec-threshold0")
    if (CFLAGS_AVX_VERBOSE)
      set(C_FLAGS_AVX "${C_FLAGS_AVX} -qopt-report=5 -qopt-report-file=stdout")
    endif (CFLAGS_AVX_VERBOSE)
  else()
    message(WARNING "No default AVX-512 flags for C compiler: "
                    "${CMAKE_C_COMPILER_ID}\n"
                    "Set C_FLAGS_AVX explicitly to force build with AVX-512")
  endif()
  if(CMAKE_C_COMPILER_VERSION_MIN)
    if(CMAKE_C_COMPILER_VERSION VERSION_LESS ${CMAKE_C_COMPILER_VERSION_MIN})
      message(WARNING
              "${CMAKE_C_COMPILER_ID} C compiler version is too old for AVX-512: "
              "${CMAKE_C_COMPILER_VERSION} < ${CMAKE_C_COMPILER_VERSION_MIN}")
    else(CMAKE_C_COMPILER_VERSION_MIN)
      set(ENABLE_AVX ON)
    endif(CMAKE_C_COMPILER_VERSION VERSION_LESS ${CMAKE_C_COMPILER_VERSION_MIN})
  endif(CMAKE_C_COMPILER_VERSION_MIN)
endif(C_FLAGS_AVX)
string(REPLACE " " ";" C_FLAGS_AVX_LIST "${C_FLAGS_AVX}") # string->list
if(ENABLE_AVX)
  message("--   C_FLAGS_AVX: ${C_FLAGS_AVX}")
endif(ENABLE_AVX)

# Use intrinsic AVX
if(ENABLE_AVX)
  function(add_exec_avx_intr name main definitions)
    add_executable(${name} ${main} ptime.c transpose-avx.c util.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_compile_options(${name} PRIVATE ${C_FLAGS_AVX_LIST})
    target_link_libraries(${name} ${LIBRT} ${LIBM})
    install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})
  endfunction(add_exec_avx_intr)

  add_exec_avx_intr(transp-dbl-avx512-intr transp.c "-DUSE_DBL_AVX512_INTR")
  add_exec_avx_intr(transp-dbl-avx512-intr-ss transp.c
                    "-DUSE_DBL_AVX512_INTR;-DUSE_AVX_STREAMING_STORES")
endif(ENABLE_AVX)

# Use threads with intrinsic AVX
if(Threads_FOUND AND ENABLE_AVX)
  function(add_exec_threads_avx name main definitions)
    add_executable(${name} ${main} ptime.c transpose-threads-avx.c util.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_compile_options(${name} PRIVATE ${C_FLAGS_AVX_LIST})
    target_link_libraries(${name} ${CMAKE_THREAD_LIBS_INIT} ${LIBRT} ${LIBM})
    install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})
  endfunction(add_exec_threads_avx)

  add_exec_threads_avx(transp-dbl-thrrow-avx512-intr transp.c
                       "-DUSE_DBL_THRROW_AVX512_INTR")
  add_exec_threads_avx(transp-dbl-thrrow-avx512-intr-ss transp.c
                       "-DUSE_DBL_THRROW_AVX512_INTR;-DUSE_AVX_STREAMING_STORES")
  add_exec_threads_avx(transp-dbl-thrcol-avx512-intr transp.c
                       "-DUSE_DBL_THRCOL_AVX512_INTR")
  add_exec_threads_avx(transp-dbl-thrcol-avx512-intr-ss transp.c
                       "-DUSE_DBL_THRCOL_AVX512_INTR;-DUSE_AVX_STREAMING_STORES")
endif(Threads_FOUND AND ENABLE_AVX)

# Use FFTWF library with intrinsic AVX
if(FFTWF_FOUND AND ENABLE_AVX)
  function(add_exec_fftwf_avx name main definitions)
    add_executable(${name} ${main} ptime.c transpose-fftwf-avx.c transpose-avx.c
                                   util.c util-fftwf.c)
    target_compile_options(${name} PRIVATE ${FFTWF_CFLAGS}
                                           ${FFTWF_CFLAGS_OTHER}
                                           ${C_FLAGS_AVX_LIST})
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${FFTWF_STATIC_LIBRARIES}
                                  ${LIBRT} ${LIBM})
    install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})
  endfunction(add_exec_fftwf_avx)

  add_exec_fftwf_avx(transp-fftwf-avx512-intr transp.c
                     "-DUSE_FFTWF_AVX512_INTR")
  add_exec_fftwf_avx(transp-fftwf-avx512-intr-ss transp.c
                     "-DUSE_FFTWF_AVX512_INTR;-DUSE_AVX_STREAMING_STORES")

  add_exec_fftwf_avx(fft-ct-fftwf-avx512-intr fft-ct.c
                     "-DUSE_FFTWF_AVX512_INTR")
  add_exec_fftwf_avx(fft-ct-fftwf-avx512-intr-ss fft-ct.c
                     "-DUSE_FFTWF_AVX512_INTR;-DUSE_AVX_STREAMING_STORES")
endif(FFTWF_FOUND AND ENABLE_AVX)

# Use FFTWF library with threads and intrinsic AVX
if(FFTWF_FOUND AND Threads_FOUND AND ENABLE_AVX)
  function(add_exec_fftwf_threads_avx name main definitions)
    add_executable(${name} ${main} ptime.c fft-threads-fftwf.c
                                   transpose-fftwf-threads-avx.c
                                   transpose-threads-avx.c
                                   util.c util-fftwf.c)
    target_compile_options(${name} PRIVATE ${FFTWF_CFLAGS}
                                           ${FFTWF_CFLAGS_OTHER}
                                           ${C_FLAGS_AVX_LIST})
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${FFTWF_STATIC_LIBRARIES}
                                  ${CMAKE_THREAD_LIBS_INIT}
                                  ${LIBRT} ${LIBM})
    install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})
  endfunction(add_exec_fftwf_threads_avx)

  add_exec_fftwf_threads_avx(transp-fftwf-thrrow-avx512-intr transp.c
                             "-DUSE_FFTWF_THRROW_AVX512_INTR")
  add_exec_fftwf_threads_avx(transp-fftwf-thrrow-avx512-intr-ss transp.c
                             "-DUSE_FFTWF_THRROW_AVX512_INTR;-DUSE_AVX_STREAMING_STORES")
  add_exec_fftwf_threads_avx(transp-fftwf-thrcol-avx512-intr transp.c
                             "-DUSE_FFTWF_THRCOL_AVX512_INTR")
  add_exec_fftwf_threads_avx(transp-fftwf-thrcol-avx512-intr-ss transp.c
                             "-DUSE_FFTWF_THRCOL_AVX512_INTR;-DUSE_AVX_STREAMING_STORES")

  add_exec_fftwf_threads_avx(fft-ct-fftwf-thrrow-avx512-intr fft-ct.c
                             "-DUSE_FFTWF_THRROW_AVX512_INTR")
  add_exec_fftwf_threads_avx(fft-ct-fftwf-thrrow-avx512-intr-ss fft-ct.c
                             "-DUSE_FFTWF_THRROW_AVX512_INTR;-DUSE_AVX_STREAMING_STORES")
  add_exec_fftwf_threads_avx(fft-ct-fftwf-thrcol-avx512-intr fft-ct.c
                             "-DUSE_FFTWF_THRCOL_AVX512_INTR")
  add_exec_fftwf_threads_avx(fft-ct-fftwf-thrcol-avx512-intr-ss fft-ct.c
                             "-DUSE_FFTWF_THRCOL_AVX512_INTR;-DUSE_AVX_STREAMING_STORES")
endif(FFTWF_FOUND AND Threads_FOUND AND ENABLE_AVX)


# Uninstall

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
  @ONLY
)
add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
