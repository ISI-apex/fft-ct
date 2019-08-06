cmake_minimum_required(VERSION 2.8.12)

project(fft-ct)
set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)
set(VERSION_PATCH 0)
set(PROJECT_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -std=c99")


# Dependencies

find_package(Threads REQUIRED)

find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(FFTW fftw3)
  if(FFTW_FOUND)
    include_directories(${FFTW_INCLUDE_DIRS})
  endif()

  pkg_check_modules(MKL mkl-static-ilp64-seq)
  if(MKL_FOUND)
    include_directories(${MKL_INCLUDE_DIRS})
  endif()
endif()


# Binaries

# Name format: ${prog}-${datatype}-{impl}
# datatype is probably one of:
# [float, double, fftw_complex, MKL_Complex8, MKL_Complex16]
# impl is probably one or more of:
# [naive, blocked, mkl, mkl-fftw, pthread]

function(add_exec_test_transp name definitions)
  add_executable(${name} test-transpose.c transpose.c util.c)
  target_compile_definitions(${name} PRIVATE ${definitions})
endfunction(add_exec_test_transp)

add_exec_test_transp(test-transp-float-naive "-DUSE_FLOAT_NAIVE")
add_exec_test_transp(test-transp-double-naive "-DUSE_DOUBLE_NAIVE")
add_exec_test_transp(test-transp-float-blocked "-DUSE_FLOAT_BLOCKED")
add_exec_test_transp(test-transp-double-blocked "-DUSE_DOUBLE_BLOCKED")

function(add_exec_transp name definitions)
  add_executable(${name} transp.c transpose.c util.c)
  target_compile_definitions(${name} PRIVATE ${definitions})
endfunction(add_exec_transp)

add_exec_transp(transp-float-naive "-DUSE_FLOAT_NAIVE")
add_exec_transp(transp-double-naive "-DUSE_DOUBLE_NAIVE")
add_exec_transp(transp-float-blocked "-DUSE_FLOAT_BLOCKED")
add_exec_transp(transp-double-blocked "-DUSE_DOUBLE_BLOCKED")

# Use FFTW library
if(FFTW_FOUND)
  function(add_exec_test_transp_fftw name definitions)
    add_executable(${name} test-transpose.c transpose-fftw.c util.c util-fftw.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${FFTW_LDFLAGS} ${FFTW_STATIC_LIBRARIES})
  endfunction(add_exec_test_transp_fftw)

  add_exec_test_transp_fftw(test-transp-fftw_complex-naive "-DUSE_FFTW_NAIVE")

  function(add_exec_transp_fftw name definitions)
    add_executable(${name} transp.c transpose-fftw.c util.c util-fftw.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${FFTW_LDFLAGS} ${FFTW_STATIC_LIBRARIES})
  endfunction(add_exec_transp_fftw)

  add_exec_transp_fftw(transp-fftw_complex-naive "-DUSE_FFTW_NAIVE")

  function(add_exec_fft_fftw name main)
    add_executable(${name} ${main} transpose-fftw.c util.c util-fftw.c)
    target_link_libraries(${name} ${FFTW_LDFLAGS} ${FFTW_STATIC_LIBRARIES})
  endfunction(add_exec_fft_fftw)

  add_exec_fft_fftw(fft-ct-fftw_complex-naive fft-ct.c)
  add_exec_fft_fftw(fft-2d-fftw_complex-fftw fft-2d.c)
endif()

# Use MKL library
if(MKL_FOUND)
  function(add_exec_test_transp_mkl name definitions)
    add_executable(${name} test-transpose.c transpose-mkl.c util.c util-mkl.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_compile_options(${name} PRIVATE ${MKL_CFLAGS} ${MKL_CFLAGS_OTHER})
    target_link_libraries(${name} ${MKL_LDFLAGS})
  endfunction(add_exec_test_transp_mkl)

  function(add_exec_transp_mkl name definitions)
    add_executable(${name} transp.c transpose-mkl.c util.c util-mkl.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_compile_options(${name} PRIVATE ${MKL_CFLAGS} ${MKL_CFLAGS_OTHER})
    target_link_libraries(${name} ${MKL_LDFLAGS})
  endfunction(add_exec_transp_mkl)

  add_exec_test_transp_mkl(test-transp-float-mkl "-DUSE_MKL_FLOAT")
  add_exec_test_transp_mkl(test-transp-double-mkl "-DUSE_MKL_DOUBLE")
  add_exec_test_transp_mkl(test-transp-cmplx8-mkl "-DUSE_MKL_CMPLX8")
  add_exec_test_transp_mkl(test-transp-cmplx16-mkl "-DUSE_MKL_CMPLX16")

  add_exec_transp_mkl(transp-float-mkl "-DUSE_MKL_FLOAT")
  add_exec_transp_mkl(transp-double-mkl "-DUSE_MKL_DOUBLE")
  add_exec_transp_mkl(transp-cmplx8-mkl "-DUSE_MKL_CMPLX8")
  add_exec_transp_mkl(transp-cmplx16-mkl "-DUSE_MKL_CMPLX16")
endif()

# Use MKL library implementations of the FFTW interface
if(MKL_FOUND)
  function(add_exec_transp_mkl_fftw name definitions)
    add_executable(${name} transp.c transpose-fftw.c util.c util-fftw.c)
    target_compile_definitions(${name} PRIVATE ${definitions})
    target_link_libraries(${name} ${MKL_LDFLAGS})
  endfunction(add_exec_transp_mkl_fftw)

  add_exec_transp_mkl_fftw(transp-fftw_complex-naive-mkl-fftw "-DUSE_FFTW_NAIVE")

  function(add_exec_fft_mkl_fftw name main)
    add_executable(${name} ${main} transpose-fftw.c util.c util-fftw.c)
    target_link_libraries(${name} ${MKL_LDFLAGS})
  endfunction(add_exec_fft_mkl_fftw)

  add_exec_fft_mkl_fftw(fft-ct-fftw_complex-naive-mkl-fftw fft-ct.c)
  add_exec_fft_mkl_fftw(fft-2d-fftw_complex-mkl-fftw fft-2d.c)
endif()
