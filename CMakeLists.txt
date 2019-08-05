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

add_executable(test-transp-float-naive test-transpose.c transpose.c util.c)
target_compile_definitions(test-transp-float-naive
                           PRIVATE "-DUSE_FLOAT_NAIVE")

add_executable(test-transp-double-naive test-transpose.c transpose.c util.c)
target_compile_definitions(test-transp-double-naive
                           PRIVATE "-DUSE_DOUBLE_NAIVE")

add_executable(test-transp-float-blocked test-transpose.c transpose.c util.c)
target_compile_definitions(test-transp-float-blocked
                           PRIVATE "-DUSE_FLOAT_BLOCKED")

add_executable(test-transp-double-blocked test-transpose.c transpose.c util.c)
target_compile_definitions(test-transp-double-blocked
                           PRIVATE "-DUSE_DOUBLE_BLOCKED")

add_executable(transp-float-naive transp.c transpose.c util.c)
target_compile_definitions(transp-float-naive PRIVATE "-DUSE_FLOAT_NAIVE")

add_executable(transp-double-naive transp.c transpose.c util.c)
target_compile_definitions(transp-double-naive PRIVATE "-DUSE_DOUBLE_NAIVE")

add_executable(transp-float-blocked transp.c transpose.c util.c)
target_compile_definitions(transp-float-blocked PRIVATE "-DUSE_FLOAT_BLOCKED")

add_executable(transp-double-blocked transp.c transpose.c util.c)
target_compile_definitions(transp-double-blocked PRIVATE "-DUSE_DOUBLE_BLOCKED")

# Use FFTW library
if(FFTW_FOUND)
  add_executable(test-transp-fftw_complex-naive test-transpose.c
                                                transpose-fftw.c
                                                util.c util-fftw.c)
  target_compile_definitions(test-transp-fftw_complex-naive
                             PRIVATE "-DUSE_FFTW_NAIVE")
  target_link_libraries(test-transp-fftw_complex-naive ${FFTW_LDFLAGS}
                                                       ${FFTW_STATIC_LIBRARIES})

  add_executable(transp-fftw_complex-naive transp.c
                                           transpose-fftw.c
                                           util.c util-fftw.c)
  target_compile_definitions(transp-fftw_complex-naive
                             PRIVATE "-DUSE_FFTW_NAIVE")
  target_link_libraries(transp-fftw_complex-naive ${FFTW_LDFLAGS}
                                                  ${FFTW_STATIC_LIBRARIES})

  add_executable(fft-ct-fftw_complex-naive fft-ct.c
                                           transpose-fftw.c
                                           util.c util-fftw.c)
  target_link_libraries(fft-ct-fftw_complex-naive ${FFTW_LDFLAGS}
                                                  ${FFTW_STATIC_LIBRARIES})

  add_executable(fft-2d-fftw_complex-fftw fft-2d.c
                                          transpose-fftw.c
                                          util.c util-fftw.c)
  target_link_libraries(fft-2d-fftw_complex-fftw ${FFTW_LDFLAGS}
                                                 ${FFTW_STATIC_LIBRARIES})
endif()

# Use MKL library
if(MKL_FOUND)
  add_executable(test-transp-float-mkl test-transpose.c
                                       transpose-mkl.c
                                       util.c util-mkl.c)
  target_compile_definitions(test-transp-float-mkl PRIVATE "-DUSE_MKL_FLOAT")
  target_compile_options(test-transp-float-mkl PRIVATE ${MKL_CFLAGS}
                                                       ${MKL_CFLAGS_OTHER})
  target_link_libraries(test-transp-float-mkl ${MKL_LDFLAGS})

  add_executable(test-transp-double-mkl test-transpose.c
                                        transpose-mkl.c
                                        util.c util-mkl.c)
  target_compile_definitions(test-transp-double-mkl PRIVATE "-DUSE_MKL_DOUBLE")
  target_compile_options(test-transp-double-mkl PRIVATE ${MKL_CFLAGS}
                                                        ${MKL_CFLAGS_OTHER})
  target_link_libraries(test-transp-double-mkl ${MKL_LDFLAGS})

  add_executable(test-transp-cmplx8-mkl test-transpose.c
                                        transpose-mkl.c
                                        util.c util-mkl.c)
  target_compile_definitions(test-transp-cmplx8-mkl PRIVATE "-DUSE_MKL_CMPLX8")
  target_compile_options(test-transp-cmplx8-mkl PRIVATE ${MKL_CFLAGS}
                                                        ${MKL_CFLAGS_OTHER})
  target_link_libraries(test-transp-cmplx8-mkl ${MKL_LDFLAGS})

  add_executable(test-transp-cmplx16-mkl test-transpose.c
                                         transpose-mkl.c
                                         util.c util-mkl.c)
  target_compile_definitions(test-transp-cmplx16-mkl PRIVATE "-DUSE_MKL_CMPLX16")
  target_compile_options(test-transp-cmplx16-mkl PRIVATE ${MKL_CFLAGS}
                                                         ${MKL_CFLAGS_OTHER})
  target_link_libraries(test-transp-cmplx16-mkl ${MKL_LDFLAGS})

  add_executable(transp-float-mkl transp.c
                                  transpose-mkl.c
                                  util.c util-mkl.c)
  target_compile_definitions(transp-float-mkl PRIVATE "-DUSE_MKL_FLOAT")
  target_compile_options(transp-float-mkl PRIVATE ${MKL_CFLAGS}
                                                  ${MKL_CFLAGS_OTHER})
  target_link_libraries(transp-float-mkl ${MKL_LDFLAGS})

  add_executable(transp-double-mkl transp.c
                                   transpose-mkl.c
                                   util.c util-mkl.c)
  target_compile_definitions(transp-double-mkl PRIVATE "-DUSE_MKL_DOUBLE")
  target_compile_options(transp-double-mkl PRIVATE ${MKL_CFLAGS}
                                                   ${MKL_CFLAGS_OTHER})
  target_link_libraries(transp-double-mkl ${MKL_LDFLAGS})

  add_executable(transp-cmplx8-mkl transp.c
                                   transpose-mkl.c
                                   util.c util-mkl.c)
  target_compile_definitions(transp-cmplx8-mkl PRIVATE "-DUSE_MKL_CMPLX8")
  target_compile_options(transp-cmplx8-mkl PRIVATE ${MKL_CFLAGS}
                                                   ${MKL_CFLAGS_OTHER})
  target_link_libraries(transp-cmplx8-mkl ${MKL_LDFLAGS})

  add_executable(transp-cmplx16-mkl transp.c
                                    transpose-mkl.c
                                    util.c util-mkl.c)
  target_compile_definitions(transp-cmplx16-mkl PRIVATE "-DUSE_MKL_CMPLX16")
  target_compile_options(transp-cmplx16-mkl PRIVATE ${MKL_CFLAGS}
                                                    ${MKL_CFLAGS_OTHER})
  target_link_libraries(transp-cmplx16-mkl ${MKL_LDFLAGS})
endif()

# Use MKL library implementations of the FFTW interface
if(MKL_FOUND)
  add_executable(transp-fftw_complex-naive-mkl-fftw transp.c
                                                    transpose-fftw.c
                                                    util.c util-fftw.c)
  target_compile_definitions(transp-fftw_complex-naive-mkl-fftw
                             PRIVATE "-DUSE_FFTW_NAIVE")
  target_link_libraries(transp-fftw_complex-naive-mkl-fftw ${MKL_LDFLAGS})

  add_executable(fft-ct-fftw_complex-naive-mkl-fftw fft-ct.c
                                                    transpose-fftw.c
                                                    util.c util-fftw.c)
  target_link_libraries(fft-ct-fftw_complex-naive-mkl-fftw ${MKL_LDFLAGS})

  add_executable(fft-2d-fftw_complex-mkl-fftw fft-2d.c
                                              transpose-fftw.c
                                              util.c util-fftw.c)
  target_link_libraries(fft-2d-fftw_complex-mkl-fftw ${MKL_LDFLAGS})
endif()
