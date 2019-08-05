cmake_minimum_required(VERSION 2.8.5)

project(fft-ct)
set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)
set(VERSION_PATCH 0)
set(PROJECT_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -std=c99")


# Dependencies

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

add_executable(test-transpose test-transpose.c transpose.c util.c)
target_compile_definitions(test-transpose PRIVATE "-DUSE_PRIMITIVE")

add_executable(transp-float-naive transp.c transpose.c util.c)
target_compile_definitions(transp-float-naive PRIVATE "-DUSE_FLOAT_NAIVE")

add_executable(transp-double-naive transp.c transpose.c util.c)
target_compile_definitions(transp-double-naive PRIVATE "-DUSE_DOUBLE_NAIVE")

# Use FFTW library
if(FFTW_FOUND)
  add_executable(test-transpose-fftw test-transpose.c
                                     transpose-fftw.c
                                     util.c util-fftw.c)
  target_compile_definitions(test-transpose-fftw PRIVATE "-DUSE_FFTW")
  target_link_libraries(test-transpose-fftw ${FFTW_LDFLAGS}
                                            ${FFTW_STATIC_LIBRARIES})

  add_executable(fft-ct fft-ct.c
                        transpose-fftw.c
                        util.c util-fftw.c)
  target_link_libraries(fft-ct ${FFTW_LDFLAGS} ${FFTW_STATIC_LIBRARIES})

  add_executable(fft-2d fft-2d.c
                        transpose-fftw.c
                        util.c util-fftw.c)
  target_link_libraries(fft-2d ${FFTW_LDFLAGS} ${FFTW_STATIC_LIBRARIES})

  add_executable(transp-fftw-naive transp.c
                        transpose-fftw.c
                        util.c util-fftw.c)
  target_compile_definitions(transp-fftw-naive PRIVATE "-DUSE_FFTW_NAIVE")
  target_link_libraries(transp-fftw-naive ${FFTW_LDFLAGS}
                                          ${FFTW_STATIC_LIBRARIES})
endif()

# Use MKL library
if(MKL_FOUND)
  add_executable(test-transpose-mkl test-transpose.c
                                    transpose-mkl.c
                                    util.c util-mkl.c)
  target_compile_definitions(test-transpose-mkl PRIVATE "-DUSE_MKL")
  target_compile_options(test-transpose-mkl PRIVATE ${MKL_CFLAGS}
                                                    ${MKL_CFLAGS_OTHER})
  target_link_libraries(test-transpose-mkl ${MKL_LDFLAGS})
endif()

# Use MKL library implementations of the FFTW interface
if(MKL_FOUND)
  add_executable(test-transpose-mkl-fftw test-transpose.c
                                         transpose-fftw.c
                                         util.c util-fftw.c)
  target_compile_definitions(test-transpose-mkl-fftw PRIVATE "-DUSE_FFTW")
  target_compile_options(test-transpose-mkl-fftw PRIVATE ${MKL_CFLAGS}
                                                         ${MKL_CFLAGS_OTHER})
  target_link_libraries(test-transpose-mkl-fftw ${MKL_LDFLAGS})

  add_executable(fft-ct-mkl-fftw fft-ct.c
                                 transpose-fftw.c
                                 util.c util-fftw.c)
  target_link_libraries(fft-ct-mkl-fftw ${MKL_LDFLAGS})

  add_executable(fft-2d-mkl-fftw fft-2d.c
                                 transpose-fftw.c
                                 util.c util-fftw.c)
  target_link_libraries(fft-2d-mkl-fftw ${MKL_LDFLAGS})

  add_executable(transp-mkl-fftw-naive transp.c
                                 transpose-fftw.c
                                 util.c util-fftw.c)
  target_compile_definitions(transp-mkl-fftw-naive PRIVATE "-DUSE_FFTW_NAIVE")
  target_link_libraries(transp-mkl-fftw-naive ${MKL_LDFLAGS})
endif()
