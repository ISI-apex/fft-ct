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

  add_executable(transp transp.c
                        transpose-fftw.c
                        util.c util-fftw.c)
  target_link_libraries(transp ${FFTW_LDFLAGS} ${FFTW_STATIC_LIBRARIES})
endif()

if(MKL_FOUND)
  add_executable(test-transpose-mkl test-transpose.c
                                    transpose-mkl.c
                                    util.c util-mkl.c)
  target_compile_definitions(test-transpose-mkl PRIVATE "-DUSE_MKL")
  target_compile_options(test-transpose-mkl PRIVATE ${MKL_CFLAGS}
                                                    ${MKL_CFLAGS_OTHER})
  target_link_libraries(test-transpose-mkl ${MKL_LDFLAGS})
endif()
