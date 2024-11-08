cmake_minimum_required(VERSION 3.2)

project(CSL LANGUAGES C CXX Fortran VERSION 0.0.10)
    if(DEFINED ENV{FC} AND "$ENV{FC}" MATCHES "^.*ifort$")
        set(FC $ENV{FC} CACHE STRING "Fortran compiler")
        message("FC set to ${FC}")
    else()
        set(FC "gfortran" CACHE STRING "Fortran compiler")
        message("Defaulting to gfortran")
    endif()

    set(QUIET_MAKE "--no-print-directory")

    add_custom_target(csl ALL)

    # Adding customized cmake module
    list(APPEND CMAKE_MODULE_PATH  "${PROJECT_SOURCE_DIR}/cmake/Modules/")

    #if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    #    message(FATAL_ERROR "Require at least gcc-5.0")
    #elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.0)
    #    message(FATAL_ERROR "Require at least clang 10 (this is an arbitrary number higher than 9.1, which doesn't work).")
    #endif()

    find_package(CosmoSIS REQUIRED)

    set(COSMOSIS_SRC_DIR ${COSMOSIS_DIR})

    include(BuildGSL)
    include(BuildFFTW)
    include(BuildCFITSIO)

    find_package(GFortranLibs)
    set(GFORTRAN_LIBPATH ${GFORTRAN_LIBRARIES_DIR})
    
    if(APPLE)
        set(LAPACK_LINK "-framework Accelerate")
    else()
        if(FC MATCHES "^.*ifort$")
            set(BLA_VENDOR Intel10_64lp)
        endif()
	find_package(LAPACK REQUIRED)
	list(GET LAPACK_LIBRARIES 0 LAPACK_LIB)
	get_filename_component(LAPACK_LIB_DIR ${LAPACK_LIB} DIRECTORY)
        set(LAPACK_LINK "-L${LAPACK_LIB_DIR} -llapack -lblas")
	message(STATUS "LAPACK_LINK: ${LAPACK_LIB} ${LAPACK_LIB_DIR}")
    endif()

    add_subdirectory(boltzmann/camb)
    add_subdirectory(boltzmann/halofit)
    add_subdirectory(boltzmann/halofit_takahashi)

    add_subdirectory(shear/cl_to_xi_nicaea)
    
    add_subdirectory(structure/cosmic_emu)
    add_subdirectory(structure/meadcb)
    add_subdirectory(structure/projection)

    add_subdirectory(likelihood/planck2018)

    add_subdirectory(supernovae/jla_v3)

    add_dependencies(csl camb halofit_takahashi halofit cosmic_emu meadcb cl_to_xi_nicaea projection jla planck2018)

    add_custom_target(csl-clean)
    add_dependencies(csl-clean camb-clean halofit_takahashi-clean halofit-clean cosmic_emu-clean meadcb-clean cl_to_xi_nicaea-clean projection-clean jla-clean planck2018-clean)

    
