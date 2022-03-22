# CMake entry point
cmake_minimum_required (VERSION 3.0)

if (_APPLE_)
    project(ARCHI LANGUAGES CX X C)

    SET(CMAKE_CXX_FLAGS "-O3 -g -std=c++11  -Wno-c++11-narrowing ${CMAKE_CXX_FLAGS}")

    #find_package(OpenGL REQUIRED)
    #find_package(GLUT REQUIRED)
    #find_package(OpenCV REQUIRED)
    find_package(Eigen3 REQUIRED)

    if( CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR )
        message( FATAL_ERROR "Please select another Build Directory ! (and give it a clever name, like bin_Visual2012_64bits/)" )
    endif()
    if( CMAKE_SOURCE_DIR MATCHES " " )
        message( "Your Source Directory contains spaces. If you experience problems when compiling, this can be the cause." )
    endif()
    if( CMAKE_BINARY_DIR MATCHES " " )
        message( "Your Build Directory contains spaces. If you experience problems when compiling, this can be the cause." )
    endif()

    include_directories(
        ${EIGEN3_INCLUDE_DIR}
        /usr/include/
        .
    )

    add_definitions(
        -DTW_STATIC
        -DTW_NO_LIB_PRAGMA
        -DTW_NO_DIRECT3D
        -DGLEW_STATIC
        -D_CRT_SECURE_NO_WARNINGS
        -DGL_SILENCE_DEPRECATION
        -D__gl3_h_
        -DGL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED
        -D_APPLE_
    )

    add_executable(archi
        src/main.cpp
        src/Utilities.cpp
        src/CubicMesh.cpp
        include/Utilities.h
        include/MarchingCubes.h
        include/LevelSet.h
        include/MeshUtils.h
        include/CubicMesh.h
    )

elseif (_CUDA_)
    
    project(ARCHI LANGUAGES CXX C CUDA)

    enable_language(CUDA)

    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/bin/nvcc")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
    #add_compile_options(-std=c++11)

    find_package(CUDA 10.0 REQUIRED)
    SET(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/eigen-3.3.9")

    if( CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR )
        message( FATAL_ERROR "Please select another Build Directory ! (and give it a clever name, like bin_Visual2012_64bits/)" )
    endif()
    if( CMAKE_SOURCE_DIR MATCHES " " )
        message( "Your Source Directory contains spaces. If you experience problems when compiling, this can be the cause." )
    endif()
    if( CMAKE_BINARY_DIR MATCHES " " )
        message( "Your Build Directory contains spaces. If you experience problems when compiling, this can be the cause." )
    endif()

    # On Visual 2005 and above, this module can set the debug working directory
    cmake_policy(SET CMP0026 OLD)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external/rpavlik-cmake-modules-fe2273")
    include(CreateLaunchers)
    include(MSVCMultipleProcessCompile) # /MP

    include_directories(
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include"
        ${EIGEN3_INCLUDE_DIR}
        .
    )

    link_directories(
    )

    add_definitions(
        -DTW_STATIC
        -DTW_NO_LIB_PRAGMA
        -DTW_NO_DIRECT3D
        -DGLEW_STATIC
        -D_CRT_SECURE_NO_WARNINGS
        -DGL_SILENCE_DEPRECATION
        -D__gl3_h_
        -DGL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED
        -D_CUDA_
    )

    cuda_add_executable(archi
        src_gpu/main.cpp
        src_gpu/Utilities.cpp
        src_gpu/CubicMesh.cpp
        src_gpu/LevelSet_gpu.cu
        include_gpu/Utilities.h
        include_gpu/MarchingCubes.h
        include_gpu/LevelSet.h
        include_gpu/MeshUtils.h
        include_gpu/CubicMesh.h
        include_gpu/cudaTypes.cuh
        include_gpu/cutils_math.h
        include_gpu/cutils_matrix.h
        include_gpu/helper_cuda.h
        include_gpu/LevelSet_gpu.cuh
    )


    target_link_libraries(archi
 	${CUDA_LIBRARIES}
        cusparse.lib 
        cusolver.lib
        cublas.lib
    )

    # Xcode and Visual working directories
    set_target_properties(archi PROPERTIES XCODE_ATTRIBUTE_CONFIGURATION_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/")
    create_target_launcher(archi WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/")

    set_property(TARGET archi PROPERTY CUDA_ARCHITECTURES 35 50 72)

else()
    project(DEEPANIM LANGUAGES CXX C)

    #SET(CMAKE_CXX_FLAGS "-O3 -g -std=c++11 ${CMAKE_CXX_FLAGS}")

    SET(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/eigen-3.3.9")

    if( CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR )
        message( FATAL_ERROR "Please select another Build Directory ! (and give it a clever name, like bin_Visual2012_64bits/)" )
    endif()
    if( CMAKE_SOURCE_DIR MATCHES " " )
        message( "Your Source Directory contains spaces. If you experience problems when compiling, this can be the cause." )
    endif()
    if( CMAKE_BINARY_DIR MATCHES " " )
        message( "Your Build Directory contains spaces. If you experience problems when compiling, this can be the cause." )
    endif()

    # On Visual 2005 and above, this module can set the debug working directory
    cmake_policy(SET CMP0026 OLD)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external/rpavlik-cmake-modules-fe2273")
    include(CreateLaunchers)
    include(MSVCMultipleProcessCompile) # /MP

    include_directories(
        ${EIGEN3_INCLUDE_DIR}
        .
    )
    
    link_directories(
    )

    add_definitions(
        -DTW_STATIC
        -DTW_NO_LIB_PRAGMA
        -DTW_NO_DIRECT3D
        -DGLEW_STATIC
        -D_CRT_SECURE_NO_WARNINGS
        -DGL_SILENCE_DEPRECATION
        -D__gl3_h_
        -DGL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED
    )

    add_executable(deepanim
        src/main.cpp
        src/Utilities.cpp
        src/CubicMesh.cpp
        include/Utilities.h
        include/MarchingCubes.h
        #include/MarchingTets.h
        #include/Fit.h
        include/LevelSet.h
        include/MeshUtils.h
        #include/TetraMesh.h
        include/CubicMesh.h
    )

    target_link_libraries(deepanim
    )

    # Xcode and Visual working directories
    set_target_properties(deepanim PROPERTIES XCODE_ATTRIBUTE_CONFIGURATION_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/")
    create_target_launcher(deepanim WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/")

endif()



