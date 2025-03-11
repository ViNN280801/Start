# src/Geometry/CMakeLists.txt

set(GEOMETRY_SOURCES
    CubicGrid.cpp
    GeometryTypes.cpp
    Mesh.cpp
    RayTriangleIntersection.cpp
)

if(GPU_CUDA_ON)
    set(CUDA_GEOMETRY_SOURCES
        CUDA/AABBTreeDevice.cu
        CUDA/GeometryDeviceMemoryConverter.cu
        CUDA/GeometryKernels.cu
    )
    
    # Set compilation flags for all CUDA files in the Geometry module
    set_source_files_properties(
        CUDA/AABBTreeDevice.cu
        CUDA/GeometryDeviceMemoryConverter.cu
        CUDA/GeometryKernels.cu
        PROPERTIES COMPILE_FLAGS "-rdc=true -Wno-deprecated-gpu-targets ${CUDA_ARCH_FLAGS}"
    )
endif()

set(LIB_GEOMETRY Geometry)
add_library(${LIB_GEOMETRY} ${GEOMETRY_SOURCES})

target_link_libraries(${LIB_GEOMETRY} PRIVATE
    ${GMSH_LIBRARIES}
)

target_include_directories(${LIB_GEOMETRY} PRIVATE
    ${CGAL_INCLUDE_DIRS}
    ${GMSH_INCLUDE_DIRS}
)

if(GPU_CUDA_ON)
    target_sources(${LIB_GEOMETRY} PRIVATE ${CUDA_GEOMETRY_SOURCES})
    set_target_properties(${LIB_GEOMETRY} PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(${LIB_GEOMETRY} PRIVATE cudadevrt)
endif()
