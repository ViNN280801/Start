set(CUDA_GEOMETRY_SOURCES
    AABBTreeDevice.cu
    GeometryDeviceMemoryConverter.cu
    GeometryIntersectionChecker.cu
)

list(APPEND GEOMETRY_SOURCES ${CUDA_GEOMETRY_SOURCES})
