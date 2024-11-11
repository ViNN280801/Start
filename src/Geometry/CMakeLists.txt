set(GEOMETRY_SOURCES
    CubicGrid.cpp
    Mesh.cpp
    RayTriangleIntersection.cpp
)

set(LIB_GEOMETRY Geometry)
add_library(${LIB_GEOMETRY} ${GEOMETRY_SOURCES})

target_link_libraries(${LIB_GEOMETRY} PRIVATE
    ${GMSH_LIBRARIES}
)

target_include_directories(${LIB_GEOMETRY} PRIVATE
    ${CGAL_INCLUDE_DIRS}
    ${GMSH_INCLUDE_DIRS}
)
