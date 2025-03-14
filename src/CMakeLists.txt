# src/CMakeLists.txt

add_subdirectory(DataHandling)
add_subdirectory(FiniteElementMethod)
add_subdirectory(Generators)
add_subdirectory(Geometry)
add_subdirectory(Particle)
add_subdirectory(ParticleInCellEngine)
add_subdirectory(SessionManagement)
add_subdirectory(Utilities)

if (COMPILE_SPUTTERING_MODEL)
    add_subdirectory(SputteringModel)
endif()
