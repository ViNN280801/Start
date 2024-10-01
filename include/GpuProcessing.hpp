#ifndef GPU_PROCESSING_HPP
#define GPU_PROCESSING_HPP

#include <memory>

#include "FiniteElementMethod/FEMTypes.hpp"
#include "FiniteElementMethod/GSMAssemblier.hpp"
#include "Geometry/CubicGrid.hpp"
#include "Particles/Particle.hpp"

void processParticleTrackerOnGPU(std::vector<Particle> &particles, size_t start_index, size_t end_index,
                                 std::shared_ptr<CubicGrid> cubicGrid,
                                 std::shared_ptr<GSMAssemblier> assemblier,
                                 std::map<GlobalOrdinal, double> &nodeChargeDensityMap, double time);

#endif // !GPU_PROCESSING_HPP
