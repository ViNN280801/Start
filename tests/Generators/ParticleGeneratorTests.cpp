#include <gtest/gtest.h>
#include <iomanip>

#include "Generators/Host/ParticleGeneratorHost.hpp"
#include "Generators/Host/RealNumberGeneratorHost.hpp"
#include "Particle/ParticlePropertiesManager.hpp"
#include "Particle/PhysicsCore/CollisionModel/CollisionModelFactory.hpp"
#include "Particle/PhysicsCore/ParticleDynamicUtils.hpp"

using ParticleGenerator = ParticleGeneratorHost;

template <typename T>
void printMemoryUsage(size_t count, std::string_view description = "Memory usage")
{
    size_t totalBytes{sizeof(T) * count};
    double kb{static_cast<double>(totalBytes) / 1024},
        mb{kb / 1024},
        gb{mb / 1024};

    std::cout << description << " for " << count << " elements of type "
              << typeid(T).name() << ":\n"
              << "Bytes: " << totalBytes << " bytes\n"
              << "Kilobytes: " << std::fixed << std::setprecision(2) << kb << " Kb\n"
              << "Megabytes: " << mb << " Mb\n"
              << "Gigabytes: " << gb << " Gb\n";
}

class ParticleGeneratorTest : public ::testing::Test
{
protected:
    ParticleGeneratorTest() = default;
};

// === Clean Test for fromPointSource === //

TEST_F(ParticleGeneratorTest, CleanFromPointSource)
{
    std::vector<point_source_t> pointSources = {
        {"Ar", 50, 10.0, 1.0, 1.0, 2.0, {0.0, 0.0, 0.0}},
        {"He", 30, 20.0, 2.0, 2.0, 3.0, {10.0, 10.0, 10.0}}};

    ParticleVector particles = ParticleGenerator::fromPointSource(pointSources);

    size_t expectedParticleCount = 50 + 30;
    ASSERT_EQ(particles.size(), expectedParticleCount);
    for (Particle const &p : particles)
    {
        EXPECT_TRUE(p.getType() == ParticleType::Ar || p.getType() == ParticleType::He);
        EXPECT_GE(p.getEnergy_eV(), 10.0);
        EXPECT_LE(p.getEnergy_eV(), 20.0);

        EXPECT_GE(p.getEnergy_J(), 9.0 / 6.241506363094e+18);
        EXPECT_LE(p.getEnergy_J(), 21.0 / 6.241506363094e+18);
    }
}

// === Dirty Tests for fromPointSource === //

TEST_F(ParticleGeneratorTest, DirtyFromPointSourceNegativeEnergy)
{
    std::vector<point_source_t> pointSources = {
        {"Ne", 10, -10.0, 1.0, 1.0, 2.0, {0.0, 0.0, 0.0}} // Negative energy
    };

    ParticleVector particles = ParticleGenerator::fromPointSource(pointSources);
    ASSERT_EQ(particles.size(), 10);
    for (Particle const &p : particles)
    {
        EXPECT_LT(p.getEnergy_J(), 0.0);
    }
}

TEST_F(ParticleGeneratorTest, DirtyFromPointSourceInvalidParticleType)
{
    std::vector<point_source_t> pointSources = {
        {"InvalidType", 10, 10.0, 1.0, 1.0, 2.0, {0.0, 0.0, 0.0}} // Invalid particle type
    };

    EXPECT_THROW(ParticleGenerator::fromPointSource(pointSources), std::invalid_argument);
}

TEST_F(ParticleGeneratorTest, DirtyFromPointSourceZeroEnergy)
{
    std::vector<point_source_t> pointSources = {
        {"O2", 10, 0.0, 1.0, 1.0, 2.0, {0.0, 0.0, 0.0}} // Zero energy
    };

    ParticleVector particles = ParticleGenerator::fromPointSource(pointSources);
    ASSERT_EQ(particles.size(), 10);
    for (Particle const &p : particles)
    {
        EXPECT_DOUBLE_EQ(p.getEnergy_eV(), 0.0);
    }
}

// === Clean Test for fromSurfaceSource === //

TEST_F(ParticleGeneratorTest, CleanFromSurfaceSource)
{
    std::vector<surface_source_t> surfaceSources = {
        {"Ar", 50, 10.0, {{"0,0,0", {0.0, 0.0, 1.0}}}},
        {"Sn", 30, 20.0, {{"1,1,1", {0.0, 1.0, 0.0}}}}};

    ParticleVector particles = ParticleGenerator::fromSurfaceSource(surfaceSources);

    size_t expectedParticleCount = 50 + 30;
    ASSERT_EQ(particles.size(), expectedParticleCount);
    for (Particle const &p : particles)
    {
        EXPECT_TRUE(p.getType() == ParticleType::Ar || p.getType() == ParticleType::Sn);
        EXPECT_GE(p.getEnergy_eV(), 10.0);
        EXPECT_LE(p.getEnergy_eV(), 20.0);

        EXPECT_GE(p.getEnergy_J(), 9.0 / 6.241506363094e+18);
        EXPECT_LE(p.getEnergy_J(), 21.0 / 6.241506363094e+18);
    }
}

// === Dirty Tests for fromSurfaceSource === //

TEST_F(ParticleGeneratorTest, DirtyFromSurfaceSourceNegativeEnergy)
{
    std::vector<surface_source_t> surfaceSources = {
        {"Ne", 10, -10.0, {{"0,0,0", {0.0, 0.0, 1.0}}}} // Negative energy
    };

    ParticleVector particles = ParticleGenerator::fromSurfaceSource(surfaceSources);
    ASSERT_EQ(particles.size(), 10);
    for (Particle const &p : particles)
    {
        EXPECT_LT(p.getEnergy_J(), 0.0);
    }
}

TEST_F(ParticleGeneratorTest, DirtyFromSurfaceSourceInvalidParticleType)
{
    std::vector<surface_source_t> surfaceSources = {
        {"InvalidType", 10, 10.0, {{"0,0,0", {0.0, 0.0, 1.0}}}} // Invalid particle type
    };

    EXPECT_THROW(ParticleGenerator::fromSurfaceSource(surfaceSources), std::invalid_argument);
}

TEST_F(ParticleGeneratorTest, DirtyFromSurfaceSourceZeroEnergy)
{
    std::vector<surface_source_t> surfaceSources = {
        {"Ne", 10, 0.0, {{"0,0,0", {0.0, 0.0, 1.0}}}} // Zero energy
    };

    ParticleVector particles = ParticleGenerator::fromSurfaceSource(surfaceSources);
    ASSERT_EQ(particles.size(), 10);
    for (Particle const &p : particles)
    {
        EXPECT_DOUBLE_EQ(p.getEnergy_eV(), 0.0);
    }
}

TEST_F(ParticleGeneratorTest, DirtyFromPointSourceZeroParticles)
{
    std::vector<point_source_t> pointSources = {
        {"Au", 50, 10.0, 1.0, 1.0, 2.0, {0.0, 0.0, 0.0}},
        {"W", 0, 20.0, 2.0, 2.0, 3.0, {10.0, 10.0, 10.0}} // Zero particles
    };

    EXPECT_ANY_THROW(ParticleGenerator::fromPointSource(pointSources));
}

TEST_F(ParticleGeneratorTest, DirtyFromSurfaceSourceZeroParticles)
{
    std::vector<surface_source_t> surfaceSources = {
        {"Ag", 10000, 0.0, {{"0,0,0", {0.0, 0.0, 1.0}}}},
        {"He", 0, 0.0, {{"0,0,0", {0.0, 0.0, 1.0}}}} // Zero count of particles
    };

    EXPECT_ANY_THROW(ParticleGenerator::fromSurfaceSource(surfaceSources));
}

// === Stress testing === //

TEST_F(ParticleGeneratorTest, StressLargeNumberOfParticles)
{
    size_t particleCount{10'000'000ul};

    // Creating a point source with a large number of particles
    point_source_t pointSource = {
        .type = "Ar",
        .count = particleCount,
        .energy = 100.0,
        .phi = 0.0,
        .theta = 0.0,
        .expansionAngle = 0.0,
        .baseCoordinates = {0.0, 0.0, 0.0}};

    EXPECT_NO_THROW({
        ParticleVector particles = ParticleGenerator::fromPointSource({pointSource});
        ASSERT_EQ(particles.size(), particleCount);

        printMemoryUsage<Particle>(particleCount, "Memory usage for large particle count test");
    });
}

TEST_F(ParticleGeneratorTest, StressExtremeVelocityValues)
{
    size_t particleCount{1'000'000ul};

    // Creating a surface source with extreme velocity ranges
    surface_source_t surfaceSource = {
        .type = "O2",
        .count = particleCount,
        .energy = 1e6,
        .baseCoordinates = {
            {"-1e12, -1e12, -1e12", {0.0, 0.0, 1.0}},
            {"1e12, 1e12, 1e12", {0.0, 0.0, 1.0}}}};

    ParticleVector particles = ParticleGenerator::fromSurfaceSource({surfaceSource});
    ASSERT_EQ(particles.size(), particleCount);
    printMemoryUsage<Particle>(particleCount, "Memory usage for extreme velocity values");
}

TEST_F(ParticleGeneratorTest, StressParallelGeneration)
{
    size_t particleCount{5'000'000ul};

    // Creating a point source for parallel generation test
    point_source_t pointSource = {
        .type = "He",
        .count = particleCount,
        .energy = 50.0,
        .phi = 0.0,
        .theta = 0.0,
        .expansionAngle = 0.0,
        .baseCoordinates = {0.0, 0.0, 0.0}};

    auto start = std::chrono::high_resolution_clock::now();

    ParticleVector particles = ParticleGenerator::fromPointSource({pointSource});

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Spent time for " << particleCount << " particles: " << duration.count() << " ms\n";

    ASSERT_EQ(particles.size(), particleCount);
    EXPECT_LT(duration.count(), 5000);

    printMemoryUsage<Particle>(particleCount, "Memory usage for parallel generation test");
}

// === Integration testing === //

TEST_F(ParticleGeneratorTest, ParticleEachMethodTesting)
{
    size_t particleCount{500'000ul};

    // Create a surface source for particle generation
    surface_source_t surfaceSource = {
        .type = "Au",
        .count = particleCount,
        .energy = 1.0, // Example energy in eV
        .baseCoordinates = {
            {"0.0, 0.0, 0.0", {0.0, 0.0, 1.0}},
            {"100.0, 0.0, 0.0", {0.0, 0.0, 1.0}},
            {"0.0, 100.0, 0.0", {0.0, 0.0, 1.0}}}};

    // Generate particles using the surface source
    ParticleVector particles = ParticleGenerator::fromSurfaceSource({surfaceSource});

    double dt{0.01};
    for (Particle const &particle : particles)
    {
        // Checking bounds of the generated particles.
        // $$ Coordinates. $$ //
        EXPECT_GE(particle.getX(), 0.0);
        EXPECT_LE(particle.getX(), 100.0);
        EXPECT_GE(particle.getY(), 0.0);
        EXPECT_LE(particle.getY(), 100.0);
        EXPECT_GE(particle.getZ(), 0.0);
        EXPECT_LE(particle.getZ(), 100.0);

        // $$ Velocities. $$ //
        // Velocities are set based on normal and energy in fromSurfaceSource.
        EXPECT_NEAR(particle.getVx(), 0.0, 1e-5);
        EXPECT_NEAR(particle.getVy(), 0.0, 1e-5);
        EXPECT_NEAR(particle.getVz(), 1e3, 50);

        // Checking correctness of the calculation.
        double expectedRadius{ParticlePropertiesManager::getRadiusFromType(util::getParticleTypeFromStrRepresentation(surfaceSource.type))},
            expectedMass{ParticlePropertiesManager::getMassFromType(util::getParticleTypeFromStrRepresentation(surfaceSource.type))},
            expectedVTI{ParticlePropertiesManager::getViscosityTemperatureIndexFromType(util::getParticleTypeFromStrRepresentation(surfaceSource.type))},
            expectedVSSDeflection{ParticlePropertiesManager::getVSSDeflectionParameterFromType(util::getParticleTypeFromStrRepresentation(surfaceSource.type))},
            expectedCharge{ParticlePropertiesManager::getChargeFromType(util::getParticleTypeFromStrRepresentation(surfaceSource.type))};

        EXPECT_DOUBLE_EQ(particle.getRadius(), expectedRadius);
        EXPECT_DOUBLE_EQ(particle.getMass(), expectedMass);
        EXPECT_DOUBLE_EQ(particle.getViscosityTemperatureIndex(), expectedVTI);
        EXPECT_DOUBLE_EQ(particle.getVSSDeflectionParameter(), expectedVSSDeflection);
        EXPECT_DOUBLE_EQ(particle.getCharge(), expectedCharge);

        double vx{particle.getVx()},
            vy{particle.getVy()},
            vz{particle.getVz()},
            mass{particle.getMass()},
            expectedEnergy{0.5 * mass * std::sqrt(vx * vx + vy * vy + vz * vz)};
        EXPECT_NEAR(particle.getEnergy_J(), expectedEnergy, 1e-15);

        Particle other(util::getParticleTypeFromStrRepresentation(surfaceSource.type), 50.0, 50.0, 50.0, -25.0, -25.0, -25.0);
        if (particle.overlaps(other))
        {
            EXPECT_TRUE(CGAL::do_overlap(particle.getBoundingBox(), other.getBoundingBox()));
        }

        double initX{particle.getX()},
            initY{particle.getY()},
            initZ{particle.getZ()};

        Particle updatedParticle{particle};
        updatedParticle.updatePosition(dt);

        EXPECT_NEAR(updatedParticle.getX(), initX + particle.getVx() * dt, 1e-5);
        EXPECT_NEAR(updatedParticle.getY(), initY + particle.getVy() * dt, 1e-5);
        EXPECT_NEAR(updatedParticle.getZ(), initZ + particle.getVz() * dt, 1e-5);
    }

    Particle p1(util::getParticleTypeFromStrRepresentation(surfaceSource.type), 0.0, 0.0, 0.0, 10.0, 0.0, 0.0),
        p2(util::getParticleTypeFromStrRepresentation(surfaceSource.type), 0.5, 0.0, 0.0, -10.0, 0.0, 0.0);
    EXPECT_TRUE(p1.getVx() > 0 && p2.getVx() < 0);

    auto collisionModel = CollisionModelFactory::create("HS");
    bool collided{collisionModel->collide(p1, util::getParticleTypeFromStrRepresentation(surfaceSource.type), 1.0, dt)};
    if (collided)
    {
        EXPECT_TRUE(CGAL::do_overlap(p1.getBoundingBox(), p2.getBoundingBox()));
    }

    MagneticInduction magneticField{0.01, 0.0, 0.0};
    ElectricField electricField{0.0, 0.01, 0.0};
    p1.electroMagneticPush(magneticField, electricField, dt);

    double expectedNewVx{p1.getVx() + (p1.getCharge() / p1.getMass()) * electricField.getX() * dt};
    EXPECT_NEAR(p1.getVx(), expectedNewVx, 1e-5);
}
