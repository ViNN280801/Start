#include <gtest/gtest.h>
#include <iomanip>

#include "ParticleUtils.hpp"
#include "ParticleGenerator.hpp"
#include "RealNumberGenerator.hpp"


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

// === Clean Test for byVelocities Method === //

TEST_F(ParticleGeneratorTest, CleanByVelocities)
{
    size_t particleCount = 100;
    ParticleType type = ParticleType::Ar;

    ParticleVector particles = ParticleGenerator::byVelocities(
        particleCount, type,
        0.0, 0.0, 0.0, 100.0, 100.0, 100.0,
        -50.0, -50.0, -50.0, 50.0, 50.0, 50.0);

    ASSERT_EQ(particles.size(), particleCount);
    for (Particle const &p : particles)
    {
        EXPECT_GE(p.getX(), 0.0);
        EXPECT_LE(p.getX(), 100.0);
        EXPECT_GE(p.getVx(), -50.0);
        EXPECT_LE(p.getVx(), 50.0);
    }
}

// === Dirty Tests for byVelocities Method === //

TEST_F(ParticleGeneratorTest, DirtyInvalidVelocityBounds)
{
    size_t particleCount = 10;
    ParticleType type = ParticleType::Ag;

    ParticleVector particles = ParticleGenerator::byVelocities(
        particleCount, type,
        0.0, 0.0, 0.0, 100.0, 100.0, 100.0,
        50.0, 50.0, 50.0, -50.0, -50.0, -50.0); // Invalid velocity bounds

    ASSERT_EQ(particles.size(), particleCount);
    for (Particle const &p : particles)
    {
        EXPECT_GE(p.getVx(), -50.0);
        EXPECT_LE(p.getVx(), 50.0);
    }
}

TEST_F(ParticleGeneratorTest, DirtyZeroParticleCountVelocities)
{
    size_t particleCount = 0;
    ParticleType type = ParticleType::Ne;

    EXPECT_ANY_THROW(ParticleGenerator::byVelocities(
        particleCount, type,
        0.0, 0.0, 0.0, 100.0, 100.0, 100.0,
        -50.0, -50.0, -50.0, 50.0, 50.0, 50.0));
}

TEST_F(ParticleGeneratorTest, DirtyNegativePositionRange)
{
    size_t particleCount = 10;
    ParticleType type = ParticleType::Ni;

    ParticleVector particles = ParticleGenerator::byVelocities(
        particleCount, type,
        -100.0, -100.0, -100.0, -10.0, -10.0, -10.0,
        -50.0, -50.0, -50.0, 50.0, 50.0, 50.0);

    ASSERT_EQ(particles.size(), particleCount);
    for (Particle const &p : particles)
    {
        EXPECT_GE(p.getX(), -100.0);
        EXPECT_LE(p.getX(), -10.0);
    }
}

TEST_F(ParticleGeneratorTest, DirtyExtremeVelocityValues)
{
    size_t particleCount = 10;
    ParticleType type = ParticleType::O2;

    ParticleVector particles = ParticleGenerator::byVelocities(
        particleCount, type,
        0.0, 0.0, 0.0, 100.0, 100.0, 100.0,
        -1e12, -1e12, -1e12, 1e12, 1e12, 1e12); // Extreme velocities

    ASSERT_EQ(particles.size(), particleCount);
    for (Particle const &p : particles)
    {
        EXPECT_GE(p.getVx(), -1e12);
        EXPECT_LE(p.getVx(), 1e12);
    }
}

TEST_F(ParticleGeneratorTest, DirtyInvertedPositionBounds)
{
    size_t particleCount = 10;
    ParticleType type = ParticleType::He;

    ParticleVector particles = ParticleGenerator::byVelocities(
        particleCount, type,
        100.0, 100.0, 100.0, 0.0, 0.0, 0.0, // Inverted bounds
        -50.0, -50.0, -50.0, 50.0, 50.0, 50.0);

    ASSERT_EQ(particles.size(), particleCount);
    for (Particle const &p : particles)
    {
        EXPECT_GE(p.getX(), 0.0);
        EXPECT_LE(p.getX(), 100.0);
    }
}

// === Clean Test for byVelocityModule Method === //

TEST_F(ParticleGeneratorTest, CleanByVelocityModule)
{
    size_t particleCount = 50;
    ParticleType type = ParticleType::Ne;

    ParticleVector particles = ParticleGenerator::byVelocityModule(
        particleCount, type,
        0.0, 0.0, 0.0,
        50.0, M_PI, M_PI / 2);

    ASSERT_EQ(particles.size(), particleCount);
    for (Particle const &p : particles)
    {
        EXPECT_DOUBLE_EQ(p.getVelocityModule(), 50.0); // Velocity magnitude should match the input
    }
}

// === Dirty Tests for byVelocityModule Method === //

TEST_F(ParticleGeneratorTest, DirtyByVelocityModuleNegativeVelocity)
{
    size_t particleCount = 10;
    ParticleType type = ParticleType::Ar;

    ParticleVector particles = ParticleGenerator::byVelocityModule(
        particleCount, type,
        0.0, 0.0, 0.0,
        -10.0, M_PI, M_PI / 2); // Negative velocity magnitude

    ASSERT_EQ(particles.size(), particleCount);
    for (Particle const &p : particles)
    {
        EXPECT_GE(p.getVx(), -10.0); // Velocities should be valid despite negative input
    }
}

TEST_F(ParticleGeneratorTest, DirtyByVelocityModuleZeroParticles)
{
    size_t particleCount = 0;
    ParticleType type = ParticleType::O2;

    EXPECT_THROW(ParticleGenerator::byVelocityModule(
                     particleCount, type,
                     0.0, 0.0, 0.0,
                     50.0, M_PI, M_PI / 2),
                 std::logic_error);
}

TEST_F(ParticleGeneratorTest, DirtyByVelocityModuleZeroVelocity)
{
    size_t particleCount = 10;
    ParticleType type = ParticleType::Ag;

    ParticleVector particles = ParticleGenerator::byVelocityModule(
        particleCount, type,
        0.0, 0.0, 0.0,
        0.0, M_PI, M_PI / 2); // Zero velocity

    ASSERT_EQ(particles.size(), particleCount);
    for (Particle const &p : particles)
    {
        EXPECT_DOUBLE_EQ(p.getVelocityModule(), 0.0); // All velocities should be zero
    }
}

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
        EXPECT_GE(p.getEnergy_J(), 10.0);
        EXPECT_LE(p.getEnergy_J(), 20.0);
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
        EXPECT_GE(p.getEnergy_J(), 10.0);
        EXPECT_LE(p.getEnergy_J(), 20.0);
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
    ParticleType type{ParticleType::Ar};

    EXPECT_NO_THROW({
        ParticleVector particles = ParticleGenerator::byVelocities(
            particleCount, type,
            0.0, 0.0, 0.0, 100.0, 100.0, 100.0,
            -50.0, -50.0, -50.0, 50.0, 50.0, 50.0
        );
        ASSERT_EQ(particles.size(), particleCount);

        printMemoryUsage<Particle>(particleCount, "Memory usage for large particle count test");
    });
}

TEST_F(ParticleGeneratorTest, StressExtremeVelocityValues) 
{
    size_t particleCount{1'000'000};
    ParticleType type{ParticleType::O2};

    ParticleVector particles = ParticleGenerator::byVelocities(
        particleCount, type,
        -1e6, -1e6, -1e6, 1e6, 1e6, 1e6, 
        -1e12, -1e12, -1e12, 1e12, 1e12, 1e12 
    );

    ASSERT_EQ(particles.size(), particleCount);
    for (Particle const &p : particles) {
        EXPECT_GE(p.getVx(), -1e12);
        EXPECT_LE(p.getVx(), 1e12);
    }
    printMemoryUsage<Particle>(particleCount, "Memory usage for extreme velocity values");
}

TEST_F(ParticleGeneratorTest, StressParallelGeneration) 
{
    size_t particleCount{5'000'000};
    ParticleType type{ParticleType::He};

    auto start = std::chrono::high_resolution_clock::now();

    ParticleVector particles = ParticleGenerator::byVelocities(
        particleCount, type,
        0.0, 0.0, 0.0, 100.0, 100.0, 100.0,
        -50.0, -50.0, -50.0, 50.0, 50.0, 50.0
    );

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
    ParticleType type{ParticleType::Au};

    ParticleVector particles{ParticleGenerator::byVelocities(
        particleCount, type,
        0.0, 0.0, 0.0,
        100.0, 100.0, 100.0,
        -50.0, -50.0, -50.0,
        50.0, 50.0, 50.0
    )};

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
        EXPECT_GE(particle.getVx(), -50.0);
        EXPECT_LE(particle.getVx(), 50.0);
        EXPECT_GE(particle.getVy(), -50.0);
        EXPECT_LE(particle.getVy(), 50.0);
        EXPECT_GE(particle.getVz(), -50.0);
        EXPECT_LE(particle.getVz(), 50.0);

        // Checking correctness of the calculation.
        double expectedRadius{ParticleUtils::getRadiusFromType(type)},
            expectedMass{ParticleUtils::getMassFromType(type)},
            expectedVTI{ParticleUtils::getViscosityTemperatureIndexFromType(type)},
            expectedVSSDeflection{ParticleUtils::getVSSDeflectionParameterFromType(type)},
            expectedCharge{ParticleUtils::getChargeFromType(type)};

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

        Particle other(type, 50.0, 50.0, 50.0, -25.0, -25.0, -25.0);
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
    Particle p1(type, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0),
        p2(type, 0.5, 0.0, 0.0, -10.0, 0.0, 0.0);
    EXPECT_TRUE(p1.getVx() > 0 && p2.getVx() < 0);
    
    bool colided{p1.colide(p2, 1.0, "HS", dt)};
    if (colided)
    {
        EXPECT_TRUE(CGAL::do_overlap(p1.getBoundingBox(), p2.getBoundingBox()));
    }

    MagneticInduction magneticField{0.01, 0.0, 0.0};
    ElectricField electricField{0.0, 0.01, 0.0};
    p1.electroMagneticPush(magneticField, electricField, dt);

    double expectedNewVx{p1.getVx() + (p1.getCharge() / p1.getMass()) * electricField.getX() * dt};
    EXPECT_NEAR(p1.getVx(), expectedNewVx, 1e-5);
}
