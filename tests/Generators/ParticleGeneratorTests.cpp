#include <gtest/gtest.h>

#include "ParticleGenerator.hpp"
#include "RealNumberGenerator.hpp"

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
        EXPECT_GE(p.getEnergy_eV(), 10.0);
        EXPECT_LE(p.getEnergy_eV(), 20.0);
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
