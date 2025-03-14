#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#ifdef USE_OMP
#include <omp.h>
#endif

#include <CGAL/Bbox_3.h>
#include <atomic>

#include "Geometry/Basics/BaseTypes.hpp"
#include "Geometry/Basics/GeometryVector.hpp"
#include "Particle/ParticlePropertiesManager.hpp"
#include "Utilities/ConfigParser.hpp"

#ifdef USE_CUDA
#include "Particle/CUDA/ParticleDevice.cuh"
#endif

/// @brief Represents a particle in a simulation.
class Particle
{
private:
    static std::atomic<size_t> m_nextId; ///< Static member for generating unique IDs.
    size_t m_id;                         ///< Id of the particle.
    ParticleType m_type{};               ///< Type of the particle.
    Point m_centre;                      ///< Position in Cartesian coordinates (x, y, z).
    VelocityVector m_velocity;           ///< Velocity vector (Vx, Vy, Vz).
    double m_energy{};                   ///< Particle energy in [J] by default.

public:
    Particle() {}
    Particle(ParticleType type_);
    Particle(ParticleType type_, double x_, double y_, double z_, double energyJ_, std::array<double, 3> const &thetaPhi);
    Particle(ParticleType type_, double x_, double y_, double z_, double vx_, double vy_, double vz_);
    Particle(ParticleType type_, Point_cref centre, double vx_, double vy_, double vz_);
    Particle(ParticleType type_, Point_rref centre, double vx_, double vy_, double vz_);
    Particle(ParticleType type_, Point_cref centre, double energyJ_, std::array<double, 3> const &thetaPhi);
    Particle(ParticleType type_, Point_rref centre, double energyJ_, std::array<double, 3> const &thetaPhi);
    Particle(ParticleType type_, double x_, double y_, double z_, VelocityVector_cref velvec);
    Particle(ParticleType type_, double x_, double y_, double z_, VelocityVector_rref velvec);
    Particle(ParticleType type_, Point_cref centre, VelocityVector_cref velvec);
    Particle(ParticleType type_, Point_rref centre, VelocityVector_rref velvec);
    ~Particle() {}

#ifdef USE_CUDA
    explicit Particle(ParticleDevice_cref deviceParticle)
        : m_id(deviceParticle.id),
          m_type(static_cast<ParticleType>(deviceParticle.type)),
          m_centre(deviceParticle.x, deviceParticle.y, deviceParticle.z),
          m_velocity(deviceParticle.vx, deviceParticle.vy, deviceParticle.vz),
          m_energy(deviceParticle.energy)
    {
    }

    Particle &operator=(ParticleDevice_cref deviceParticle)
    {
        m_id = deviceParticle.id;
        m_type = static_cast<ParticleType>(deviceParticle.type);
        m_centre = {deviceParticle.x, deviceParticle.y, deviceParticle.z};
        m_velocity = {deviceParticle.vx, deviceParticle.vy, deviceParticle.vz};
        m_energy = deviceParticle.energy;
        return *this;
    }
#endif

    /**
     * @brief Updates the position of the particle after a time interval.
     * @param dt Time interval for the update [s].
     */
    void updatePosition(double dt);

    /**
     * @brief Checks if the current particle overlaps with another particle.
     * @param other The other Particle to check against.
     * @return `true` if the particles overlap, otherwise `false`.
     */
    bool overlaps(Particle const &other) const;
    bool overlaps(Particle &&other) const;

    /* === Getters for particle params. === */
    constexpr size_t getId() const { return m_id; }
    double getX() const;
    double getY() const;
    double getZ() const;
    double getPositionModule() const;
    constexpr double getEnergy_J() const { return m_energy; }
    double getEnergy_eV() const { return m_energy / physical_constants::eV_J; }
    constexpr double getVx() const { return m_velocity.getX(); }
    constexpr double getVy() const { return m_velocity.getY(); }
    constexpr double getVz() const { return m_velocity.getZ(); }
    double getVelocityModule() const;
    constexpr Point const &getCentre() const { return m_centre; }
    constexpr VelocityVector const &getVelocityVector() const { return m_velocity; }
    constexpr VelocityVector &getVelocityVector() { return m_velocity; }
    constexpr ParticleType getType() const { return m_type; }
    double getMass() const { return ParticlePropertiesManager::getMassFromType(m_type); }
    double getRadius() const { return ParticlePropertiesManager::getRadiusFromType(m_type); }
    double getViscosityTemperatureIndex() const { return ParticlePropertiesManager::getViscosityTemperatureIndexFromType(m_type); }
    double getVSSDeflectionParameter() const { return ParticlePropertiesManager::getVSSDeflectionParameterFromType(m_type); }
    double getCharge() const { return ParticlePropertiesManager::getChargeFromType(m_type); }
    int getChargeInIons() const { return ParticlePropertiesManager::getChargeInIonsFromType(m_type); }
    /* === ---- ---- ---- ---- ---- --- === */

    /* === Setters for particle params. === */
    void setEnergy_eV(double energy_eV) { m_energy = util::convert_energy_eV_to_energy_J(energy_eV); }
    void setEnergy_J(double energy_J) { m_energy = energy_J; }

    void setVelocity(VelocityVector const &velvec) { m_velocity = velvec; }
    void setVelocity(double vx, double vy, double vz) { m_velocity = VelocityVector(vx, vy, vz); }

    void setCentre(Point const &centre) { m_centre = centre; }
    void setCentre(double x, double y, double z) { m_centre = Point(x, y, z); }
    /* === ---- ---- ---- ---- ---- --- === */

    /**
     * @brief Uses Boris Integrator to calculate updated velocity.
     * @details Lorentz force: F_L = q(E + v × B), where E - is the electric field,
                                                B - magnetic field,
                                                v - instantaneous velocity (velocity of the particle),
                                                q - charge of the particle.
                By using II-nd Newton's Law: a = F/m.
                                             a_L = F_L/m.
                                             a_L = [q(E + v × B)]/m.
     */
    void electroMagneticPush(MagneticInduction const &magneticInduction, ElectricField const &electricField, double time_step) noexcept;

    /**
     * @brief Compares this Particle object to another for equality.
     * @details Two particles are considered equal if all their corresponding
     *          properties are equal.
     * @param other The Particle object to compare against.
     * @return `true` if the particles are equal, `false` otherwise.
     */
    [[nodiscard("Check of Particle equality should not be ignored to prevent logical errors")]] friend bool operator==(const Particle &lhs, const Particle &rhs)
    {
        return lhs.m_id == rhs.m_id &&
               lhs.m_type == rhs.m_type &&
               lhs.m_centre == rhs.m_centre &&
               lhs.m_velocity == rhs.m_velocity &&
               lhs.m_energy == rhs.m_energy;
    }

    /**
     * @brief Compares this Particle object to another for inequality.
     * @details Two particles are considered unequal if any of their corresponding
     *          properties are not equal.
     * @param other The Particle object to compare against.
     * @return `true` if the particles are not equal, `false` otherwise.
     */
    [[nodiscard("Check of Particle inequality should not be ignored to ensure correct logic flow")]] friend bool operator!=(Particle const &lhs, Particle const &rhs) { return !(lhs == rhs); }
};

std::ostream &operator<<(std::ostream &os, Particle const &particle);

using Particle_ref = Particle &;
using Particle_cref = Particle const &;

using ParticleVector = std::vector<Particle>;
using ParticleVector_ref = ParticleVector &;
using ParticleVector_cref = ParticleVector const &;

#endif // !PARTICLE_HPP
