#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#ifdef USE_OMP
#include <omp.h>
#endif

#include <CGAL/Bbox_3.h>
#include <atomic>

#include "Geometry/GeometryTypes.hpp"
#include "Geometry/MathVector.hpp"
#include "Particle/ParticleUtils.hpp"
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
    CGAL::Bbox_3 m_bbox;                 ///< Bounding box for particle.

    /// @brief Calculates bounding box for the current particle.
    void calculateBoundingBox() noexcept;

public:
    Particle() : m_bbox(0, 0, 0, 0, 0, 0) {}
    Particle(ParticleType type_);
    Particle(ParticleType type_, double x_, double y_, double z_, double energyJ_, std::array<double, 3> const &thetaPhi);
    Particle(ParticleType type_, double x_, double y_, double z_, double vx_, double vy_, double vz_);
    Particle(ParticleType type_, Point const &centre, double vx_, double vy_, double vz_);
    Particle(ParticleType type_, Point &&centre, double vx_, double vy_, double vz_);
    Particle(ParticleType type_, Point const &centre, double energyJ_, std::array<double, 3> const &thetaPhi);
    Particle(ParticleType type_, Point &&centre, double energyJ_, std::array<double, 3> const &thetaPhi);
    Particle(ParticleType type_, double x_, double y_, double z_, VelocityVector const &velvec);
    Particle(ParticleType type_, double x_, double y_, double z_, VelocityVector &&velvec);
    Particle(ParticleType type_, Point const &centre, VelocityVector const &velvec);
    Particle(ParticleType type_, Point &&centre, VelocityVector &&velvec);
    ~Particle() {}

#ifdef USE_CUDA
    explicit Particle(ParticleDevice_t const &deviceParticle)
        : m_id(deviceParticle.id),
          m_type(static_cast<ParticleType>(deviceParticle.type)),
          m_centre(deviceParticle.x, deviceParticle.y, deviceParticle.z),
          m_velocity(deviceParticle.vx, deviceParticle.vy, deviceParticle.vz),
          m_energy(deviceParticle.energy)
    {
    }

    Particle &operator=(ParticleDevice_t const &deviceParticle)
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
    constexpr CGAL::Bbox_3 const &getBoundingBox() const { return m_bbox; }
    constexpr ParticleType getType() const { return m_type; }
    double getMass() const { return ParticleUtils::getMassFromType(m_type); }
    double getRadius() const { return ParticleUtils::getRadiusFromType(m_type); }
    double getViscosityTemperatureIndex() const { return ParticleUtils::getViscosityTemperatureIndexFromType(m_type); }
    double getVSSDeflectionParameter() const { return ParticleUtils::getVSSDeflectionParameterFromType(m_type); }
    double getCharge() const { return ParticleUtils::getChargeFromType(m_type); }
    int getChargeInIons() const { return ParticleUtils::getChargeInIonsFromType(m_type); }

    /**
     * @brief Chooses the specified scattering model.
     * @param target particle of gas with which current particle will colide.
     * @param n_concentration concentration of particles.
     * @param model scattering model (available: HS/VHS/VSS)
     * @param time_step simulation time step.
     * @return `true` if colided, otherwise `false`.
     */
    bool colide(Particle target, double n_concentration, std::string_view model, double time_step);

    bool colideHS(Particle target, double n_concentration, double time_step);
    bool colideVHS(Particle target, double n_concentration, double omega, double time_step);
    bool colideVSS(Particle target, double n_concentration, double omega, double alpha, double time_step);

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
    void electroMagneticPush(MagneticInduction const &magneticInduction, ElectricField const &electricField, double time_step);

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
               lhs.m_energy == rhs.m_energy &&
               lhs.m_bbox == rhs.m_bbox;
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
using ParticleVector = std::vector<Particle>;

#endif // !PARTICLE_HPP
