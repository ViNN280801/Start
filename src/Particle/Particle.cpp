#include "Particle/Particle.hpp"
#include "Generators/Host/RealNumberGeneratorHost.hpp"
#include "Particle/PhysicsCore/ParticleDynamicUtils.hpp"

std::atomic<size_t> Particle::m_nextId{0ul};

void Particle::calculateBoundingBox() noexcept
{
	m_bbox = CGAL::Bbox_3(getX() - getRadius(), getY() - getRadius(), getZ() - getRadius(),
						  getX() + getRadius(), getY() + getRadius(), getZ() + getRadius());
}

Particle::Particle(ParticleType type_)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(Point(0, 0, 0)),
	  m_velocity(0, 0, 0),
	  m_bbox(0, 0, 0, 0, 0, 0) {}

Particle::Particle(ParticleType type_, double x_, double y_, double z_,
				   double energy_eV, std::array<double, 3> const &thetaPhi)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(Point(x_, y_, z_)),
	  m_energy(energy_eV)
{
	m_velocity = ParticleDynamicUtils::calculateVelocityFromEnergy_eV(m_energy, getMass(), thetaPhi);
	calculateBoundingBox();
}

Particle::Particle(ParticleType type_, double x_, double y_, double z_,
				   double vx_, double vy_, double vz_)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(Point(x_, y_, z_)),
	  m_velocity(MathVector(vx_, vy_, vz_))
{
	ParticleDynamicUtils::calculateEnergyJFromVelocity(m_energy, getMass(), vx_, vy_, vz_);
	calculateBoundingBox();
}

Particle::Particle(ParticleType type_, Point const &centre,
				   double vx_, double vy_, double vz_)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(centre),
	  m_velocity(MathVector(vx_, vy_, vz_))
{
	ParticleDynamicUtils::calculateEnergyJFromVelocity(m_energy, getMass(), vx_, vy_, vz_);
	calculateBoundingBox();
}

Particle::Particle(ParticleType type_, Point &&centre,
				   double vx_, double vy_, double vz_)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(std::move(centre)),
	  m_velocity(MathVector(vx_, vy_, vz_))
{
	ParticleDynamicUtils::calculateEnergyJFromVelocity(m_energy, getMass(), vx_, vy_, vz_);
	calculateBoundingBox();
}

Particle::Particle(ParticleType type_, Point const &centre, double energy_eV,
				   std::array<double, 3> const &thetaPhi)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(centre),
	  m_energy(energy_eV)
{
	m_velocity = ParticleDynamicUtils::calculateVelocityFromEnergy_eV(m_energy, getMass(), thetaPhi);
	calculateBoundingBox();
}

Particle::Particle(ParticleType type_, Point &&centre, double energy_eV,
				   std::array<double, 3> const &thetaPhi)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(std::move(centre)),
	  m_energy(energy_eV)
{
	m_velocity = ParticleDynamicUtils::calculateVelocityFromEnergy_eV(m_energy, getMass(), thetaPhi);
	calculateBoundingBox();
}

Particle::Particle(ParticleType type_, double x_, double y_, double z_,
				   VelocityVector const &velvec)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(Point(x_, y_, z_)),
	  m_velocity(velvec)
{
	ParticleDynamicUtils::calculateEnergyJFromVelocity(m_energy, getMass(), velvec.getX(), velvec.getY(), velvec.getZ());
	calculateBoundingBox();
}

Particle::Particle(ParticleType type_, double x_, double y_, double z_,
				   VelocityVector &&velvec)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(Point(x_, y_, z_)),
	  m_velocity(std::move(velvec))
{
	ParticleDynamicUtils::calculateEnergyJFromVelocity(m_energy, getMass(), velvec.getX(), velvec.getY(), velvec.getZ());
	calculateBoundingBox();
}

Particle::Particle(ParticleType type_, Point const &centre,
				   VelocityVector const &velvec)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(centre),
	  m_velocity(velvec)
{
	ParticleDynamicUtils::calculateEnergyJFromVelocity(m_energy, getMass(), velvec.getX(), velvec.getY(), velvec.getZ());
	calculateBoundingBox();
}

Particle::Particle(ParticleType type_, Point &&centre,
				   VelocityVector &&velvec)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(std::move(centre)),
	  m_velocity(std::move(velvec))
{
	ParticleDynamicUtils::calculateEnergyJFromVelocity(m_energy, getMass(), velvec.getX(), velvec.getY(), velvec.getZ());
	calculateBoundingBox();
}

void Particle::updatePosition(double dt)
{
	// Update particle positions: x = x + Vx ⋅ Δt
	double upd_x{CGAL_TO_DOUBLE(m_centre.x()) + getVx() * dt},
		upd_y{CGAL_TO_DOUBLE(m_centre.y()) + getVy() * dt},
		upd_z{CGAL_TO_DOUBLE(m_centre.z()) + getVz() * dt};

	m_centre = Point(upd_x, upd_y, upd_z);
}

bool Particle::overlaps(Particle const &other) const
{
	// Distance between particles
	double distance_{PositionVector(CGAL_TO_DOUBLE(m_centre.x()),
									CGAL_TO_DOUBLE(m_centre.y()),
									CGAL_TO_DOUBLE(m_centre.z()))
						 .distance(PositionVector(CGAL_TO_DOUBLE(other.m_centre.x()),
												  CGAL_TO_DOUBLE(other.m_centre.y()),
												  CGAL_TO_DOUBLE(other.m_centre.z())))};
	return distance_ < (getRadius() + other.getRadius());
}

bool Particle::overlaps(Particle &&other) const
{
	double distance_{PositionVector(CGAL_TO_DOUBLE(m_centre.x()),
									CGAL_TO_DOUBLE(m_centre.y()),
									CGAL_TO_DOUBLE(m_centre.z()))
						 .distance(PositionVector(CGAL_TO_DOUBLE(other.m_centre.x()),
												  CGAL_TO_DOUBLE(other.m_centre.y()),
												  CGAL_TO_DOUBLE(other.m_centre.z())))};
	return distance_ < (getRadius() + other.getRadius());
}

double Particle::getX() const { return CGAL_TO_DOUBLE(m_centre.x()); }
double Particle::getY() const { return CGAL_TO_DOUBLE(m_centre.y()); }
double Particle::getZ() const { return CGAL_TO_DOUBLE(m_centre.z()); }
double Particle::getPositionModule() const { return PositionVector(CGAL_TO_DOUBLE(m_centre.x()), CGAL_TO_DOUBLE(m_centre.y()), CGAL_TO_DOUBLE(m_centre.z())).module(); }
double Particle::getVelocityModule() const { return m_velocity.module(); }

void Particle::electroMagneticPush(MagneticInduction const &magneticInduction, ElectricField const &electricField, double time_step)
{
	// Checking 1. Time step can't be null.
	if (time_step == 0.0)
	{
		WARNINGMSG(util::stringify("There is no any movement in particle[", m_id, "]: Time step is 0"));
		return;
	}

	// Checking 2. If both of vectors are null - just skip pushing particle with EM.
	if (magneticInduction.isNull() && electricField.isNull())
		return;

	// 1. Calculating acceleration using II-nd Newton's Law:
	MathVector<double> a_L{getCharge() * (electricField + m_velocity.crossProduct(magneticInduction)) / getMass()};

	/// Update particle positions: \( x = x + V_x \cdot \Delta t \)
	/// 2. Acceleration semistep: \( V_- = V_{\text{old}} + a_L \cdot \frac{\Delta t}{2} \)
	MathVector<double> v_minus{m_velocity + a_L * time_step / 2.};

	/// 3. Rotation:
	/// \f{align}{
	/// t &= \frac{q B \Delta t}{2 m},
	/// s &= \frac{2 t}{1 + |t|^2},
	/// V' &= V_- + V_- \times t,
	/// V_+ &= V_- + V' \times s.
	/// \f}
	MathVector<double> t{getCharge() * magneticInduction * time_step / (2. * getMass())},
		s{2. * t / (1 + t.module() * t.module())},
		v_apostrophe{v_minus + v_minus.crossProduct(t)},
		v_plus{v_minus + v_apostrophe.crossProduct(s)};

	/// 4. Final acceleration semistep: \( v_{\text{upd}} = V_+ + a_L \cdot \frac{\Delta t}{2} \)
	/// \( E_{\text{cell}} = \sum (\varphi_i \cdot \nabla \varphi_i) \), where \( i \) is the global index of the node.
	m_velocity = v_plus + a_L * time_step / 2.;

	// 5. Updating energy after updating velocity:
	ParticleDynamicUtils::calculateEnergyJFromVelocity(m_energy, getMass(), m_velocity.getX(), m_velocity.getY(), m_velocity.getZ());
}

std::ostream &operator<<(std::ostream &os, Particle const &particle)
{
	std::cout << "Particle[" << particle.getId() << "]:\n"
			  << "Center: " << particle.getX() << " " << particle.getY() << " " << particle.getZ() << "\n"
			  << "Radius: " << particle.getRadius() << "\n"
			  << "Velocity components: " << particle.getVx() << " " << particle.getVy() << " " << particle.getVz() << "\n"
			  << "Energy: " << particle.getEnergy_eV() << " eV\n\n";
	return os;
}
