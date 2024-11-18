#include "Particle/Particle.hpp"
#include "Generators/RealNumberGenerator.hpp"

std::atomic<size_t> Particle::m_nextId{0ul};

void Particle::calculateVelocityFromEnergy_eV(std::array<double, 3> const &thetaPhi)
{
	// GUI sends energy in eV, so, we need to convert it from eV to J:
	m_energy *= constants::physical_constants::eV_J;

	auto [thetaUsers, phiCalculated, thetaCalculated]{thetaPhi};
	RealNumberGenerator rng;

	double theta{thetaCalculated + rng(-1, 1) * thetaUsers},
		v{std::sqrt(2 * getEnergy_J() / getMass())},
		vx{v * std::sin(theta) * std::cos(phiCalculated)},
		vy{v * std::sin(theta) * std::sin(phiCalculated)},
		vz{v * std::cos(theta)};

	m_velocity = VelocityVector(vx, vy, vz);
}

void Particle::calculateEnergyJFromVelocity(double vx, double vy, double vz) noexcept { m_energy = getMass() * std::pow((VelocityVector(vx, vy, vz).module()), 2) / 2; }
void Particle::calculateEnergyJFromVelocity(VelocityVector const &v) noexcept { calculateEnergyJFromVelocity(VelocityVector(v.getX(), v.getZ(), v.getZ())); }
void Particle::calculateEnergyJFromVelocity(VelocityVector &&v) noexcept { calculateEnergyJFromVelocity(v.getX(), v.getZ(), v.getZ()); }

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
	calculateVelocityFromEnergy_eV(thetaPhi);
	calculateBoundingBox();
}

Particle::Particle(ParticleType type_, double x_, double y_, double z_,
				   double vx_, double vy_, double vz_)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(Point(x_, y_, z_)),
	  m_velocity(MathVector(vx_, vy_, vz_))
{
	calculateEnergyJFromVelocity(m_velocity);
	calculateBoundingBox();
}

Particle::Particle(ParticleType type_, Point const &centre,
				   double vx_, double vy_, double vz_)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(centre),
	  m_velocity(MathVector(vx_, vy_, vz_))
{
	calculateEnergyJFromVelocity(m_velocity);
	calculateBoundingBox();
}

Particle::Particle(ParticleType type_, Point &&centre,
				   double vx_, double vy_, double vz_)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(std::move(centre)),
	  m_velocity(MathVector(vx_, vy_, vz_))
{
	calculateEnergyJFromVelocity(m_velocity);
	calculateBoundingBox();
}

Particle::Particle(ParticleType type_, Point const &centre, double energy_eV,
				   std::array<double, 3> const &thetaPhi)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(centre),
	  m_energy(energy_eV)
{
	calculateVelocityFromEnergy_eV(thetaPhi);
	calculateBoundingBox();
}

Particle::Particle(ParticleType type_, Point &&centre, double energy_eV,
				   std::array<double, 3> const &thetaPhi)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(std::move(centre)),
	  m_energy(energy_eV)
{
	calculateVelocityFromEnergy_eV(thetaPhi);
	calculateBoundingBox();
}

Particle::Particle(ParticleType type_, double x_, double y_, double z_,
				   VelocityVector const &velvec)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(Point(x_, y_, z_)),
	  m_velocity(velvec)
{
	calculateEnergyJFromVelocity(m_velocity);
	calculateBoundingBox();
}

Particle::Particle(ParticleType type_, double x_, double y_, double z_,
				   VelocityVector &&velvec)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(Point(x_, y_, z_)),
	  m_velocity(std::move(velvec))
{
	calculateEnergyJFromVelocity(m_velocity);
	calculateBoundingBox();
}

Particle::Particle(ParticleType type_, Point const &centre,
				   VelocityVector const &velvec)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(centre),
	  m_velocity(velvec)
{
	calculateEnergyJFromVelocity(m_velocity);
	calculateBoundingBox();
}

Particle::Particle(ParticleType type_, Point &&centre,
				   VelocityVector &&velvec)
	: m_id(m_nextId++),
	  m_type(type_),
	  m_centre(std::move(centre)),
	  m_velocity(std::move(velvec))
{
	calculateEnergyJFromVelocity(m_velocity);
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

bool Particle::colide(Particle target, double n_concentration, std::string_view model, double time_step)
{
	if (std::string(model) == "HS")
		return colideHS(target, n_concentration, time_step);
	else if (std::string(model) == "VHS")
		return colideVHS(target, n_concentration, target.getViscosityTemperatureIndex(), time_step);
	else if (std::string(model) == "VSS")
		return colideVSS(target, n_concentration, target.getViscosityTemperatureIndex(), target.getVSSDeflectionParameter(), time_step);
	else
	{
		ERRMSG("No such kind of scattering model. Available only: HS/VHS/VSS");
	}
	return false;
}

bool Particle::colideHS(Particle target, double n_concentration, double time_step)
{
	double p_mass{getMass()},
		t_mass{target.getMass()},
		sigma{(START_PI_NUMBER)*std::pow(getRadius() + target.getRadius(), 2)};

	// Probability of the scattering
	double Probability{sigma * getVelocityModule() * n_concentration * time_step};

	// Result of the collision: if colide -> change attributes of the particle
	RealNumberGenerator rng;
	bool iscolide{rng() < Probability};
	if (iscolide)
	{
		double xi_cos{rng(-1, 1)}, xi_sin{sqrt(1 - xi_cos * xi_cos)},
			phi{rng(0, 2 * START_PI_NUMBER)};

		double x{xi_sin * cos(phi)}, y{xi_sin * sin(phi)}, z{xi_cos},
			mass_cp{p_mass / (t_mass + p_mass)},
			mass_ct{t_mass / (t_mass + p_mass)};

		VelocityVector cm_vel(m_velocity * mass_cp), p_vec(mass_ct * m_velocity);
		double mp{p_vec.module()};
		VelocityVector dir_vector(x * mp, y * mp, z * mp);

		m_velocity = dir_vector + cm_vel;
		
		// Updating energy after updating velocity:
		calculateEnergyJFromVelocity(m_velocity);
	}
	return iscolide;
}

bool Particle::colideVHS(Particle target, double n_concentration, double omega, double time_step)
{
	double d_reference{(getRadius() + target.getRadius())},
		mass_constant{getMass() * target.getMass() / (getMass() + target.getMass())},
		t_mass{target.getMass()}, p_mass{getMass()},
		Exponent{omega - 1. / 2.};

	double d_vhs_2{(std::pow(d_reference, 2) / std::tgamma(5. / 2. - omega)) *
				   std::pow(2 * KT_reference /
								(mass_constant * getVelocityModule() * getVelocityModule()),
							Exponent)};

	double sigma{START_PI_NUMBER * d_vhs_2},
		Probability{sigma * getVelocityModule() * n_concentration * time_step};

	RealNumberGenerator rng;
	bool iscolide{rng() < Probability};
	if (iscolide)
	{
		double xi_cos{rng(-1, 1)}, xi_sin{sqrt(1 - xi_cos * xi_cos)},
			phi{rng(0, 2 * START_PI_NUMBER)};

		double x{xi_sin * cos(phi)}, y{xi_sin * sin(phi)}, z{xi_cos},
			mass_cp{p_mass / (t_mass + p_mass)},
			mass_ct{t_mass / (t_mass + p_mass)};

		VelocityVector cm_vel(m_velocity * mass_cp), p_vec(mass_ct * m_velocity);
		double mp{p_vec.module()};
		VelocityVector dir_vector(x * mp, y * mp, z * mp);

		m_velocity = dir_vector + cm_vel;
		
		// Updating energy after updating velocity:
		calculateEnergyJFromVelocity(m_velocity);
	}

	return iscolide;
}

bool Particle::colideVSS(Particle target, double n_concentration, double omega,
						 double alpha, double time_step)
{
	double d_reference{(getRadius() + target.getRadius())},
		mass_constant{getMass() * target.getMass() / (getMass() + target.getMass())},
		t_mass{target.getMass()},
		p_mass{getMass()},
		Exponent{omega - 1. / 2.};

	double d_vhs_2{(std::pow(d_reference, 2) / std::tgamma(5. / 2. - omega)) *
				   std::pow(2 * KT_reference /
								(mass_constant * getVelocityModule() * getVelocityModule()),
							Exponent)};

	double sigma{START_PI_NUMBER * d_vhs_2},
		Probability{sigma * getVelocityModule() * n_concentration * time_step};

	RealNumberGenerator rng;
	bool iscolide{rng() < Probability};
	if (iscolide)
	{
		double xi_cos{2 * std::pow(rng(), 1. / alpha) - 1.}, xi_sin{sqrt(1 - xi_cos * xi_cos)},
			phi{rng(0, 2 * START_PI_NUMBER)};

		double x{xi_sin * cos(phi)}, y{xi_sin * sin(phi)}, z{xi_cos},
			mass_cp{p_mass / (t_mass + p_mass)},
			mass_ct{t_mass / (t_mass + p_mass)};

		VelocityVector cm_vel(m_velocity * mass_cp), p_vec(mass_ct * m_velocity);
		double mp{p_vec.module()};
		auto angles{p_vec.calcBetaGamma()};
		VelocityVector dir_vector(x * mp, y * mp, z * mp);
		dir_vector.rotation(angles);

		m_velocity = dir_vector + cm_vel;

		// Updating energy after updating velocity:
		calculateEnergyJFromVelocity(m_velocity);
	}
	return iscolide;
}

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

	// 2. Acceleration semistep: V_- = V_old + a_L ⋅ Δt/2.
	MathVector<double> v_minus{m_velocity + a_L * time_step / 2.};

	// 3. Rotation:
	/*
		t = qBΔt/(2m).
		s = 2t/(1 + |t|^2).
		V' = V_- + V_- × t.
		V_+ = V_- + V' × s.
	*/
	MathVector<double> t{getCharge() * magneticInduction * time_step / (2. * getMass())},
		s{2. * t / (1 + t.module() * t.module())},
		v_apostrophe{v_minus + v_minus.crossProduct(t)},
		v_plus{v_minus + v_apostrophe.crossProduct(s)};

	// 4. Final acceleration semistep: v_upd = v_+ + a_L ⋅ Δt/2.
	m_velocity = v_plus + a_L * time_step / 2.;

	// 5. Updating energy after updating velocity:
	calculateEnergyJFromVelocity(m_velocity);
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
