#ifdef LOG
#include <format>
#endif

#include <utility>

#include "../include/Particles/Particles.hpp"

void ParticleGeneric::calculateVelocityFromEnergy_J()
{
  // TODO: Here we need to calculate the velocity vector not only for sphere distribution
  // Example below:

  RealNumberGenerator rng;
  [[maybe_unused]] double v{std::sqrt(2 * m_energy / getMass())},
      theta{rng.get_double(0 - std::numeric_limits<long double>::min(),
                           std::numbers::pi + std::numeric_limits<long double>::min())},
      phi{rng.get_double(0 - std::numeric_limits<long double>::min(),
                         2 * std::numbers::pi + std::numeric_limits<long double>::min())},
      vx{m_radius * sin(theta) * cos(phi)},
      vy{m_radius * sin(theta) * sin(phi)},
      vz{m_radius * cos(theta)};

  m_velocity = VelocityVector(vx, vy, vz);
}

ParticleGeneric::ParticleGeneric(double x_, double y_, double z_,
                                 double energy_, double radius_)
    : m_centre(PointD(x_, y_, z_)),
      m_energy(energy_)
{
  calculateVelocityFromEnergy_J();
  m_radius = radius_;
  m_boundingBox = aabb::AABB({x_ - radius_, y_ - radius_, z_ - radius_},
                             {x_ + radius_, y_ + radius_, z_ + radius_});
}

ParticleGeneric::ParticleGeneric(double x_, double y_, double z_,
                                 double vx_, double vy_, double vz_,
                                 double radius_)
    : m_centre(PointD(x_, y_, z_)),
      m_velocity(MathVector(vx_, vy_, vz_)),
      m_radius(radius_),
      m_boundingBox({x_ - radius_, y_ - radius_, z_ - radius_},
                    {x_ + radius_, y_ + radius_, z_ + radius_}) {}

ParticleGeneric::ParticleGeneric(PointD centre,
                                 double vx_, double vy_, double vz_,
                                 double radius_)
    : m_centre(centre),
      m_velocity(MathVector(vx_, vy_, vz_)),
      m_radius(radius_),
      m_boundingBox({m_centre.x - radius_, m_centre.y - radius_, m_centre.z - radius_},
                    {m_centre.x + radius_, m_centre.y + radius_, m_centre.z + radius_}) {}

ParticleGeneric::ParticleGeneric(PointD centre, double energy_, double radius_)
    : m_centre(centre),
      m_energy(energy_)
{
  calculateVelocityFromEnergy_J();
  m_radius = radius_;
  m_boundingBox = aabb::AABB({m_centre.x - radius_, m_centre.y - radius_, m_centre.z - radius_},
                             {m_centre.x + radius_, m_centre.y + radius_, m_centre.z + radius_});
}

ParticleGeneric::ParticleGeneric(double x_, double y_, double z_,
                                 VelocityVector velvec,
                                 double radius_)
    : m_centre(PointD(x_, y_, z_)),
      m_velocity(velvec),
      m_radius(radius_),
      m_boundingBox({x_ - radius_, y_ - radius_, z_ - radius_},
                    {x_ + radius_, y_ + radius_, z_ + radius_}) {}

ParticleGeneric::ParticleGeneric(PointD centre,
                                 VelocityVector velvec,
                                 double radius_)
    : m_centre(centre),
      m_velocity(velvec),
      m_radius(radius_),
      m_boundingBox({m_centre.x - radius_, m_centre.y - radius_, m_centre.z - radius_},
                    {m_centre.x + radius_, m_centre.y + radius_, m_centre.z + radius_}) {}

void ParticleGeneric::updatePosition(double dt)
{
  // Update particle positions: x = x + Vx ⋅ Δt
  m_centre.x = m_centre.x + getVx() * dt;
  m_centre.y = m_centre.y + getVy() * dt;
  m_centre.z = m_centre.z + getVz() * dt;

  // Update the bounding box to the new position
  m_boundingBox = aabb::AABB({m_centre.x - m_radius, m_centre.y - m_radius, m_centre.z - m_radius},
                             {m_centre.x + m_radius, m_centre.y + m_radius, m_centre.z + m_radius});
}

bool ParticleGeneric::overlaps(ParticleGeneric const &other) const
{
  // Distance between particles
  double distance_{PositionVector(m_centre.x, m_centre.y, m_centre.z)
                       .distance(PositionVector(other.m_centre.x, other.m_centre.y, other.m_centre.z))};
  return distance_ < (m_radius + other.m_radius);
}

bool ParticleGeneric::isOutOfBounds(aabb::AABB const &bounding_volume) const
{
  return (m_boundingBox.lowerBound[0] <= bounding_volume.lowerBound[0] ||
          m_boundingBox.upperBound[0] >= bounding_volume.upperBound[0] ||
          m_boundingBox.lowerBound[1] <= bounding_volume.lowerBound[1] ||
          m_boundingBox.upperBound[1] >= bounding_volume.upperBound[1] ||
          m_boundingBox.lowerBound[2] <= bounding_volume.lowerBound[2] ||
          m_boundingBox.upperBound[2] >= bounding_volume.upperBound[2]);
}

void ParticleGeneric::colide(double rand, double p_mass, double t_mass)
{

  double mass_cp{p_mass / (t_mass + p_mass)};
  double mass_ct{t_mass / (t_mass + p_mass)};

  MathVector p_vector = mass_ct*m_velocity;
  MathVector cm_vel= mass_cp*m_velocity;

  double beta  = acos(p_vector.getZ() / p_vector.module());
  double gamma = atan2(p_vector.getY(), p_vector.getX());

  MathVector<double> rotation_y[3];
  rotation_y[0] = MathVector<double>(cos(beta),0.,sin(beta));
  rotation_y[1] = MathVector<double>(0.,1.,0.);
  rotation_y[2] = MathVector<double>(-sin(beta),0.,cos(beta));

  MathVector<double> rotation_z[3];
  rotation_z[0] = MathVector<double>(cos(gamma),-sin(gamma), 0.);
  rotation_z[1] = MathVector<double>(sin(gamma),cos(gamma), 0.);
  rotation_z[2] = MathVector<double>(0.       ,0.       , 1.);

  MathVector<double> T_rotation_y[3];
  T_rotation_y[0] = MathVector<double>(cos(beta),0.,-sin(beta));
  T_rotation_y[1] = MathVector<double>(0.,1.,0.);
  T_rotation_y[2] = MathVector<double>(sin(beta),0.,cos(beta));


  MathVector<double> T_rotation_z[3];
  T_rotation_z[0] = MathVector<double>(cos(gamma),sin(gamma),  0.);
  T_rotation_z[1] = MathVector<double>(-sin(gamma),cos(gamma), 0.);
  T_rotation_z[2] = MathVector<double>(0.       ,0.       ,  1.);


  double components[3]={0};
  for(int i{0}; i<3 ; i++){
    components[i]=T_rotation_z[i]*p_vector;
  }
  p_vector.setX(components[0]);
  p_vector.setY(components[1]);
  p_vector.setZ(components[2]);


  //rotation_z
  for(int i{0}; i<3 ; i++){
    components[i]=T_rotation_y[i]*p_vector;
  }

  p_vector.setX(components[0]);
  p_vector.setY(components[1]);
  p_vector.setZ(components[2]);
  //end of the first rotation
  //beguining algorithm to scatter in new CS

  double phi = rand*2*std::numbers::pi;
  double cos_xi = 1-2*rand;
  double sin_xi = sqrt(1-pow(cos_xi,2));
  double x{sin_xi * cos(phi)};
  double y{sin_xi * sin(phi)};
  double z{cos_xi};

  p_vector.setX(x*components[0]);
  p_vector.setY(y*components[1]);
  p_vector.setZ(z*components[2]);

  //going back from rotation
 //
 for(int i{0}; i<3 ; i++){
   components[i]=rotation_y[i]*p_vector;
  }
  p_vector.setX(components[0]);
  p_vector.setY(components[1]);
  p_vector.setZ(components[2]);
  //std::cout<<"first inverse rotation on y \t"<<a_vector<<"\n"<<a_vector.module()<<"\n";

  //rotation_z
  for(int i{0}; i<1 ; i++){
    components[i]=rotation_z[i]*p_vector;
  }
  p_vector.setX(components[0]);
  p_vector.setY(components[1]);
  p_vector.setZ(components[2]);

  // Updating velocity vector of the current particle after collision
  // Updated velocity = [directory vector ⋅ (mass_ct ⋅ |old velocity|)] + (old velocity ⋅ mass_cp)
   m_velocity = p_vector + cm_vel;

}

