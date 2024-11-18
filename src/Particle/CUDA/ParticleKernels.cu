#ifdef USE_CUDA

#include <curand_kernel.h>
#include <math_constants.h>

#include "Particle/ParticleKernels.cuh"
#include "Particle/ParticleUtils.hpp"
#include "Utilities/Constants.hpp"

__global__ void updateParticlePositionsKernel(ParticleDevice_t *particles,
                                              size_t count,
                                              double dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    // Update particle positions using velocity and time step
    particles[idx].x += particles[idx].vx * dt;
    particles[idx].y += particles[idx].vy * dt;
    particles[idx].z += particles[idx].vz * dt;
}

__global__ void applyElectroMagneticPushKernel(ParticleDevice_t *particles,
                                               size_t count,
                                               double dt,
                                               double3 electricField,
                                               double3 magneticField)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    ParticleDevice_t &p = particles[idx];

    // Retrieve particle properties
    double charge = ParticleUtils::getChargeFromType(static_cast<ParticleType>(p.type));
    double mass = ParticleUtils::getMassFromType(static_cast<ParticleType>(p.type));

    // Check for zero time step or zero fields
    if (dt == 0.0 || (electricField.x == 0.0 && electricField.y == 0.0 && electricField.z == 0.0 &&
                      magneticField.x == 0.0 && magneticField.y == 0.0 && magneticField.z == 0.0))
    {
        return;
    }

    // Velocity vector
    double3 velocity = make_double3(p.vx, p.vy, p.vz);

    // Calculate Lorentz acceleration: a_L = (q/m)*(E + v x B)
    double3 vCrossB = make_double3(
        velocity.y * magneticField.z - velocity.z * magneticField.y,
        velocity.z * magneticField.x - velocity.x * magneticField.z,
        velocity.x * magneticField.y - velocity.y * magneticField.x);

    double3 a_L = make_double3(
        (charge / mass) * (electricField.x + vCrossB.x),
        (charge / mass) * (electricField.y + vCrossB.y),
        (charge / mass) * (electricField.z + vCrossB.z));

    // First half acceleration step: v_minus = v + a_L * dt/2
    double3 v_minus = make_double3(
        velocity.x + a_L.x * dt / 2.0,
        velocity.y + a_L.y * dt / 2.0,
        velocity.z + a_L.z * dt / 2.0);

    // Compute t vector: t = (q * B * dt) / (2 * m)
    double3 t = make_double3(
        (charge * magneticField.x * dt) / (2.0 * mass),
        (charge * magneticField.y * dt) / (2.0 * mass),
        (charge * magneticField.z * dt) / (2.0 * mass));

    // Compute magnitude squared of t
    double t_mag2 = t.x * t.x + t.y * t.y + t.z * t.z;

    // Compute s factor: s = 2 * t / (1 + t_mag2)
    double s_factor = 2.0 / (1.0 + t_mag2);
    double3 s = make_double3(t.x * s_factor, t.y * s_factor, t.z * s_factor);

    // Rotate velocity: v_prime = v_minus + v_minus x t
    double3 v_prime = make_double3(
        v_minus.x + (v_minus.y * t.z - v_minus.z * t.y),
        v_minus.y + (v_minus.z * t.x - v_minus.x * t.z),
        v_minus.z + (v_minus.x * t.y - v_minus.y * t.x));

    // v_plus = v_minus + v_prime x s
    double3 v_plus = make_double3(
        v_minus.x + (v_prime.y * s.z - v_prime.z * s.y),
        v_minus.y + (v_prime.z * s.x - v_prime.x * s.z),
        v_minus.z + (v_prime.x * s.y - v_prime.y * s.x));

    // Second half acceleration step: v_new = v_plus + a_L * dt/2
    velocity.x = v_plus.x + a_L.x * dt / 2.0;
    velocity.y = v_plus.y + a_L.y * dt / 2.0;
    velocity.z = v_plus.z + a_L.z * dt / 2.0;

    // Update particle velocities
    p.vx = velocity.x;
    p.vy = velocity.y;
    p.vz = velocity.z;

    // Update kinetic energy: E = 0.5 * m * (vx^2 + vy^2 + vz^2)
    p.energy = 0.5 * mass * (velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);
}

__global__ void collideWithGasKernel(ParticleDevice_t *particles,
                                     size_t count,
                                     int gasType,
                                     double gasConcentration,
                                     double timeStep,
                                     int modelType,
                                     double omega,
                                     double alpha,
                                     unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    ParticleDevice_t &p = particles[idx];

    // Initialize curand state
    curandState state;
    curand_init(seed, idx, 0, &state);

    // Get particle properties
    double p_mass = ParticleUtils::getMassFromType(static_cast<ParticleType>(p.type));
    double p_radius = ParticleUtils::getRadiusFromType(static_cast<ParticleType>(p.type));
    double p_velocity = sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);

    // Get gas particle properties
    ParticleType gas_particle_type = static_cast<ParticleType>(gasType);
    double t_mass = ParticleUtils::getMassFromType(gas_particle_type);
    double t_radius = ParticleUtils::getRadiusFromType(gas_particle_type);

    // Compute collision parameters
    double sigma;       // Collision cross-section
    double Probability; // Collision probability

    if (modelType == 0) // HS model
    {
        // HS collision cross-section
        sigma = CUDART_PI * (p_radius + t_radius) * (p_radius + t_radius);

        // Collision probability
        Probability = sigma * p_velocity * gasConcentration * timeStep;
    }
    else if (modelType == 1 || modelType == 2) // VHS or VSS model
    {
        double d_reference = p_radius + t_radius;
        double mass_constant = (p_mass * t_mass) / (p_mass + t_mass);
        double Exponent = omega - 0.5;

        double d_eff_squared = (pow(d_reference, 2) / tgamma(2.5 - omega)) * pow((2 * constants::physical_constants::KT_reference) / (mass_constant * p_velocity * p_velocity), Exponent);

        sigma = CUDART_PI * d_eff_squared;

        Probability = sigma * p_velocity * gasConcentration * timeStep;
    }
    else
    {
        // Invalid model type
        return;
    }

    // Generate random number
    double randVal = curand_uniform(&state);

    // Check if collision occurs
    if (randVal < Probability)
    {
        // Generate random angles
        double xi_cos, xi_sin, phi;

        if (modelType == 2) // VSS model
        {
            // xi_cos = 2 * (rand()^(1/alpha)) - 1
            xi_cos = 2.0 * pow(curand_uniform(&state), 1.0 / alpha) - 1.0;
        }
        else // HS and VHS models
        {
            xi_cos = 2.0 * curand_uniform(&state) - 1.0; // Random value between -1 and 1
        }
        xi_sin = sqrt(1.0 - xi_cos * xi_cos);
        phi = 2.0 * CUDART_PI * curand_uniform(&state); // Random value between 0 and 2π

        // Compute new velocity components
        double x = xi_sin * cos(phi);
        double y = xi_sin * sin(phi);
        double z = xi_cos;

        double mass_cp = p_mass / (t_mass + p_mass);
        double mass_ct = t_mass / (t_mass + p_mass);

        // Center-of-mass velocity
        double cm_vx = p.vx * mass_cp;
        double cm_vy = p.vy * mass_cp;
        double cm_vz = p.vz * mass_cp;

        // Particle relative velocity magnitude
        double mp = p_velocity * mass_ct;

        // New velocity direction
        double dir_vx = x * mp;
        double dir_vy = y * mp;
        double dir_vz = z * mp;

        // Rotate velocity for VSS model
        if (modelType == 2)
        {
            double3 p_vec = make_double3(p.vx, p.vy, p.vz);

            // Compute angles beta and gamma from p_vec
            double beta = atan2(p_vec.y, p_vec.x); // Beta angle
            double gamma = acos(p_vec.z / mp);     // Gamma angle

            // Create direction vector
            double dir_vx = x * mp;
            double dir_vy = y * mp;
            double dir_vz = z * mp;

            // Rotate dir_vector by angles beta and gamma
            // Rotation around z-axis by beta
            double tmp_vx = dir_vx * cos(beta) - dir_vy * sin(beta);
            double tmp_vy = dir_vx * sin(beta) + dir_vy * cos(beta);
            double tmp_vz = dir_vz;

            // Rotation around y-axis by gamma
            double rotated_vx = tmp_vx * cos(gamma) + tmp_vz * sin(gamma);
            double rotated_vy = tmp_vy;
            double rotated_vz = -tmp_vx * sin(gamma) + tmp_vz * cos(gamma);

            // Update particle velocity
            p.vx = rotated_vx + cm_vx;
            p.vy = rotated_vy + cm_vy;
            p.vz = rotated_vz + cm_vz;
        }
        else
        {
            // For HS and VHS models, update particle velocity without rotation
            p.vx = dir_vx + cm_vx;
            p.vy = dir_vy + cm_vy;
            p.vz = dir_vz + cm_vz;
        }

        // Update particle velocity
        p.vx = dir_vx + cm_vx;
        p.vy = dir_vy + cm_vy;
        p.vz = dir_vz + cm_vz;

        // Update kinetic energy
        p.energy = 0.5 * p_mass * (p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);
    }
}

__global__ void detectCollisionsWithMeshKernel(
    ParticleDevice_t *particles,
    size_t particleCount,
    const AABBNodeDevice_t *nodes,
    size_t nodeCount,
    const TriangleDevice_t *triangles,
    float dt)
{
    // Determine particle index for the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount)
        return;

    ParticleDevice_t &p = particles[idx];

    // Calculate ray direction based on particle velocity and time step
    Vec3Device_t rayOrigin{p.x, p.y, p.z};
    Vec3Device_t rayDir = {p.vx * dt, p.vy * dt, p.vz * dt};

    // Check if there is movement
    float dirLength = sqrtf(rayDir.x * rayDir.x + rayDir.y * rayDir.y + rayDir.z * rayDir.z);
    if (dirLength < 1e-8f)
        return; // Skip if particle is not moving

    // Normalize ray direction and calculate its inverse
    Vec3Device_t rayDirNorm = {rayDir.x / dirLength, rayDir.y / dirLength, rayDir.z / dirLength};
    Vec3Device_t rayDirInv = {1.0f / rayDirNorm.x, 1.0f / rayDirNorm.y, 1.0f / rayDirNorm.z};

    // Initialize stack for AABB tree traversal
    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0; // Start with the root node

    bool collisionDetected = false;
    float minT = dirLength; // Maximum allowable distance for particle movement
    Vec3Device_t collisionPoint;

    while (stackPtr > 0)
    {
        int nodeIdx = stack[--stackPtr];
        if (nodeIdx >= nodeCount)
            continue;

        const AABBNodeDevice_t &node = nodes[nodeIdx];

        // Check for intersection between ray and the node's AABB
        if (!intersectRayAABB(rayOrigin, rayDirInv, node.bbox))
            continue;

        if (node.left == -1 && node.right == -1)
        {
            // Leaf node: check for intersection with the triangle
            int triIdx = node.triangleIdx;
            const TriangleDevice_t &tri = triangles[triIdx];
            float t;
            if (intersectRayTriangle(rayOrigin, rayDirNorm, tri, t))
            {
                if (t >= 0.0f && t <= minT)
                {
                    // Update minimum distance to collision
                    minT = t;
                    collisionDetected = true;

                    // Calculate collision point
                    collisionPoint.x = rayOrigin.x + rayDirNorm.x * t;
                    collisionPoint.y = rayOrigin.y + rayDirNorm.y * t;
                    collisionPoint.z = rayOrigin.z + rayDirNorm.z * t;
                }
            }
        }
        else
        {
            // Internal node: push child nodes onto the stack
            if (node.left != -1)
                stack[stackPtr++] = node.left;
            if (node.right != -1)
                stack[stackPtr++] = node.right;
        }
    }

    if (collisionDetected)
    {
        // Process collision
        // Update particle position to the collision point
        p.x = collisionPoint.x;
        p.y = collisionPoint.y;
        p.z = collisionPoint.z;

        // Reflect particle velocity (simple stop behavior)
        // Modify velocity handling as needed for more complex collision response
        p.vx = 0.0f;
        p.vy = 0.0f;
        p.vz = 0.0f;
    }
    else
    {
        // No collision detected: update position for full time step
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.z += p.vz * dt;
    }
}

#endif // !USE_CUDA
