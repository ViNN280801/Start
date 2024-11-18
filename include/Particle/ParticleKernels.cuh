#ifndef PARTICLEKERNELS_CUH
#define PARTICLEKERNELS_CUH

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include "Geometry/Intersections.cuh"
#include "Particle/ParticleDevice.cuh"

/**
 * @brief CUDA kernel for updating the position of particles.
 *
 * This kernel computes new positions of particles over a specified time interval (`dt`).
 * Each thread is responsible for updating a single particle's position in the simulation.
 *
 * The update includes:
 * 1. **Position Update:** Calculates the new x, y, z coordinates based on the particle's velocity
 *    components (vx, vy, vz) and the time step (dt) using the formula:
 *      \f[
 *      x_{new} = x + vx \cdot dt
 *      \f]
 *    Similar calculations are applied for y and z components.
 * 2. **Energy Conservation Check:** Maintains conservation of kinetic energy, recalculating energy
 *    based on updated velocities if needed.
 *
 * Memory Management:
 * - The function assumes particle data (positions, velocities, etc.) is already loaded onto the device.
 * - Synchronization is handled using block barriers to ensure all threads in a block have completed
 *   their calculations before the next operation begins.
 *
 * @param particles Device array of ParticleDevice_t structures, representing particles in the simulation.
 * @param count The number of particles to process.
 * @param dt The time interval for the update step.
 */
__global__ void updateParticlePositionsKernel(ParticleDevice_t *particles,
                                              size_t count,
                                              double dt);

/**
 * @brief CUDA kernel for applying electromagnetic push to particles.
 *
 * This kernel updates the velocities of particles based on the Lorentz force due to electric and magnetic fields
 * over a specified time interval (`dt`). It uses the Boris integrator algorithm for stable and accurate integration
 * of particle motion in electromagnetic fields.
 *
 * Each thread processes a single particle, performing the following steps:
 * 1. **Calculate Lorentz Acceleration:** Computes the acceleration due to electric and magnetic fields:
 *      \f[
 *      \mathbf{a}_L = \frac{q}{m} \left( \mathbf{E} + \mathbf{v} \times \mathbf{B} \right)
 *      \f]
 *    where \( q \) is the particle's charge, \( m \) is its mass, \( \mathbf{v} \) is its velocity,
 *    \( \mathbf{E} \) is the electric field, and \( \mathbf{B} \) is the magnetic field.
 * 2. **First Half Acceleration Step:** Updates the velocity half a time step using the Lorentz acceleration:
 *      \f[
 *      \mathbf{v}^{-} = \mathbf{v} + \mathbf{a}_L \frac{\Delta t}{2}
 *      \f]
 * 3. **Rotation Step (Boris Algorithm):**
 *      - Compute the rotation vector:
 *          \f[
 *          \mathbf{t} = \frac{q \mathbf{B} \Delta t}{2m}
 *          \f]
 *      - Compute the magnitude squared of \( \mathbf{t} \):
 *          \f[
 *          t_{\text{mag}}^2 = t_x^2 + t_y^2 + t_z^2
 *          \f]
 *      - Compute the scaling factor \( s \):
 *          \f[
 *          s = \frac{2}{1 + t_{\text{mag}}^2}
 *          \f]
 *      - Rotate the velocity:
 *          \f[
 *          \mathbf{v}^{\prime} = \mathbf{v}^{-} + \mathbf{v}^{-} \times \mathbf{t}
 *          \f]
 *          \f[
 *          \mathbf{v}^{+} = \mathbf{v}^{-} + \mathbf{v}^{\prime} \times \mathbf{s}
 *          \f]
 * 4. **Second Half Acceleration Step:** Updates the velocity another half time step:
 *      \f[
 *      \mathbf{v}_{\text{new}} = \mathbf{v}^{+} + \mathbf{a}_L \frac{\Delta t}{2}
 *      \f]
 * 5. **Update Kinetic Energy:** Recalculates the kinetic energy based on the updated velocity:
 *      \f[
 *      E_{\text{kinetic}} = \frac{1}{2} m \left( v_x^2 + v_y^2 + v_z^2 \right)
 *      \f]
 *
 * Memory Management:
 * - Assumes that the particles array is allocated on the device.
 * - Requires synchronization if subsequent operations depend on the updated velocities.
 *
 * @param particles Device array of ParticleDevice_t structures.
 * @param count The number of particles to process.
 * @param dt The time interval for the update step.
 * @param electricField The electric field vector \( \mathbf{E} \) applied to all particles.
 * @param magneticField The magnetic field vector \( \mathbf{B} \) applied to all particles.
 */
__global__ void applyElectroMagneticPushKernel(ParticleDevice_t *particles,
                                               size_t count,
                                               double dt,
                                               double3 electricField,
                                               double3 magneticField);

/**
 * @brief CUDA kernel for simulating collisions of particles with gas.
 *
 * This kernel processes collisions between particles and a background gas using specified collision models:
 * HS (Hard Sphere), VHS (Variable Hard Sphere), or VSS (Variable Soft Sphere).
 *
 * Each thread handles one particle, performing the following steps:
 * 1. **Calculate Collision Probability:** Based on the collision cross-section, particle velocity, gas concentration, and time step.
 * 2. **Generate Random Number:** Using curand to determine if a collision occurs.
 * 3. **Update Velocity if Collision Occurs:** Modify the particle's velocity according to the collision model.
 * 4. **Update Kinetic Energy:** Recalculate the kinetic energy based on the new velocity.
 *
 * @param particles Device array of ParticleDevice_t structures.
 * @param count The number of particles to process.
 * @param gasType The type of gas particles (ParticleType).
 * @param gasConcentration The concentration of gas particles.
 * @param timeStep The simulation time step.
 * @param modelType The collision model to use: 0 for HS, 1 for VHS, 2 for VSS.
 * @param omega The viscosity temperature index (used in VHS and VSS).
 * @param alpha The deflection parameter (used in VSS).
 * @param seed Seed for random number generation.
 */
__global__ void collideWithGasKernel(ParticleDevice_t *particles,
                                     size_t count,
                                     int gasType,
                                     double gasConcentration,
                                     double timeStep,
                                     int modelType,
                                     double omega,
                                     double alpha,
                                     unsigned long long seed);

/**
 * @brief CUDA kernel for detecting collisions of particles with a triangular mesh using an AABB tree.
 *
 * Each thread processes a single particle, checks for potential collisions with
 * triangles in an AABB tree structure, and updates the particle position and velocity accordingly.
 *
 * @param particles Pointer to an array of particles on the device.
 * @param particleCount Total number of particles.
 * @param nodes Pointer to the AABB tree nodes on the device.
 * @param nodeCount Total number of nodes in the AABB tree.
 * @param triangles Pointer to the array of triangles on the device.
 * @param dt Time step for the simulation.
 */
__global__ void detectCollisionsWithMeshKernel(
    ParticleDevice_t *particles,
    size_t particleCount,
    const AABBNodeDevice_t *nodes,
    size_t nodeCount,
    const TriangleDevice_t *triangles,
    float dt);

#endif // !USE_CUDA

#endif // PARTICLEKERNELS_CUH
