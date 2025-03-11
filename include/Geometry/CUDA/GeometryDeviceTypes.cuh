#ifndef GEOMETRYDEVICETYPES_CUH
#define GEOMETRYDEVICETYPES_CUH

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include "Utilities/PreprocessorUtils.hpp"
#include "Utilities/CUDAWarningSuppress.hpp"

/**
 * @brief Device representation of a 3D point
 */
struct DevicePoint
{
    double x, y, z;
    
    START_CUDA_HOST_DEVICE DevicePoint() noexcept : x(0.0), y(0.0), z(0.0) {}
    START_CUDA_HOST_DEVICE DevicePoint(double x_, double y_, double z_) noexcept : x(x_), y(y_), z(z_) {}
    
    /**
     * @brief Calculate the squared distance to another point
     */
    START_CUDA_HOST_DEVICE double squared_distance(DevicePoint const& other) const noexcept
    {
        double dx = x - other.x;
        double dy = y - other.y;
        double dz = z - other.z;
        return dx * dx + dy * dy + dz * dz;
    }
    
    /**
     * @brief Calculate the distance to another point
     */
    START_CUDA_HOST_DEVICE double distance(DevicePoint const& other) const noexcept
    {
        return sqrt(squared_distance(other));
    }
};

/**
 * @brief Device representation of a triangle
 */
struct DeviceTriangle
{
    DevicePoint v0, v1, v2;
    
    START_CUDA_HOST_DEVICE DeviceTriangle() noexcept = default;
    START_CUDA_HOST_DEVICE DeviceTriangle(DevicePoint const& v0_, DevicePoint const& v1_, DevicePoint const& v2_) noexcept
        : v0(v0_), v1(v1_), v2(v2_) {}
    
    /**
     * @brief Calculate the normal vector of the triangle
     */
    START_CUDA_HOST_DEVICE DevicePoint normal() const noexcept
    {
        // Calculate two edges
        double e1x = v1.x - v0.x;
        double e1y = v1.y - v0.y;
        double e1z = v1.z - v0.z;
        
        double e2x = v2.x - v0.x;
        double e2y = v2.y - v0.y;
        double e2z = v2.z - v0.z;
        
        // Cross product
        double nx = e1y * e2z - e1z * e2y;
        double ny = e1z * e2x - e1x * e2z;
        double nz = e1x * e2y - e1y * e2x;
        
        // Normalize
        double len = sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 1e-10) {
            nx /= len;
            ny /= len;
            nz /= len;
        }
        
        return DevicePoint(nx, ny, nz);
    }
    
    /**
     * @brief Calculate the area of the triangle
     */
    START_CUDA_HOST_DEVICE double area() const noexcept
    {
        // Calculate two edges
        double e1x = v1.x - v0.x;
        double e1y = v1.y - v0.y;
        double e1z = v1.z - v0.z;
        
        double e2x = v2.x - v0.x;
        double e2y = v2.y - v0.y;
        double e2z = v2.z - v0.z;
        
        // Cross product magnitude is twice the area
        double cx = e1y * e2z - e1z * e2y;
        double cy = e1z * e2x - e1x * e2z;
        double cz = e1x * e2y - e1y * e2x;
        
        return 0.5 * sqrt(cx * cx + cy * cy + cz * cz);
    }
};

/**
 * @brief Device representation of a ray (line segment)
 */
struct DeviceRay
{
    DevicePoint origin;
    DevicePoint direction;
    double length;
    
    START_CUDA_HOST_DEVICE DeviceRay() noexcept : length(0.0) {}
    
    START_CUDA_HOST_DEVICE DeviceRay(DevicePoint const& origin_, DevicePoint const& target) noexcept
        : origin(origin_)
    {
        direction.x = target.x - origin.x;
        direction.y = target.y - origin.y;
        direction.z = target.z - origin.z;
        
        length = sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
        
        // Normalize direction if length is not too small
        if (length > 1e-10) {
            direction.x /= length;
            direction.y /= length;
            direction.z /= length;
        }
    }
    
    /**
     * @brief Check if the ray is degenerate (too short)
     */
    START_CUDA_HOST_DEVICE bool is_degenerate() const noexcept
    {
        return length < 1e-10;
    }
    
    /**
     * @brief Get a point at a certain distance along the ray
     */
    START_CUDA_HOST_DEVICE DevicePoint point_at_distance(double distance) const noexcept
    {
        return DevicePoint(
            origin.x + direction.x * distance,
            origin.y + direction.y * distance,
            origin.z + direction.z * distance
        );
    }
};

/**
 * @brief Device representation of a tetrahedron
 */
struct DeviceTetrahedron
{
    DevicePoint v0, v1, v2, v3;
    
    START_CUDA_HOST_DEVICE DeviceTetrahedron() noexcept = default;
    START_CUDA_HOST_DEVICE DeviceTetrahedron(DevicePoint const& v0_, DevicePoint const& v1_, DevicePoint const& v2_, DevicePoint const& v3_) noexcept
        : v0(v0_), v1(v1_), v2(v2_), v3(v3_) {}
    
    /**
     * @brief Check if a point is inside this tetrahedron
     */
    START_CUDA_HOST_DEVICE bool contains_point(DevicePoint const& p) const noexcept
    {
        // This is a simplified implementation
        // For a real implementation, use barycentric coordinates or similar method
        
        // Calculate normals for each face pointing outward
        DevicePoint n0 = DeviceTriangle(v0, v2, v1).normal(); // Face 0: v0, v2, v1
        DevicePoint n1 = DeviceTriangle(v0, v1, v3).normal(); // Face 1: v0, v1, v3
        DevicePoint n2 = DeviceTriangle(v0, v3, v2).normal(); // Face 2: v0, v3, v2
        DevicePoint n3 = DeviceTriangle(v1, v2, v3).normal(); // Face 3: v1, v2, v3
        
        // Check if point is on the correct side of all planes
        // Direction from face to the opposite vertex
        DevicePoint d0(v3.x - v0.x, v3.y - v0.y, v3.z - v0.z);
        DevicePoint d1(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
        DevicePoint d2(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        DevicePoint d3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
        
        // Ensure normals point outward
        double dot0 = d0.x * n0.x + d0.y * n0.y + d0.z * n0.z;
        if (dot0 > 0) { n0.x = -n0.x; n0.y = -n0.y; n0.z = -n0.z; }
        
        double dot1 = d1.x * n1.x + d1.y * n1.y + d1.z * n1.z;
        if (dot1 > 0) { n1.x = -n1.x; n1.y = -n1.y; n1.z = -n1.z; }
        
        double dot2 = d2.x * n2.x + d2.y * n2.y + d2.z * n2.z;
        if (dot2 > 0) { n2.x = -n2.x; n2.y = -n2.y; n2.z = -n2.z; }
        
        double dot3 = d3.x * n3.x + d3.y * n3.y + d3.z * n3.z;
        if (dot3 > 0) { n3.x = -n3.x; n3.y = -n3.y; n3.z = -n3.z; }
        
        // Test if point is on correct side of all planes
        DevicePoint v0p(p.x - v0.x, p.y - v0.y, p.z - v0.z);
        DevicePoint v1p(p.x - v1.x, p.y - v1.y, p.z - v1.z);
        
        double side0 = v0p.x * n0.x + v0p.y * n0.y + v0p.z * n0.z;
        double side1 = v0p.x * n1.x + v0p.y * n1.y + v0p.z * n1.z;
        double side2 = v0p.x * n2.x + v0p.y * n2.y + v0p.z * n2.z;
        double side3 = v1p.x * n3.x + v1p.y * n3.y + v1p.z * n3.z;
        
        return side0 <= 0 && side1 <= 0 && side2 <= 0 && side3 <= 0;
    }
};

#endif // USE_CUDA

#endif // GEOMETRYDEVICETYPES_CUH 