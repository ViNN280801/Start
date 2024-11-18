#ifndef GEOMETRYTYPES_CUH
#define GEOMETRYTYPES_CUH

#ifdef USE_CUDA
#include "Utilities/PreprocessorUtils.hpp"

/// @brief Structure representing a 3D point or vector.
struct Vec3Device_t
{
    double x; ///< Position by X-axis in Cartesian system.
    double y; ///< Position by Y-axis in Cartesian system.
    double z; ///< Position by Z-axis in Cartesian system.

    START_CUDA_HOST_DEVICE Vec3Device_t() : x(0.0), y(0.0), z(0.0) {}
    START_CUDA_HOST_DEVICE Vec3Device_t(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    START_CUDA_HOST_DEVICE Vec3Device_t(Vec3Device_t const &other) : x(other.x), y(other.y), z(other.z) {}
    START_CUDA_HOST_DEVICE Vec3Device_t &operator=(Vec3Device_t const &other)
    {
        if (this != &other)
        {
            x = other.x;
            y = other.y;
            z = other.z;
        }
        return *this;
    }

    START_CUDA_HOST_DEVICE Vec3Device_t(Vec3Device_t &&other) noexcept : x(other.x), y(other.y), z(other.z) {}
    START_CUDA_HOST_DEVICE Vec3Device_t &operator=(Vec3Device_t &&other) noexcept
    {
        if (this != &other)
        {
            x = other.x;
            y = other.y;
            z = other.z;
        }
        return *this;
    }
};

/// @brief Structure representing a triangle in 3D space.
struct TriangleDevice_t
{
    Vec3Device_t v0; ///< First vertex of the triangle.
    Vec3Device_t v1; ///< Second vertex of the triangle.
    Vec3Device_t v2; ///< Third vertex of the triangle.

    /// @brief Default ctor.
    START_CUDA_HOST_DEVICE TriangleDevice_t() {}

    /// @brief Ctor with vertices.
    START_CUDA_HOST_DEVICE TriangleDevice_t(Vec3Device_t const &v0_, Vec3Device_t const &v1_, Vec3Device_t const &v2_)
        : v0(v0_), v1(v1_), v2(v2_) {}

    /// @brief Copy ctor.
    START_CUDA_HOST_DEVICE TriangleDevice_t(TriangleDevice_t const &other)
        : v0(other.v0), v1(other.v1), v2(other.v2) {}

    /// @brief Copy assignment operator.
    START_CUDA_HOST_DEVICE TriangleDevice_t &operator=(TriangleDevice_t const &other)
    {
        if (this != &other)
        {
            v0 = other.v0;
            v1 = other.v1;
            v2 = other.v2;
        }
        return *this;
    }

    /// @brief Move ctor.
    START_CUDA_HOST_DEVICE TriangleDevice_t(TriangleDevice_t &&other) noexcept
        : v0(std::move(other.v0)), v1(std::move(other.v1)), v2(std::move(other.v2)) {}

    /// @brief Move assignment operator.
    START_CUDA_HOST_DEVICE TriangleDevice_t &operator=(TriangleDevice_t &&other) noexcept
    {
        if (this != &other)
        {
            v0 = std::move(other.v0);
            v1 = std::move(other.v1);
            v2 = std::move(other.v2);
        }
        return *this;
    }
};
#endif // !USE_CUDA

#endif // !GEOMETRYTYPES_CUH
