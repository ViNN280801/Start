#ifndef GEOMETRY_VECTOR_HPP
#define GEOMETRY_VECTOR_HPP

#include <compare>
#include <random>
#include <stdexcept>

#include "Geometry/GeometryExceptions.hpp"
#include "Utilities/Utilities.hpp"

#if __cplusplus >= 202002L
#include <concepts>

/**
 * @brief Concept that enforces the template type to be either integral or floating-point.
 * @tparam GeometryVectorComponent The type to be checked.
 */
template <typename GeometryVectorComponent>
concept Numeric = std::integral<GeometryVectorComponent> || std::floating_point<GeometryVectorComponent>;

#else
#include <type_traits>

/**
 * @brief Type trait that checks if the template type is either integral or floating-point.
 * @tparam GeometryVectorComponent The type to be checked.
 */
template <typename GeometryVectorComponent>
struct is_numeric : std::integral_constant<bool, std::is_integral<GeometryVectorComponent>::value ||
                                                     std::is_floating_point<GeometryVectorComponent>::value>
{
};

/**
 * @brief Helper to simplify checking if the type is numeric.
 */
template <typename GeometryVectorComponent>
inline constexpr bool is_numeric_v = is_numeric<GeometryVectorComponent>::value;
#endif

/**
 * Description of the mathematical vector. In the simplest case, a
 * mathematical object characterized by magnitude and direction.
 *               ->  | x |
 * @brief GeometryVector P = | y |
 *                   | z |
 */
#if __cplusplus >= 202002L
template <Numeric T>
#else
template <typename T, typename = std::enable_if_t<is_numeric_v<T>>>
#endif
class GeometryVector
{
private:
    T x{}; ///< X-coordinate of the vector.
    T y{}; ///< Y-coordinate of the vector.
    T z{}; ///< Z-coordinate of the vector.

    /**
     * @brief Helper func. Rotates the vector around the y-axis.
     * @details The rotation is performed by multiplying the current vector
     *          with the rotation matrix for the y-axis.
     * @param beta The angle of rotation in radians.
     */
    void _rotate_y(T beta)
    {
        T cosBeta{std::cos(beta)}, sinBeta{std::sin(beta)},
            tempX{cosBeta * x + sinBeta * z},
            tempZ{-sinBeta * x + cosBeta * z};

        x = tempX;
        z = tempZ;
    }

    /**
     * @brief Helper func. Rotates the vector around the z-axis.
     * @details The rotation is performed by multiplying the current vector
     *          with the rotation matrix for the z-axis.
     * @param gamma The angle of rotation in radians.
     */
    void _rotate_z(T gamma)
    {
        T cosGamma{std::cos(gamma)}, sinGamma{std::sin(gamma)},
            tempX{cosGamma * x - sinGamma * y},
            tempY{sinGamma * x + cosGamma * y};

        x = tempX;
        y = tempY;
    }

public:
    /// @brief Default constructor. Initializes vector components to zero.
    GeometryVector() = default;

    /**
     * @brief Constructor that initializes the vector with specific values for x, y, and z.
     * @param x_ X-coordinate to assign.
     * @param y_ Y-coordinate to assign.
     * @param z_ Z-coordinate to assign.
     */
    START_CUDA_HOST_DEVICE GeometryVector(T x_, T y_, T z_) : x{x_}, y{y_}, z{z_} {}

    /**
     * @brief Assignment operator with custom double.
     *        Sets all components of vector to custom value.
     * @param value The value to assign to all components.
     * @return Reference to this vector.
     */
    GeometryVector &operator=(T value)
    {
        x = value;
        y = value;
        z = value;
        return *this;
    }

    /**
     * @brief Fills `GeometryVector` object with specified values.
     * @param x_ X-coordinate of the point.
     * @param y_ Y-coordinate of the point.
     * @param z_ Z-coordinate of the point.
     * @return Filled structure of the Cartesian coordinates.
     */
    static GeometryVector createCoordinates(T x_, T y_, T z_)
    {
        GeometryVector vec;

        vec.x = x_;
        vec.y = y_;
        vec.z = z_;

        return vec;
    }

    /**
     * @brief Creates a random GeometryVector with components in the range [from, to].
     * @tparam T The type of the vector components (must be integral or floating-point).
     * @param from The lower bound of the range for random values.
     * @param to The upper bound of the range for random values.
     * @return A randomly generated GeometryVector with components in the specified range.
     *
     * @details
     * This function uses a random number generator to create a GeometryVector where each
     * component is a random value within the given range. The type of the random distribution
     * (real or integer) depends on the type of the vector components.
     *
     * @note
     * - For floating-point types, uses std::uniform_real_distribution.
     * - For integral types, uses std::uniform_int_distribution.
     *
     * @exception noexcept
     * This function does not throw exceptions.
     *
     * @thread_safety
     * - **Thread-safe** due to the use of thread_local random number generator.
     */
    static GeometryVector createRandomVector(T const &from, T const &to) noexcept
    {
        thread_local std::random_device rd;
        thread_local std::mt19937 gen(rd());

        if constexpr (std::is_floating_point_v<T>)
        {
            std::uniform_real_distribution<T> dis(from, to);
            return GeometryVector(dis(gen), dis(gen), dis(gen));
        }
        else if constexpr (std::is_integral_v<T>)
        {
            std::uniform_int_distribution<T> dis(from, to);
            return GeometryVector(dis(gen), dis(gen), dis(gen));
        }
    }

    /* === Getters for each component. === */
    START_CUDA_HOST_DEVICE constexpr T getX() const { return x; }
    START_CUDA_HOST_DEVICE constexpr T getY() const { return y; }
    START_CUDA_HOST_DEVICE constexpr T getZ() const { return z; }

    /* === Setters for each component. === */
    constexpr void setX(T x_) { x = x_; }
    constexpr void setY(T y_) { y = y_; }
    constexpr void setZ(T z_) { z = z_; }

    /**
     * @brief Fills the vector with specified values for x, y, and z.
     * @param x_ X-coordinate.
     * @param y_ Y-coordinate.
     * @param z_ Z-coordinate.
     */
    void setXYZ(T x_, T y_, T z_)
    {
        x = x_;
        y = y_;
        z = z_;
    }

    /// @brief Calculates the module of the vector.
    STARTCONSTEXPRFUNC T module() const { return std::sqrt(x * x + y * y + z * z); }

    /// @brief Calculates the distance between two vectors.
    T distance(GeometryVector const &other) const { return std::sqrt(std::pow(other.x - x, 2) +
                                                                     std::pow(other.y - y, 2) +
                                                                     std::pow(other.z - z, 2)); }
    T distance(GeometryVector &&other) const { return std::sqrt(std::pow(other.x - x, 2) +
                                                                std::pow(other.y - y, 2) +
                                                                std::pow(other.z - z, 2)); }

    /// @brief Clears the vector (Sets all components to null).
    void clear() & noexcept { *this = T{}; }

    /**
     * @brief Checker for empty vector (are all values null).
     * @return `true` if vector is null, otherwise `false`.
     */
    [[nodiscard]] constexpr bool isNull() const { return (x == 0 && y == 0 && z == 0); }

    /**
     * @brief Checks if vectors are parallel.
     *        \( a \) is parallel to \( b \) if \( a = k \cdot b \) or \( b = k \cdot a \) for some scalar \( k \).
     * @return `true` if vectors are parallel, otherwise `false`.
     */
    bool isParallel(GeometryVector const &other) const
    {
        T koef{x / other.x};
        return (y == koef * other.y) && (z == koef * other.z);
    }
    bool isParallel(GeometryVector &&other) const
    {
        T koef{x / other.x};
        return (y == koef * other.y) && (z == koef * other.z);
    }

    /**
     * @brief Checks if vectors are orthogonal.
     * `a` is orthogonal to `b` if their dot (scalar) product is equals to 0.
     * @return `true` if vectors are orthogonal, otherwise `false`.
     */
    bool isOrthogonal(GeometryVector const &other) const { return dotProduct(other) == 0; }
    bool isOrthogonal(GeometryVector &&other) const { return dotProduct(std::move(other)) == 0; }

    /**
     * @brief Calculates the area of a triangle given its three vertices.
     * @details This function computes the area of a triangle in a 2D space using the vertices A, B, and C.
     *          The formula used is the absolute value of half the determinant of a 2x2 matrix formed by
     *          subtracting the coordinates of A from those of B and C. This method is efficient and
     *          works well for triangles defined in a Cartesian coordinate system.
     * @param A The first vertex of the triangle, represented as a GeometryVector.
     * @param B The second vertex of the triangle, represented as a GeometryVector.
     * @param C The third vertex of the triangle, represented as a GeometryVector.
     * @return The area of the triangle as a double value.
     */
    static T calculateTriangleArea(GeometryVector const &A,
                                   GeometryVector const &B,
                                   GeometryVector const &C)
    {
        return std::fabs((B.getX() - A.getX()) * (C.getY() - A.getY()) -
                         (B.getY() - A.getY()) * (C.getX() - A.getX())) /
               2.0;
    }
    static T calculateTriangleArea(GeometryVector &&A,
                                   GeometryVector &&B,
                                   GeometryVector &&C)
    {
        return std::fabs((B.getX() - A.getX()) * (C.getY() - A.getY()) -
                         (B.getY() - A.getY()) * (C.getX() - A.getX())) /
               2.0;
    }

    T &operator[](int k)
    {
        switch (k)
        {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        default:
            START_THROW_EXCEPTION(GeometryIndexOutOfRangeException,
                                  util::stringify("Requested index ", k, " for GeometryVector is out of range"));
        }
    }

    T const &operator[](int k) const
    {
        switch (k)
        {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        default:
            START_THROW_EXCEPTION(GeometryIndexOutOfRangeException,
                                  util::stringify("Requested index ",
                                                  k, " for GeometryVector is out of range. Vector is 3D."));
        }
    }

    T &operator()(int k)
    {
        if (k < 0 || k > 2)
            START_THROW_EXCEPTION(GeometryIndexOutOfRangeException,
                                  util::stringify("Requested index ",
                                                  k, " for GeometryVector is out of range. Vector is 3D."));

        return (*this)[k];
    }

    T const &operator()(int k) const
    {
        if (k < 0 || k > 2)
            START_THROW_EXCEPTION(GeometryIndexOutOfRangeException,
                                  util::stringify("Requested index ",
                                                  k, " for GeometryVector is out of range. Vector is 3D."));
        return (*this)[k];
    }

    T &at(int k) { return (*this)(k); }

    T const &at(int k) const { return (*this)(k); }

    /// @brief Overload of unary minus. Negates all components of vector.
    GeometryVector operator-() { return GeometryVector(-x, -y, -z); }

    /* +++ Subtract and sum of two vectors correspondingly. +++ */
    GeometryVector operator-(GeometryVector const &other) const { return GeometryVector(x - other.x, y - other.y, z - other.z); }
    GeometryVector operator+(GeometryVector const &other) const { return GeometryVector(x + other.x, y + other.y, z + other.z); }

    /* +++ Subtract/sum/product-assign operators for scalar and for the other vector. +++ */
    GeometryVector &operator+=(GeometryVector const &other)
    {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    GeometryVector &operator-=(GeometryVector const &other)
    {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    GeometryVector &operator+=(T value)
    {
        x += value;
        y += value;
        z += value;
        return *this;
    }

    GeometryVector &operator-=(T value)
    {
        x -= value;
        y -= value;
        z -= value;
        return *this;
    }

    GeometryVector &operator*=(T value)
    {
        x *= value;
        y *= value;
        z *= value;
        return *this;
    }

    GeometryVector &operator*=(GeometryVector const &other)
    {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        return *this;
    }

    /* +++ Subtract and sum of value to vector. +++ */
    GeometryVector operator-(T value) const { return GeometryVector(x - value, y - value, z - value); }
    GeometryVector operator+(T value) const { return GeometryVector(x + value, y + value, z + value); }
    friend GeometryVector operator+(T value, GeometryVector const &other) { return GeometryVector(other.x + value, other.y + value, other.z + value); }

    /* *** Scalar and vector multiplication correspondingly. *** */
    GeometryVector operator*(T value) const { return GeometryVector(x * value, y * value, z * value); }
    friend GeometryVector operator*(T value, GeometryVector const &other) { return GeometryVector(other.x * value, other.y * value, other.z * value); }
    T operator*(GeometryVector const &other) const { return (x * other.x + y * other.y + z * other.z); }
    T operator*(GeometryVector &&other) const { return (x * other.x + y * other.y + z * other.z); }
    T dotProduct(GeometryVector const &other) const { return (*this) * other; }
    T dotProduct(GeometryVector &&other) const { return (*this) * other; }
    GeometryVector crossProduct(GeometryVector const &other) const { return GeometryVector(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x); }
    GeometryVector crossProduct(GeometryVector &&other) const { return GeometryVector(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x); }

    /* /// Division operator. GeometryVector / value. \\\ */
    GeometryVector operator/(T value) const
    {
        if (value == 0)
            START_THROW_EXCEPTION(GeometryDivisionByZeroException,
                                  util::stringify("Division by null: Elements of vector can't be divided by 0",
                                                  "\nVector: ", getX(), " ", getY(), " ", getZ(),
                                                  "\nValue: ", value));
        return GeometryVector(x / value, y / value, z / value);
    }

    /* <=> Comparison operators. <=> */
    constexpr bool operator==(GeometryVector const &other) const { return x == other.x && y == other.y && z == other.z; }
    constexpr bool operator!=(GeometryVector const &other) const { return !(*this == other); }

    /* << Output stream operator. << */
    friend std::ostream &operator<<(std::ostream &os, GeometryVector const &vector)
    {
        os << vector.x << ' ' << vector.y << ' ' << vector.z;
        return os;
    }

    /* >> Input stream operator. >> */
    friend std::istream &operator>>(std::istream &is, GeometryVector &vector)
    {
        is >> vector.x >> vector.y >> vector.z;
        return is;
    }

    /**
     * @brief Normalize the vector.
     *
     * This function normalizes the vector, which means it scales the vector
     * to have a magnitude of 1 while preserving its direction.
     *
     * @return A normalized vector with the same direction but a magnitude of 1.
     *
     * @note If the vector is a zero vector (magnitude is zero), this function
     * will return a zero vector as well to avoid division by zero.
     */
    GeometryVector normalize() const
    {
        T magnitude{module()};
        return magnitude != 0 ? GeometryVector{x / magnitude, y / magnitude, z / magnitude} : GeometryVector{};
    }

    /**
     * @brief Calculates the rotation angles required to align the vector with the Z-axis.
     * @return A pair of angles (beta, gamma) in radians.
     * @throws std::runtime_error If the vector is near-zero or exactly zero, which would lead to undefined behavior.
     */
    std::pair<T, T> calcBetaGamma() const
    {
        T magnitude{module()};
        if (magnitude == 0)
            START_THROW_EXCEPTION(GeometryDivisionByZeroException,
                                  util::stringify("Cannot calculate angles for a zero vector, magnitude = ",
                                                  magnitude, "\nVector: ", getX(), " ", getY(), " ", getZ()));

        // Calculating rotation angles
        T beta{acos(getZ() / magnitude)},
            gamma{atan2(getY(), getX())};
        return std::make_pair(beta, gamma);
    }

    /**
     * @brief Linear transformation to return to the original system.
     * @param beta The angle of rotation around the Y-axis [rad].
     * @param gamma The angle of rotation around the Z-axis [rad].
     */
    void rotation(T beta, T gamma)
    {
        _rotate_y(beta);
        _rotate_z(gamma);
    }

    void rotation(std::pair<double, double> const &p) { rotation(p.first, p.second); }
    void rotation(std::pair<double, double> &&p) noexcept { rotation(p.first, p.second); }

    /**
     * @brief Returns a GeometryVector where each component is the sign of the corresponding component.
     * @details The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0.
     *          For each component of the vector, this method computes the sign and returns
     *          a new vector with these sign values.
     * @return GeometryVector with each component being -1, 0, or 1.
     */
    GeometryVector sign() const noexcept { return GeometryVector(util::signFunc(x), util::signFunc(y), util::signFunc(z)); }
};

/* --> Aliases for human readability. <-- */
using PositionVector = GeometryVector<double>;
using VelocityVector = GeometryVector<double>;
using MagneticInduction = GeometryVector<double>;
using ElectricField = GeometryVector<double>;

using PositionVector_ref = PositionVector &;
using PositionVector_rref = PositionVector &&;
using PositionVector_cref = PositionVector const &;

using VelocityVector_ref = VelocityVector &;
using VelocityVector_rref = VelocityVector &&;
using VelocityVector_cref = VelocityVector const &;

using MagneticInduction_ref = MagneticInduction &;
using MagneticInduction_rref = MagneticInduction &&;
using MagneticInduction_cref = MagneticInduction const &;

using ElectricField_ref = ElectricField &;
using ElectricField_rref = ElectricField &&;
using ElectricField_cref = ElectricField const &;
/* ----------------->   <----------------- */

#endif // !GEOMETRY_VECTOR_HPP
