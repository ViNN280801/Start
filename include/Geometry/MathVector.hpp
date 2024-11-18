#ifndef MATHVECTOR_HPP
#define MATHVECTOR_HPP

#include <compare>
#include <random>
#include <stdexcept>

#include "Utilities/Utilities.hpp"

#if __cplusplus >= 202002L
#include <concepts>

/**
 * @brief Concept that enforces the template type to be either integral or floating-point.
 * @tparam MathVectorComponent The type to be checked.
 */
template <typename MathVectorComponent>
concept Numeric = std::integral<MathVectorComponent> || std::floating_point<MathVectorComponent>;

#else
#include <type_traits>

/**
 * @brief Type trait that checks if the template type is either integral or floating-point.
 * @tparam MathVectorComponent The type to be checked.
 */
template <typename MathVectorComponent>
struct is_numeric : std::integral_constant<bool, std::is_integral<MathVectorComponent>::value ||
                                                     std::is_floating_point<MathVectorComponent>::value>
{
};

/**
 * @brief Helper to simplify checking if the type is numeric.
 */
template <typename MathVectorComponent>
inline constexpr bool is_numeric_v = is_numeric<MathVectorComponent>::value;
#endif

/**
 * Description of the mathematical vector. In the simplest case, a
 * mathematical object characterized by magnitude and direction.
 *               ->  | x |
 * @brief Vector P = | y |
 *                   | z |
 */
#if __cplusplus >= 202002L
template <Numeric T>
#else
template <typename T, typename = std::enable_if_t<is_numeric_v<T>>>
#endif
class MathVector
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
    MathVector() = default;

    /**
     * @brief Constructor that initializes the vector with specific values for x, y, and z.
     * @param x_ X-coordinate to assign.
     * @param y_ Y-coordinate to assign.
     * @param z_ Z-coordinate to assign.
     */
    MathVector(T x_, T y_, T z_) : x{x_}, y{y_}, z{z_} {}

    /**
     * @brief Assignment operator with custom double.
     *        Sets all components of vector to custom value.
     * @param value The value to assign to all components.
     * @return Reference to this vector.
     */
    MathVector &operator=(T value)
    {
        x = value;
        y = value;
        z = value;
        return *this;
    }

    /**
     * @brief Fills `MathVector` object with specified values.
     * @param x_ X-coordinate of the point.
     * @param y_ Y-coordinate of the point.
     * @param z_ Z-coordinate of the point.
     * @return Filled structure of the Cartesian coordinates.
     */
    static MathVector createCoordinates(T x_, T y_, T z_)
    {
        MathVector vec;

        vec.x = x_;
        vec.y = y_;
        vec.z = z_;

        return vec;
    }

    /**
     * @brief Creates a random MathVector with components in the range [from, to].
     * @tparam T The type of the vector components (must be integral or floating-point).
     * @param from The lower bound of the range for random values.
     * @param to The upper bound of the range for random values.
     * @return A randomly generated MathVector with components in the specified range.
     *
     * This function uses a random number generator to create a MathVector where each
     * component is a random value within the given range. The type of the random distribution
     * (real or integer) depends on the type of the vector components.
     */
    static MathVector createRandomVector(T from, T to)
    {
        std::random_device rd;
        std::mt19937 gen(rd.entropy() ? rd() : static_cast<unsigned>(time(nullptr)));

        // Use the appropriate distribution based on whether T is integral or floating-point.
        if constexpr (std::is_floating_point_v<T>)
        {
            std::uniform_real_distribution<T> dis(from, to);
            return MathVector<T>(dis(gen), dis(gen), dis(gen));
        }
        else if constexpr (std::is_integral_v<T>)
        {
            std::uniform_int_distribution<T> dis(from, to);
            return MathVector<T>(dis(gen), dis(gen), dis(gen));
        }
    }

    /* === Getters for each component. === */
    constexpr T getX() const { return x; }
    constexpr T getY() const { return y; }
    constexpr T getZ() const { return z; }

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
    T distance(MathVector const &other) const { return std::sqrt(std::pow(other.x - x, 2) +
                                                                 std::pow(other.y - y, 2) +
                                                                 std::pow(other.z - z, 2)); }
    T distance(MathVector &&other) const { return std::sqrt(std::pow(other.x - x, 2) +
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
    bool isParallel(MathVector const &other) const
    {
        T koef{x / other.x};
        return (y == koef * other.y) && (z == koef * other.z);
    }
    bool isParallel(MathVector &&other) const
    {
        T koef{x / other.x};
        return (y == koef * other.y) && (z == koef * other.z);
    }

    /**
     * @brief Checks if vectors are orthogonal.
     * `a` is orthogonal to `b` if their dot (scalar) product is equals to 0.
     * @return `true` if vectors are orthogonal, otherwise `false`.
     */
    bool isOrthogonal(MathVector const &other) const { return dotProduct(other) == 0; }
    bool isOrthogonal(MathVector &&other) const { return dotProduct(std::move(other)) == 0; }

    /**
     * @brief Calculates the area of a triangle given its three vertices.
     * @details This function computes the area of a triangle in a 2D space using the vertices A, B, and C.
     *          The formula used is the absolute value of half the determinant of a 2x2 matrix formed by
     *          subtracting the coordinates of A from those of B and C. This method is efficient and
     *          works well for triangles defined in a Cartesian coordinate system.
     * @param A The first vertex of the triangle, represented as a MathVector.
     * @param B The second vertex of the triangle, represented as a MathVector.
     * @param C The third vertex of the triangle, represented as a MathVector.
     * @return The area of the triangle as a double value.
     */
    static T calculateTriangleArea(MathVector const &A,
                                   MathVector const &B,
                                   MathVector const &C)
    {
        return std::fabs((B.getX() - A.getX()) * (C.getY() - A.getY()) -
                         (B.getY() - A.getY()) * (C.getX() - A.getX())) /
               2.0;
    }
    static T calculateTriangleArea(MathVector &&A,
                                   MathVector &&B,
                                   MathVector &&C)
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
            throw std::out_of_range("Requested index " + std::to_string(k) + " for MathVector is out of range");
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
            throw std::out_of_range("Requested index " + std::to_string(k) + " for MathVector is out of range");
        }
    }

    T &operator()(int k)
    {
        if (k < 0 || k > 2)
            throw std::out_of_range("Requested index " + std::to_string(k) + " for MathVector is out of range");
        return (*this)[k];
    }

    T const &operator()(int k) const
    {
        if (k < 0 || k > 2)
            throw std::out_of_range("Requested index " + std::to_string(k) + " for MathVector is out of range");
        return (*this)[k];
    }

    T &at(int k) { return (*this)(k); }

    T const &at(int k) const { return (*this)(k); }

    /// @brief Overload of unary minus. Negates all components of vector.
    MathVector operator-() { return MathVector(-x, -y, -z); }

    /* +++ Subtract and sum of two vectors correspondingly. +++ */
    MathVector operator-(MathVector const &other) const { return MathVector(x - other.x, y - other.y, z - other.z); }
    MathVector operator+(MathVector const &other) const { return MathVector(x + other.x, y + other.y, z + other.z); }

    /* +++ Subtract/sum/product-assign operators for scalar and for the other vector. +++ */
    MathVector &operator+=(MathVector const &other)
    {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    MathVector &operator-=(MathVector const &other)
    {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    MathVector &operator+=(T value)
    {
        x += value;
        y += value;
        z += value;
        return *this;
    }

    MathVector &operator-=(T value)
    {
        x -= value;
        y -= value;
        z -= value;
        return *this;
    }

    MathVector &operator*=(T value)
    {
        x *= value;
        y *= value;
        z *= value;
        return *this;
    }

    MathVector &operator*=(MathVector const &other)
    {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        return *this;
    }

    /* +++ Subtract and sum of value to vector. +++ */
    MathVector operator-(T value) const { return MathVector(x - value, y - value, z - value); }
    MathVector operator+(T value) const { return MathVector(x + value, y + value, z + value); }
    friend MathVector operator+(T value, MathVector const &other) { return MathVector(other.x + value, other.y + value, other.z + value); }

    /* *** Scalar and vector multiplication correspondingly. *** */
    MathVector operator*(T value) const { return MathVector(x * value, y * value, z * value); }
    friend MathVector operator*(T value, MathVector const &other) { return MathVector(other.x * value, other.y * value, other.z * value); }
    T operator*(MathVector const &other) const { return (x * other.x + y * other.y + z * other.z); }
    T operator*(MathVector &&other) const { return (x * other.x + y * other.y + z * other.z); }
    T dotProduct(MathVector const &other) const { return (*this) * other; }
    T dotProduct(MathVector &&other) const { return (*this) * other; }
    MathVector crossProduct(MathVector const &other) const { return MathVector(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x); }
    MathVector crossProduct(MathVector &&other) const { return MathVector(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x); }

    /* /// Division operator. Vector / value. \\\ */
    MathVector operator/(T value) const
    {
        if (value == 0)
            throw std::overflow_error("Division by null: Elements of vector can't be divided by 0");
        return MathVector(x / value, y / value, z / value);
    }

    /* <=> Comparison operators. <=> */
    constexpr bool operator==(MathVector const &other) const { return x == other.x && y == other.y && z == other.z; }
    constexpr bool operator!=(MathVector const &other) const { return !(*this == other); }

    /* << Output stream operator. << */
    friend std::ostream &operator<<(std::ostream &os, MathVector const &vector)
    {
        os << vector.x << ' ' << vector.y << ' ' << vector.z;
        return os;
    }

    /* >> Input stream operator. >> */
    friend std::istream &operator>>(std::istream &is, MathVector &vector)
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
    MathVector normalize() const
    {
        T magnitude{module()};
        return magnitude != 0 ? MathVector{x / magnitude, y / magnitude, z / magnitude} : MathVector{};
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
            throw std::runtime_error("Cannot calculate angles for a zero vector.");

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
     * @brief Returns a MathVector where each component is the sign of the corresponding component.
     * @details The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0.
     *          For each component of the vector, this method computes the sign and returns
     *          a new vector with these sign values.
     * @return MathVector with each component being -1, 0, or 1.
     */
    MathVector sign() const noexcept { return MathVector(util::signFunc(x), util::signFunc(y), util::signFunc(z)); }
};

/* --> Aliases for human readability. <-- */
using PositionVector = MathVector<double>;
using VelocityVector = MathVector<double>;
using MagneticInduction = MathVector<double>;
using ElectricField = MathVector<double>;

#endif // !MATHVECTOR_HPP
