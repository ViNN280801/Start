#ifndef VECTORMANAGER_HPP
#define VECTORMANAGER_HPP

#include "FiniteElementMethod/FEMTypes.hpp"

/* ATTENTION: Works well only for the polynom order = 1. */

/**
 * @class VectorManager
 * @brief Class responsible for managing the solution vector in the equation Ax = b in FEM.
 *
 * The VectorManager class provides methods to manage and manipulate vectors used in FEM computations,
 * such as clearing, randomizing, and accessing vector elements. It supports distributed computation
 * by using Teuchos RCP (Reference-counted pointers) to manage memory in a parallel processing environment.
 *
 * @note This implementation works best for polynomial order 1.
 */
class VectorManager
{
private:
    Teuchos::RCP<MapType> m_map;             ///< A smart pointer managing the lifetime of a Map object, which defines the layout of distributed data across the processes in a parallel computation.
    Teuchos::RCP<TpetraVectorType> m_vector; ///< Solution vector `b` in the equation: Ax=b. Where `A` - global stiffness matrix; `x` - vector to find; `b` - solution vector.

public:
    /**
     * @brief Constructor for the VectorManager class.
     *
     * Initializes a solution vector of given size and polynomial order.
     *
     * @param size The size of the vector (number of elements).
     */
    VectorManager(size_t size);
    ~VectorManager() {}

    /**
     * @brief Getter for size of the vector.
     * @return Size of the vector.
     */
    size_t size() const;

    /// @brief Zeros out all the elements in the vector.
    void clear();

    /// @brief Assigns random values to all elements of the vector.
    void randomize();

    /// @brief Getter for the solution vector.
    constexpr Teuchos::RCP<TpetraVectorType> const &get() const { return m_vector; }

    /**
     * @brief Access an element of the vector by reference.
     *
     * This method provides non-const access to the element at index `i`, allowing modification.
     *
     * @param i The index of the element to access.
     * @return A reference to the element at index `i`.
     * @throws std::out_of_range if the index is out of bounds.
     */
    Scalar &at(GlobalOrdinal i);

    /**
     * @brief Access an element of the vector by constant reference.
     *
     * This method provides read-only access to the element at index `i`.
     *
     * @param i The index of the element to access.
     * @return A constant reference to the element at index `i`.
     * @throws std::out_of_range if the index is out of bounds.
     */
    Scalar const &at(GlobalOrdinal i) const;

    /**
     * @brief Access an element of the vector using the subscript operator.
     *
     * This method allows element access using `[]`.
     *
     * @param i The index of the element to access.
     * @return A reference to the element at index `i`.
     */
    Scalar &operator[](GlobalOrdinal i);

    /**
     * @brief Access an element of the vector using the subscript operator.
     *
     * This method allows constant element access using `[]`.
     *
     * @param i The index of the element to access.
     * @return A constant reference to the element at index `i`.
     */
    Scalar const &operator[](GlobalOrdinal i) const;
};

#endif // !VECTORMANAGER_HPP
