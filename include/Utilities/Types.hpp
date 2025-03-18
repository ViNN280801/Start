#ifndef TYPES_HPP
#define TYPES_HPP

#if __cplusplus >= 202002L
#include <concepts>
#include <ranges>
#else
#include <iterator>
#include <type_traits>
#endif

#if __cplusplus >= 202002L
// ===============================================================================================
// =================================== Math vector type ==========================================
// ===============================================================================================
/**
 * @brief Concept for a vector-like structure where elements are numeric (floating-point or integral).
 *
 * @tparam VectorType The type to check for vector-like behavior.
 */
template <typename VectorType>
concept Vector =
    std::ranges::range<VectorType> &&                                    // VectorType must be a range.
    (std::is_floating_point_v<std::ranges::range_value_t<VectorType>> || // Elements must be floating-point...
     std::is_integral_v<std::ranges::range_value_t<VectorType>>);        // ...or integral.

// ===============================================================================================
// =================================== Math matrix type ==========================================
// ===============================================================================================
/**
 * @brief Concept for a matrix-like structure where elements are ranges of a specified value type.
 *
 * @tparam MatrixType The type of the matrix container.
 * @tparam ValueType The type of the elements stored in the matrix (must be floating-point or integral).
 */
template <typename MatrixType, typename ValueType>
concept Matrix =
    std::ranges::range<MatrixType> &&                                                                             // MatrixType must be a range.
    std::ranges::range<std::ranges::range_reference_t<MatrixType>> &&                                             // Elements of MatrixType must also be ranges.
    std::convertible_to<std::ranges::range_reference_t<std::ranges::range_reference_t<MatrixType>>, ValueType> && // Nested elements must be convertible to ValueType.
    (std::is_floating_point_v<ValueType> || std::is_integral_v<ValueType>);                                       // ValueType must be floating-point or integral.
#else
// ===============================================================================================
// =================================== Math vector type ==========================================
// ===============================================================================================
/**
 * @brief Type trait to check if a type supports begin() and end() and contains numeric elements.
 *
 * @tparam T The type to check.
 */
template <typename T, typename = void>
struct is_vector : std::false_type
{
};

template <typename T>
struct is_vector<T, std::void_t<decltype(std::begin(std::declval<T>())), // Supports begin() and end().
                                decltype(std::end(std::declval<T>())),
                                typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::value_type>> // Has value_type.
    : std::integral_constant<bool,
                             (std::is_floating_point<typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::value_type>::value ||
                              std::is_integral<typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::value_type>::value)>
{
};

/// @brief Convenience variable template for is_vector.
template <typename T>
constexpr bool is_vector_v = is_vector<T>::value;

// ===============================================================================================
// =================================== Math matrix type ==========================================
// ===============================================================================================

/**
 * @brief Helper type trait to check if a type is a range (i.e., supports `begin()` and `end()`).
 * @tparam T The type to check.
 */
template <typename T, typename = void>
struct is_range : std::false_type
{
};

template <typename T>
struct is_range<T, std::void_t<decltype(std::begin(std::declval<T>())),
                               decltype(std::end(std::declval<T>()))>> : std::true_type
{
};

/**
 * @brief Helper type trait to extract the reference type of a range.
 * @tparam T The range type.
 */
template <typename T>
using range_reference_t = decltype(*std::begin(std::declval<T>()));

/**
 * @brief Helper type trait to check if all elements of a range are of a specific type.
 * @tparam Range The range type.
 * @tparam ValueType The value type to check.
 */
template <typename Range, typename ValueType, typename = void>
struct all_elements_convertible_to : std::false_type
{
};

template <typename Range, typename ValueType>
struct all_elements_convertible_to<Range, ValueType,
                                   std::enable_if_t<is_range<Range>::value &&
                                                    std::is_convertible<range_reference_t<Range>, ValueType>::value>> : std::true_type
{
};

/**
 * @brief Type trait to check if a type satisfies the Matrix-like structure requirements.
 *        A Matrix is defined as a range of ranges where the nested elements are convertible to a specified value type,
 *        and the value type is either floating-point or integral.
 * @tparam MatrixType The type of the matrix container.
 * @tparam ValueType The type of the elements stored in the matrix.
 */
template <typename MatrixType, typename ValueType, typename = void>
struct is_matrix : std::false_type
{
};

template <typename MatrixType, typename ValueType>
struct is_matrix<
    MatrixType, ValueType,
    std::enable_if_t<is_range<MatrixType>::value &&                                                     // MatrixType must be a range.
                     is_range<range_reference_t<MatrixType>>::value &&                                  // Elements of MatrixType must also be ranges.
                     all_elements_convertible_to<range_reference_t<MatrixType>, ValueType>::value &&    // Nested elements must be convertible to ValueType.
                     (std::is_floating_point<ValueType>::value || std::is_integral<ValueType>::value)>> // ValueType must be floating-point or integral.
    : std::true_type
{
};

/// @brief Convenience variable template for is_matrix
template <typename MatrixType, typename ValueType>
constexpr bool is_matrix_v = is_matrix<MatrixType, ValueType>::value;
#endif // __cplusplus >= 202002L
#endif // !TYPES_HPP
