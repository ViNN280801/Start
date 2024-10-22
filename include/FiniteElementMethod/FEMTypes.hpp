#ifndef FEM_TYPES_HPP
#define FEM_TYPES_HPP

#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>
#include <Intrepid2_CellTools.hpp>
#include <Intrepid2_DefaultCubatureFactory.hpp>
#include <Intrepid2_FunctionSpaceTools.hpp>
#include <Intrepid2_HGRAD_TET_C1_FEM.hpp>
#include <Intrepid2_HGRAD_TET_C2_FEM.hpp>
#include <Intrepid2_HGRAD_TET_Cn_FEM.hpp>
#include <Kokkos_Core.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <Shards_CellTopology.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

#include <array>
#include <vector>

using Scalar = double;           // ST - Scalar Type (type of the data inside the matrix node).
using LocalOrdinal = int;        // LO - indices in local matrix.
using GlobalOrdinal = long long; // GO - Global Ordinal Type (indices in global matrices).

using ExecutionSpaceHost = Kokkos::DefaultHostExecutionSpace;               // Using CPU.
using MemorySpaceHost = Kokkos::HostSpace;                                  // Host memory.
using DeviceTypeHost = Kokkos::Device<ExecutionSpaceHost, MemorySpaceHost>; // Host device.
using DynRankViewHost = Kokkos::DynRankView<Scalar, DeviceTypeHost>;        // Multi-dimensional array template for the host.
using NodeHost = Tpetra::Map<>::node_type;                                  // For CPU (OpenMP or Serial).

#ifdef USE_CUDA

using ExecutionSpaceDevice = Kokkos::Cuda;                                        // Using GPU CUDA.
using MemorySpaceDevice = Kokkos::CudaSpace;                                      // Using CUDA device memory.
using DeviceTypeDevice = Kokkos::Device<ExecutionSpaceDevice, MemorySpaceDevice>; // Device type according to the macro `USE_CUDA`.
using DynRankViewDevice = Kokkos::DynRankView<Scalar, DeviceTypeDevice>;          // Multi-dimensional array template for the device.
using NodeDevice = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType;           // For GPU.

using ExecutionSpace = ExecutionSpaceDevice;
using MemorySpace = MemorySpaceDevice;
using DeviceType = DeviceTypeDevice;
using DynRankView = DynRankViewDevice;
using Node = NodeDevice;

#else

using ExecutionSpace = ExecutionSpaceHost;
using MemorySpace = MemorySpaceHost;
using DeviceType = DeviceTypeHost;
using DynRankView = DynRankViewHost;
using Node = NodeHost;

#endif // !USE_CUDA

using MapType = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
using TpetraVectorType = Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
using TpetraMultiVector = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
using TpetraOperator = Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
using TpetraMatrixType = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
using TetrahedronIndices = std::array<LocalOrdinal, 4ul>;
using TetrahedronIndicesVector = std::vector<TetrahedronIndices>;
using Commutator = Teuchos::RCP<Teuchos::Comm<int> const>;

#if __cplusplus >= 202002L // C++20 and above
// ***** Concepts for types using C++20 concepts ***** //
/**
 * @brief Concept that ensures the type is a valid Kokkos execution space.
 *
 * This concept is used to constrain templates to types that are recognized by Kokkos as execution spaces.
 * Execution spaces define where and how parallel work is performed, such as on a CPU or GPU.
 *
 * A type satisfies this concept if `Kokkos::is_execution_space_v<T>` is `true`.
 *
 * Example:
 * @code
 * static_assert(ExecutionSpaceConcept<Kokkos::Serial>, "Kokkos::Serial is an execution space");
 * static_assert(!ExecutionSpaceConcept<int>, "int is not a valid execution space");
 * @endcode
 *
 * @tparam T The type to check if it qualifies as a valid execution space.
 */
template <typename T>
concept ExecutionSpaceConcept = Kokkos::is_execution_space_v<T>;

/**
 * @brief Concept that ensures the type is a valid Kokkos memory space.
 *
 * This concept is used to constrain templates to types that are recognized by Kokkos as memory spaces.
 * Memory spaces represent locations where data can be stored, such as host (CPU) or device (GPU) memory.
 *
 * A type satisfies this concept if `Kokkos::is_memory_space_v<T>` is `true`.
 *
 * Example:
 * @code
 * static_assert(MemorySpaceConcept<Kokkos::HostSpace>, "Kokkos::HostSpace is a memory space");
 * static_assert(!MemorySpaceConcept<int>, "int is not a valid memory space");
 * @endcode
 *
 * @tparam T The type to check if it qualifies as a valid memory space.
 */
template <typename T>
concept MemorySpaceConcept = Kokkos::is_memory_space_v<T>;

/**
 * @brief Concept that ensures the type is a valid Kokkos device type.
 *
 * This concept ensures that the provided type has both a valid execution space and a valid memory space.
 * A device type in Kokkos typically combines both an execution space and a memory space (e.g., `Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>`).
 *
 * A type satisfies this concept if:
 * - It has an inner `execution_space` type that satisfies `ExecutionSpaceConcept`.
 * - It has an inner `memory_space` type that satisfies `MemorySpaceConcept`.
 *
 * Example:
 * @code
 * static_assert(DeviceTypeConcept<Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>>, "Kokkos::Device with Serial execution and Host memory is a valid device type");
 * static_assert(!DeviceTypeConcept<int>, "int is not a valid device type");
 * @endcode
 *
 * @tparam T The type to check if it qualifies as a valid Kokkos device type.
 */
template <typename T>
concept DeviceTypeConcept = requires {
    typename T::execution_space;
    typename T::memory_space;

    Kokkos::is_execution_space_v<typename T::execution_space>;
    Kokkos::is_memory_space_v<typename T::memory_space>;
};
#else
#include <type_traits>

template <typename T>
constexpr bool ExecutionSpaceConcept_v = Kokkos::is_execution_space_v<T>;

template <typename T>
using ExecutionSpaceConcept = std::enable_if_t<ExecutionSpaceConcept_v<T>, bool>;

template <typename T>
constexpr bool MemorySpaceConcept_v = Kokkos::is_memory_space_v<T>;

template <typename T>
using MemorySpaceConcept = std::enable_if_t<MemorySpaceConcept_v<T>, bool>;

template <typename T>
constexpr bool DeviceTypeConcept_v = Kokkos::is_execution_space_v<typename T::execution_space> &&
                                     Kokkos::is_memory_space_v<typename T::memory_space>;

template <typename T>
using DeviceTypeConcept = std::enable_if_t<DeviceTypeConcept_v<T>, bool>;

#endif // __cplusplus check

#endif // !FEM_TYPES_HPP
