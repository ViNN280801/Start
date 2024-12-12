#ifndef TETRAHEDRONMESHMANAGER_HPP
#define TETRAHEDRONMESHMANAGER_HPP

#include <array>
#include <optional>

#include "FiniteElementMethod/FEMTypes.hpp"
#include "Geometry/GeometryTypes.hpp"
#include "Utilities/Utilities.hpp"

/**
 * @brief Class representing volumetric mesh data for a tetrahedron.
 *        All the global indexes refers to GMSH indexing.
 *        Uses Singleton pattern.
 */
class TetrahedronMeshManager final
{
private:
    struct TetrahedronData
    {
        struct NodeData
        {
            size_t globalNodeId{};           ///< Global Id of the node.
            Point nodeCoords;                ///< Coordinates of this node.
            std::optional<Point> nablaPhi;   ///< Optional field for the gradient of basis function.
            std::optional<double> potential; ///< Optional field for the potential in node.
        };

        size_t globalTetraId{};             ///< Global ID of the tetrahedron according to GMSH indexing.
        Tetrahedron tetrahedron;            ///< CGAL::Tetrahedron_3 object from the mesh.
        std::array<NodeData, 4ul> nodes;    ///< Tetrahedron verteces and their coordinates.
        std::optional<Point> electricField; ///< Optional field for the electric field of the tetrahedron.

        /**
         * @brief Get the center point of the tetrahedron.
         * @return Point representing the center of the tetrahedron.
         */
        [[nodiscard("The center of the tetrahedron is important for further geometric calculations and shouldn't be discarded.")]]
        Point getTetrahedronCenter() const;
    };

    std::vector<TetrahedronData> m_meshComponents; ///< Array of all the tetrahedrons from the mesh.

    int m_rank; ///< Rank of the current MPI process (for distributed mesh processing).
    int m_size; ///< Total number of MPI processes (for distributed mesh processing).

    std::vector<std::pair<size_t, std::array<size_t, 4>>> m_localElementData; ///< Local elements assigned to the current process (element ID and node IDs).
    std::vector<size_t> m_localNodeTags;                                      ///< List of node IDs that are local to the current process.
    std::map<size_t, std::array<double, 3>> m_localNodeCoordinates;           ///< Coordinates of nodes that are local to the current process.

    std::map<size_t, std::array<double, 3>> m_nodeCoordinatesMap;                         ///< Global map of node IDs to their coordinates.
    std::vector<std::vector<std::pair<size_t, std::array<size_t, 4>>>> m_elementsPerProc; ///< Elements partitioned by process (used for MPI distribution).
    std::vector<std::vector<size_t>> m_nodeTagsPerProc;                                   ///< Node tags partitioned by process (used for MPI distribution).

    DynRankViewHost m_tetraVertices;                           ///< A view storing coordinates of all tetrahedron vertices.
    DynRankViewHost m_tetraGlobalIds;                          ///< A view mapping local tetrahedron index to its global ID.
    std::unordered_map<size_t, size_t> m_globalToLocalTetraId; ///< Map from global tetra ID to local index.

    /**
     * @brief Reads and partitions the mesh file. Only called by the root process (rank 0).
     * @param mesh_filename Path to the mesh file to be read and partitioned.
     */
    void _readAndPartitionMesh(std::string_view mesh_filename);

    /**
     * @brief Distributes the mesh data from the root process to other processes.
     *        Only called by the root process (rank 0).
     */
    void _distributeMeshData();

    /**
     * @brief Receives partitioned mesh data from the root process.
     *        Only called by non-root processes (ranks != 0).
     */
    void _receiveData();

    /// @brief Constructs the local mesh representation using the received data.
    void _constructLocalMesh();

    /**
     * @brief Build DynRankViews and maps for easier index access.
     *
     * This method populates:
     * - `m_tetraVertices`: A DynRankViewHost with shape [numTets,4,3], storing coordinates.
     * - `m_tetraGlobalIds`: A DynRankViewHost mapping local indices to global tetra IDs.
     * - `m_globalToLocalTetraId`: An unordered_map for global-to-local tetra ID lookup.
     */
    void _buildViews();

public:
    using NodeData = TetrahedronMeshManager::TetrahedronData::NodeData;
    using TetrahedronData = TetrahedronMeshManager::TetrahedronData;

    /**
     * @brief Constructor. Fills storage for the volumetric mesh.
     * @details This constructor reads the mesh file once, extracts the node coordinates and tetrahedron
     *          connections, and initializes the internal storage for the volumetric mesh data.
     * @param mesh_filename The filename of the mesh file to read.
     * @throw std::runtime_error if there is an error reading the mesh file or extracting the data.
     */
    TetrahedronMeshManager(std::string_view mesh_filename);

    // Preventing copy of this object.
    TetrahedronMeshManager(TetrahedronMeshManager const &) = delete;
    TetrahedronMeshManager(TetrahedronMeshManager &&) = delete;
    TetrahedronMeshManager &operator=(TetrahedronMeshManager const &) = delete;
    TetrahedronMeshManager &operator=(TetrahedronMeshManager &&) = delete;

    /// @brief Getter for all the tetrahedra mesh components from the mesh.
    [[nodiscard("Retrieving mesh components is essential for accessing tetrahedron data.")]]
    auto &getMeshComponents()
    {
        return m_meshComponents;
    }

    [[nodiscard("Retrieving mesh components is essential for accessing tetrahedron data.")]]
    constexpr auto const &getMeshComponents() const
    {
        return m_meshComponents;
    }

    /// @brief Prints all the mesh components to the stdout.
    void print() const noexcept;

    /// @brief Returns count of the tetrahedra in the mesh.
    [[nodiscard("Knowing the number of tetrahedrons is important for mesh operations.")]]
    STARTCONSTEXPRFUNC size_t getNumTetrahedrons() const
    {
        return m_meshComponents.size();
    }

    /**
     * @brief Retrieves the mesh data for a specific tetrahedron by its local ID.
     * @param localTetraId The local ID of the tetrahedron in the mesh.
     * @return A constant reference to the `TetrahedronData` of the specified tetrahedron.
     */
    [[nodiscard("Accessing specific tetrahedron data is critical for accurate computations.")]]
    TetrahedronData const &getTetrahedron(size_t localTetraId) const
    {
        return m_meshComponents.at(localTetraId);
    }

    /**
     * @brief Retrieves the global tetrahedron ID for a specific tetrahedron.
     * @param localTetraId The local ID of the tetrahedron in the mesh.
     * @return The global ID of the specified tetrahedron.
     */
    [[nodiscard("The global ID is essential for identifying tetrahedrons in distributed systems.")]]
    size_t getGlobalTetraId(size_t localTetraId) const
    {
        return m_meshComponents.at(localTetraId).globalTetraId;
    }

    /**
     * @brief Retrieves the node data for a specific tetrahedron.
     * @param localTetraId The local ID of the tetrahedron in the mesh.
     * @return A constant reference to an array containing the node data for the specified tetrahedron.
     */
    [[nodiscard("Accessing node data is necessary for accurate mesh analysis.")]]
    std::array<TetrahedronData::NodeData, 4ul> const &getTetrahedronNodes(size_t localTetraId) const
    {
        return m_meshComponents.at(localTetraId).nodes;
    }

    /**
     * @brief Retrieves the global node ID for a specific node in a tetrahedron.
     * @param localTetraId The local ID of the tetrahedron in the mesh.
     * @param localNodeId The local ID of the node in the tetrahedron.
     * @return The global node ID of the specified node.
     */
    [[nodiscard("The global node ID is vital for cross-referencing mesh components.")]]
    size_t getGlobalNodeId(size_t localTetraId, size_t localNodeId) const
    {
        return m_meshComponents.at(localTetraId).nodes.at(localNodeId).globalNodeId;
    }

    /**
     * @brief Retrieves the volume of a specific tetrahedron based on its global ID.
     *
     * This function searches for the tetrahedron with the specified global ID in the mesh
     * and returns its volume. It uses the `getMeshDataByTetrahedronId()` method to locate
     * the tetrahedron and accesses its `tetrahedron` field to compute the volume.
     *
     * @param globalTetraId The global ID of the tetrahedron to retrieve.
     * @return double The volume of the specified tetrahedron.
     * @throw std::bad_optional_access if no tetrahedron is found for the provided global ID.
     *        This occurs if `getMeshDataByTetrahedronId()` returns `std::nullopt`.
     */
    [[nodiscard("The volume is critical for physical and geometric calculations.")]]
    double getVolumeByGlobalTetraId(size_t globalTetraId) const
    {
        return getMeshDataByTetrahedronId(globalTetraId).value().tetrahedron.volume();
    }

    /// @brief Checks and returns result of the checking if there is no tetrahedra in the mesh.
    [[nodiscard("It's necessary to check if the mesh is empty to avoid null operations.")]]
    STARTCONSTEXPRFUNC bool empty() const
    {
        return m_meshComponents.empty();
    }

    /// @brief Returns total volume of the mesh.
    [[nodiscard("The total volume is essential for global calculations over the mesh.")]]
    STARTCONSTEXPRFUNC double volume() const
    {
        return std::accumulate(m_meshComponents.cbegin(), m_meshComponents.cend(), 0.0, [](double sum, auto const &meshData)
                               { return sum + meshData.tetrahedron.volume(); });
    }

    /**
     * @brief Retrieves the mesh data for a specific tetrahedron by its global ID.
     *
     * This function searches through the mesh components and returns the data
     * for the tetrahedron with the specified global ID. If no such tetrahedron
     * is found, it returns `std::nullopt`.
     *
     * @param globalTetrahedronId The global ID of the tetrahedron to retrieve.
     * @return std::optional<TetrahedronData> An optional containing the TetrahedronData if found, or std::nullopt if not found.
     */
    std::optional<TetrahedronData> getMeshDataByTetrahedronId(size_t globalTetrahedronId) const;

    /**
     * @brief Assigns the gradient of the basis function to the corresponding node.
     * @param tetrahedronId The global ID of the tetrahedron.
     * @param nodeId The global ID of the node.
     * @param gradient The gradient of the basis function.
     */
    void assignNablaPhi(size_t tetrahedronId, size_t nodeId, Point const &gradient);

    /**
     * @brief Assigns the potential to the corresponding node.
     * @param nodeId The global ID of the node.
     * @param potential The potential value.
     */
    void assignPotential(size_t nodeId, double potential);

    /**
     * @brief Assigns the electric field to the corresponding tetrahedron.
     * @param tetrahedronId The global ID of the tetrahedron.
     * @param electricField The electric field vector.
     */
    void assignElectricField(size_t tetrahedronId, Point const &electricField);

    /**
     * @brief Gets ID of tetrahedrons and corresponding IDs of elements within.
     * @return Map with key = tetrahedron's ID, value = list of nodes inside.
     */
    [[nodiscard("Mapping tetrahedron IDs to node IDs is essential for node traversal.")]]
    std::map<size_t, std::vector<size_t>> getTetrahedronNodesMap() const;

    /**
     * @brief Map for global mesh nodes with all neighbour tetrahedrons.
     * @return Map with key = node ID, value = list of neighbour tetrahedrons to this node.
     */
    [[nodiscard("Mapping nodes to tetrahedrons is necessary for adjacency queries.")]]
    std::map<size_t, std::vector<size_t>> getNodeTetrahedronsMap() const;

    /**
     * @brief Calculates the geometric centers of all tetrahedrons in a given mesh.
     *
     * @details This function opens a mesh file in Gmsh format specified by `msh_filename` and computes
     *          the geometric centers of each tetrahedron. The center of a tetrahedron is calculated as
     *          the arithmetic mean of its vertices' coordinates. These centers are often used for
     *          various geometric and physical calculations, such as finding the centroid of mass in
     *          finite element analysis or for visualizing properties that vary across the mesh volume.
     *
     * @return std::map<size_t, CGAL::Point_3> A map where each key is a tetrahedron ID and
     *         the value is an array representing the XYZ coordinates of its geometric center. This map
     *         provides a convenient way to access the center of each tetrahedron by its identifier.
     *
     * @throws std::exception Propagates any exceptions thrown by file handling or the Gmsh API, which
     *         might occur during file opening, reading, or processing. These exceptions are typically
     *         caught and should be handled to avoid crashes and ensure that the error is reported properly.
     */
    [[nodiscard("Tetrahedron centers are vital for geometric and physical calculations.")]]
    std::map<size_t, Point> getTetrahedronCenters() const;

    /**
     * @brief Get the local tetrahedron ID given a global tetrahedron ID.
     *
     * @param globalId The global ID of the tetrahedron (from Gmsh).
     * @return The local index (0-based) of the tetrahedron.
     * @throw std::out_of_range if globalId is not found.
     */
    size_t getLocalTetraId(size_t globalId) const { return m_globalToLocalTetraId.at(globalId); }

    /**
     * @brief Access the DynRankView of tetrahedron vertices.
     *
     * @return A copy of the DynRankViewHost that stores tetrahedron vertices.
     */
    inline DynRankViewHost getTetraVerticesView() const noexcept { return m_tetraVertices; }

    /**
     * @brief Access the DynRankView of global tetrahedron IDs.
     *
     * @return A copy of the DynRankViewHost that stores global IDs for each local tetrahedron index.
     */
    inline DynRankViewHost getTetraGlobalIdsView() const noexcept { return m_tetraGlobalIds; }

    /**
     * @brief Compute local stiffness matrices and assign nablaPhi for each node of each tetrahedron.
     *
     * This method internally computes jacobians, inverse jacobians, cell measures,
     * transforms the reference gradients of the basis functions, and integrates
     * to form local stiffness matrices. It also assigns nablaPhi to each node.
     *
     * @param basis The chosen basis functions object (already constructed outside).
     * @param cubPoints The cubature points.
     * @param cubWeights The cubature weights.
     * @param numBasisFunctions The number of basis functions per tetrahedron.
     * @param numCubaturePoints The number of cubature points.
     * @return DynRankView with shape (numTetrahedrons, numBasisFunctions, numBasisFunctions)
     *         containing the local stiffness matrices.
     */
    DynRankView computeLocalStiffnessMatricesAndNablaPhi(
        Intrepid2::Basis<DeviceType, Scalar, Scalar> *basis,
        DynRankView const &cubPoints,
        DynRankView const &cubWeights,
        size_t numBasisFunctions,
        size_t numCubaturePoints);

    /**
     * @brief Compute the electric field for each tetrahedron based on assigned potentials and nablaPhi.
     *
     * This method loops over all tetrahedrons, computes:
     * \f$ E_{\text{cell}} = \sum_i (\varphi_i * \nabla \varphi_i) \f$
     * for each tetrahedron, and updates the tetrahedron's electricField field.
     *
     * If either potential or nablaPhi is not set for a node, a warning is issued and that node is skipped in the sum.
     */
    void computeElectricFields();
};

#endif // !TETRAHEDRONMESHMANAGER_HPP
