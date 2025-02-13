#ifndef TWOPLATESCREATOR_HPP
#define TWOPLATESCREATOR_HPP

#include <array>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "SessionManagement/GmshSessionManager.hpp"
#include "Utilities/ConfigParser.hpp"
#include "Utilities/GmshUtilities/GmshMesher.hpp"

/// @brief A class to create two plates for perform simple test on sputtering.
class TwoPlatesCreator
{
public:
    // Important! These parameters must initialized before GmshMesher class.
    std::string const kdefault_mesh_filename{"TwoPlates.msh"}; ///< Default mesh filename.
    std::string const kdefault_target_name{"Target"};          ///< Default target (мишень) physical group name.
    std::string const kdefault_substrate_name{"Substrate"};    ///< Default substrate (подложка) physical group name.

private:
    GmshSessionManager m_gmsh_session;  ///< RAII class to manage Gmsh session;
    GmshMesher m_gmsh_mesher;           ///< Instance of class to perform different types of meshing.
    MeshType m_mesh_type;               ///< Type of the mesh (Uniform, Adaptive).
    double m_cell_size;                 ///< The size of the mesh cells (for uniform mesh).
    double m_unit;                      ///< Scaling coefficient (e.g., for millimeters).
    std::array<int, 2ul> m_volume_tags; ///< Stores volume tags of the plates.

public:
    /**
     * @brief Constructor to initialize the TwoPlatesCreator and create two plates by default.
     * @param mesh_type The type of mesh to apply.
     * @param cell_size The size of the mesh cells (used only for uniform mesh).
     * @param unit Scaling coefficient (e.g., 1.0 for millimeters).
     * @throws std::invalid_argument If cell_size is non-positive for uniform mesh.
     *
     * Example usage:
     * @code
     * TwoPlatesCreator creator(MeshType::Uniform, 1.0, 1.0); // For uniform mesh
     * @endcode
     */
    TwoPlatesCreator(MeshType mesh_type, double cell_size = 1.0, double unit = 1.0);

    /**
     * @brief Adds two plates to the model.
     *
     * Algorithm:
     * 1. Use `gmsh::model::occ::addBox` to create the first plate.
     * 2. Use `gmsh::model::occ::addBox` to create the second plate.
     * 3. Synchronize the OpenCASCADE model with Gmsh.
     */
    void addPlates();

    /**
     * @brief Assigns materials (physical groups) to each plate.
     *
     * @param material1_name Material for the 1st plate (e.g. Ti, Au, Ag, W, etc.).
     * @param material2_name Material for the 2nd plate (e.g. Ti, Au, Ag, W, etc.).
     *
     * Restrictions: params material<n>_name must be simple (non-composite) materials.
     *               1 <= len <= 2
     */
    void assignMaterials(std::string_view material1_name, std::string_view material2_name);

    /**
     * @brief Generates the mesh based on the specified mesh type.
     *
     * @param dimension Dimension of mesh: 2D - surface triangle mesh.
     *                               3D - volumetric tetrahedral mesh.
     *
     * Algorithm:
     * 1. Depending on the mesh type:
     *    - For uniform mesh:
     *      - Set `Mesh.MeshSizeMin` and `Mesh.MeshSizeMax` to the cell size.
     *    - For adaptive mesh:
     *      - Get boundary tags using `getAllBoundaryTags`.
     *      - Create a `Distance` field for boundary refinement.
     *      - Define a `Threshold` field with minimum and maximum sizes.
     *      - Set the `Threshold` field as the background mesh.
     * 2. Generate the 2D mesh using `gmsh::model::mesh::generate`.
     * 3. Write the mesh to a file named "TwoPlates.msh".
     * 4. Launch the Gmsh GUI to display the generated mesh.
     *
     * Example usage:
     * @code
     * TwoPlatesCreator creator(MeshType::Adaptive, 0.0, 1.0);
     * creator.initializeGmsh();
     * creator.generateMesh();
     * @endcode
     */
    void generateMeshfile(int dimension = 2);

    /// @brief Automatically set the target surface by point IDs. Specified by 'kdefault_target_point_tags'.
    void setTargetSurface() const;

    /// @brief Automatically set the substrate surface by point IDs. Specified by 'kdefault_substrate_point_tags'.
    void setSubstrateSurface() const;

    /**
     * @brief Using Rectangle Area Calculation formula to calculate area of the target surface.
     * @returns Area of the target surface in [m2].
     */
    constexpr double calculateTargetSurfaceArea() const { return 10.0 * 60.0 * 1e-6; }

    /**
     * @brief Calculating count of the particles on the target surface.
     *
     * @param targetMaterialDensity Target material density [kg/m3]. For example, Titan: 4500 [kg/m3].
     * @param targetMaterialMolarMass Target material molar mass [kg/mol]. For example, Titan: 0.048 [kg/mol].
     * @return double Count of particles on surface.
     */
    double calculateCountOfParticlesOnTargetSurface(double targetMaterialDensity, double targetMaterialMolarMass) const
    {
        return (targetMaterialDensity * constants::physical_constants::N_av * calculateTargetSurfaceArea() *
                (2 * constants::physical_constants::Ti_radius)) /
               targetMaterialMolarMass;
    }

    /**
     * @brief Prepare data for the generating particles uniformly on 'Target' surface.
     *
     * @details This function identifies the physical group named "Target" in the mesh, calculates
     *          the centroids of all triangles associated with the target surface, and associates
     *          a user-defined number of particles (N_model) and their energy (energy_eV) with
     *          these centroids. The result is a surface_source_t structure that maps centroids
     *          to their normal vectors for particle emission.
     *
     *          Steps:
     *              1. Locate the physical group named "Target" and retrieve its tag.
     *              2. Extract all node tags and their coordinates associated with the "Target" physical group.
     *              3. Identify all triangular elements that belong to the "Target" group and store their nodes.
     *              4. Calculate the centroid for each triangle.
     *              5. Associate the centroids with their normal vectors ([0, 0, -1]) and store in the result.
     *
     * @param N_model Count of model particles (1 model particle = 10^w real particles, where 'w' - weight of model particle).
     * @param energy_eV Energy of each particle in [eV].
     * @return surface_source_t data for the particle generator.
     */
    surface_source_t prepareDataForSpawnParticles(size_t N_model, double energy_eV) const;

    /// @brief Runs the Gmsh application to show the model.
    void show(int argc, char *argv[]);

    /// @brief Get mesh filename with generated plates in '.msh' format.
    constexpr std::string const &getMeshFilename() const { return kdefault_mesh_filename; }
};

#endif // !TWOPLATESCREATOR_HPP
