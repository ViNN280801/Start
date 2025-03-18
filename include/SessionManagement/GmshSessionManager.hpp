#ifndef GMSH_SESSION_MANAGER_HPP
#define GMSH_SESSION_MANAGER_HPP

/**
 * @brief GmshSessionManager is a RAII (Resource Acquisition Is Initialization) class for managing
 * GMSH initialization and finalization. The constructor initializes GMSH, and the destructor finalizes it.
 * Copy and move operations are deleted to prevent multiple instances from initializing or
 * finalizing GMSH multiple times.
 */
class GmshSessionManager
{
public:
    /**
     * @brief Constructs a GmshSessionManager object and initializes GMSH.
     * @details This constructor checks if the GMSH library is already initialized.
     *          If not, it proceeds to initialize GMSH. This ensures that only a single
     *          instance of the GMSH session is active during the object's lifetime,
     *          following the RAII pattern for resource management.
     *
     * @note This constructor is explicitly defined to prevent multiple initializations
     *       by checking the GMSH state and only initializing if it has not already been done.
     */
    GmshSessionManager();

    /**
     * @brief Destructs the GmshSessionManager object and finalizes GMSH.
     * @details This destructor checks if the GMSH library is currently initialized.
     *          If it is, it finalizes GMSH to release resources and perform any necessary
     *          cleanup. This ensures that the GMSH session is properly closed when the
     *          object goes out of scope, maintaining resource integrity.
     *
     * @note This destructor is explicitly defined to ensure that GMSH is only finalized
     *       if it is initialized, preventing redundant or unsafe finalizations.
     */
    ~GmshSessionManager();

    // Preventing multiple initializations/finalizations:
    GmshSessionManager(GmshSessionManager const &) = delete;
    GmshSessionManager &operator=(GmshSessionManager const &) = delete;
    GmshSessionManager(GmshSessionManager &&) noexcept = delete;
    GmshSessionManager &operator=(GmshSessionManager &&) noexcept = delete;

    /**
     * @brief Executes the Gmsh application unless the `-nopopup` argument is provided.
     * @details This method checks the provided arguments for the presence of `-nopopup`.
     *          If `-nopopup` is not found, it initiates the Gmsh graphical user interface.
     *          This is typically used for visualizing and interacting with the Gmsh application directly.
     * @param argc An integer representing the count of command-line arguments.
     * @param argv A constant pointer to a character array, representing the command-line arguments.
     */
    void runGmsh(int argc, char *argv[]);
};

#endif // !GMSH_SESSION_MANAGER_HPP
