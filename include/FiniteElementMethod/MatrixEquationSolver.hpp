#ifndef MATRIX_EQUATION_SOLVER_HPP
#define MATRIX_EQUATION_SOLVER_HPP

/* ATTENTION: Works well only for the polynom order = 1. */

#include "FiniteElementMethod/FEMTypes.hpp"
#include "FiniteElementMethod/GSMAssemblier.hpp"
#include "FiniteElementMethod/LinearAlgebraManagers/VectorManager.hpp"

class MatrixEquationSolver
{
private:
    std::shared_ptr<GSMAssemblier> m_assemblier;     ///< Instance of the matrix assemblier.
    std::shared_ptr<VectorManager> m_solutionVector; ///< Instance of the solution vector.
    Teuchos::RCP<TpetraVectorType> m_rhs;            ///< Right-hand side vector 'b'.
    Teuchos::RCP<TpetraVectorType> m_x;              ///< Solution vector 'x'.
    Teuchos::RCP<TpetraMatrixType> m_A;              ///< Matrix 'A'.

    /// @brief Initializes the matrix, solution, and RHS vectors
    void initialize();

public:
    /// @brief Ctor with params.
    MatrixEquationSolver(std::shared_ptr<GSMAssemblier> assemblier, std::shared_ptr<VectorManager> solutionVector);

    /// @brief Dtor.
    ~MatrixEquationSolver() {}

    /// @brief Sets the RHS vector 'b'.
    void setRHS(Teuchos::RCP<TpetraVectorType> const &rhs);

    /* Getters for the all components of the equation. */
    Teuchos::RCP<TpetraVectorType> getRHS() const { return m_rhs; }
    Teuchos::RCP<TpetraVectorType> getLHS() const { return m_x; }
    Teuchos::RCP<TpetraMatrixType> getGlobalStiffnessMatrix() const { return m_A; }
    Scalar getScalarFieldValueFromX(size_t nodeID) const;
    std::vector<Scalar> getValuesFromX() const;

    /// @brief Fills node potentials to the mesh data.
    void fillNodesPotential();

    /// @brief Calculates and fills mesh data storage of the cumulative electric field vectors for each tetrahedron.
    void calculateElectricField();

    /// @brief Writes solution vector to the .pos file (GMSH format) to have the capability to view results of the solved equation Ax=b.
    void writeElectricPotentialsToPosFile(double time = -1);

    /// @brief Writes solution vector to the .pos file (GMSH format) to have the capability to view results of the solved equation Ax=b.
    void writeElectricFieldVectorsToPosFile(double time = -1);

    /**
     * @brief Creates and configures a parameter list for the specified iterative solver.
     *
     * This method initializes and configures a `Teuchos::ParameterList` with the specified parameters for the given iterative solver.
     * It sets default values for all parameters and handles exceptions that may occur during the parameter setting process.
     *
     * Supported iterative solvers include:
     * - "CG" (Conjugate Gradient)
     * - "Block CG"
     * - "GMRES" (Generalized Minimal Residual)
     * - "Block GMRES"
     * - "Pseudo-block GMRES"
     * - "Block Flexible GMRES"
     * - "GCRO-DR" (Generalized Conjugate Residual with Deflated Restarting)
     * - "Pseudo-block CG"
     * - "LSQR"
     * - "MINRES"
     *
     * @param solverName The name of the iterative solver (e.g., "GMRES", "CG").
     * @param maxIterations The maximum number of iterations the solver will perform (default: 1000).
     * @param convergenceTolerance The tolerance for the relative residual norm used to determine convergence (default: 1e-20).
     * @param verbosity Controls the amount and type of information printed during the solution process (default: Belos::Errors + Belos::Warnings + Belos::IterationDetails).
     * @param outputFrequency Determines how often information is printed during the iterative process (default: 1).
     * @param numBlocks Sets the number of blocks in the Krylov basis, related to the restart mechanism of GMRES (default: 30).
     * @param blockSize Determines the block size for block methods (default: 1).
     * @param maxRestarts Specifies the maximum number of restarts allowed (default: 20).
     * @param flexibleGMRES Indicates whether to use the flexible version of GMRES (default: false).
     * @param orthogonalization Specifies the orthogonalization method to use, options include "ICGS" and "IMGS" (default: "ICGS").
     * @param adaptiveBlockSize Indicates whether to adapt the block size in block methods (default: false).
     * @param convergenceTestFrequency Specifies how often convergence is tested (in iterations). If negative, the default setting is used (default: -1).
     *
     * @return A `Teuchos::RCP` to the configured `Teuchos::ParameterList`.
     */
    Teuchos::RCP<Teuchos::ParameterList> createSolverParams(
        std::string_view solverName = "GMRES",
        int maxIterations = 1000,
        double convergenceTolerance = 1e-20,
        int verbosity = Belos::Errors + Belos::Warnings + Belos::IterationDetails,
        int outputFrequency = 1,
        int numBlocks = 30,
        int blockSize = 1,
        int maxRestarts = 20,
        bool flexibleGMRES = false,
        std::string_view orthogonalization = "ICGS",
        bool adaptiveBlockSize = false,
        int convergenceTestFrequency = -1);

    /**
     * @brief Parses a JSON file to extract solver parameters and configures a Teuchos::ParameterList.
     *
     * This function reads a JSON file, extracts solver parameters, and sets them in a Teuchos::ParameterList.
     * It handles exceptions, checks for file existence, and validates the JSON format.
     * @warning: Deletes the temporary file after parsing.
     *
     * @param filename The name of the JSON file containing solver parameters. By default uses filename that generates by UI side.
     * @return A pair containing the solver name (std::string) and a `Teuchos::RCP<Teuchos::ParameterList>` with the solver parameters.
     * @throws std::runtime_error if the file cannot be opened, does not exist, or is not a valid JSON file.
     */
    std::pair<std::string, Teuchos::RCP<Teuchos::ParameterList>> parseSolverParamsFromJson(std::string_view filename = "temp_solver_params.json");

    /**
     * @brief Solves the linear system Ax = b using an iterative solver.
     *
     * This method configures and runs the iterative solver specified by `solverName`.
     * It sets up the linear problem, applies a preconditioner, and solves the system
     * of equations using the provided solver parameters.
     *
     * @param solverName Name of the iterative solver to use (e.g., "GMRES", "CG").
     * @param solverParams Parameters for the iterative solver, typically created
     *        using the `createSolverParams` method.
     * @throws std::runtime_error If there is a problem during setup or if the solver fails to converge.
     */
    void solve(std::string_view solverName, Teuchos::RCP<Teuchos::ParameterList> solverParams);
};

#endif // !MATRIX_EQUATION_SOLVER_HPP
