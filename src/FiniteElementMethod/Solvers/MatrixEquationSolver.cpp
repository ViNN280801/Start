#include "FiniteElementMethod/Solvers/MatrixEquationSolver.hpp"
#include "FiniteElementMethod/Solvers/SolversExceptions.hpp"
#include "Utilities/Utilities.hpp"

void MatrixEquationSolver::initialize()
{
    m_A = m_assembler->getGlobalStiffnessMatrix();
    m_x = Teuchos::rcp(new TpetraVectorType(m_A->getRowMap()));
    m_rhs = m_solutionVector->get();
    m_x->putScalar(0.0); // Initialize solution vector `x` with zeros.
}

MatrixEquationSolver::MatrixEquationSolver(std::shared_ptr<GSMAssembler> gsmAssembler, std::shared_ptr<VectorManager> solutionVector)
    : m_assembler(gsmAssembler), m_solutionVector(solutionVector) { initialize(); }

void MatrixEquationSolver::setRHS(const Teuchos::RCP<TpetraVectorType> &rhs) { m_rhs = rhs; }

Scalar MatrixEquationSolver::getScalarFieldValueFromX(size_t nodeID) const
{
    if (m_x.is_null())
    {
        START_THROW_EXCEPTION(SolversSolutionVectorNotInitializedException,
                              "Solution vector is not initialized while trying to get scalar field value from it");
    }

    // 1. Calculating initial index for `nodeID` node.
    if (nodeID >= m_x->getLocalLength())
    {
        START_THROW_EXCEPTION(SolversNodeIDOutOfRangeException,
                              util::stringify("Node index ",
                                              nodeID,
                                              " is out of range in the solution vector, max index is ",
                                              m_x->getLocalLength() - 1,
                                              "."));
    }

    Teuchos::ArrayRCP<Scalar const> data{m_x->getData(0)};
    return data[nodeID];
}

std::vector<Scalar> MatrixEquationSolver::getValuesFromX() const
{
    if (m_x.is_null())
    {
        START_THROW_EXCEPTION(SolversSolutionVectorNotInitializedException,
                              "Solution vector is not initialized while trying to get values from it");
    }

    Teuchos::ArrayRCP<Scalar const> data{m_x->getData(0)};
    return std::vector<Scalar>(data.begin(), data.end());
}

void MatrixEquationSolver::fillNodesPotential()
{
    if (m_x.is_null())
    {
        START_THROW_EXCEPTION(SolversSolutionVectorNotInitializedException,
                              "Solution vector is not initialized while trying to fill nodes potential");
    }

    GlobalOrdinal id{1};
    for (Scalar potential : getValuesFromX())
        m_assembler->getMeshManager().assignPotential(id++, potential);
}

void MatrixEquationSolver::calculateElectricField()
{
    try
    {
        fillNodesPotential();
        m_assembler->getMeshManager().computeElectricFields();
    }
    catch (std::exception const &ex)
    {
        std::string errorMessage{util::stringify("Error was occured while trying to calculate electric field: ", ex.what())};
        ERRMSG(errorMessage);
        START_THROW_EXCEPTION(SolversCalculatingElectricFieldException, errorMessage);
    }
    catch (...)
    {
        std::string errorMessage{"Unknown error was occured while trying to calculate electric field"};
        ERRMSG(errorMessage);
        START_THROW_EXCEPTION(SolversUnknownException, errorMessage);
    }
}

void MatrixEquationSolver::writeElectricPotentialsToPosFile(double time)
{
    if (time < 0)
    {
        START_THROW_EXCEPTION(SolversTimeNegativeException,
                              util::stringify("Time can't be negative. Passed time is ",
                                              time,
                                              "."));
    }

    if (m_x.is_null())
    {
        WARNINGMSG("Can't write electric potentials to the .pos file. Solution vector is empty.");
        return;
    }

    try
    {
        // Define the path to the results directory.
        std::filesystem::path resultsDir{"results"};

        // Check if the directory exists, and create it if it does not.
        if (!std::filesystem::exists(resultsDir))
            std::filesystem::create_directories(resultsDir);

        std::string filepath = (time == 0.0)
                                   ? (resultsDir / "electricPotential_time_0.pos").string()
                                   : (resultsDir.string() + "/electricPotential_time_" + std::to_string(time) + ".pos");
        std::ofstream posFile(filepath);

        posFile << "View \"Scalar Field\" {\n";
        for (auto const &entry : m_assembler->getMeshManager().getMeshComponents())
        {
            for (short i{}; i < 4; ++i)
            {
                if (!entry.m_nodes.at(i).m_potential.has_value())
                {
                    WARNINGMSG(util::stringify("Electic potential for the tetrahedron ", entry.m_globalTetraId,
                                               " and node ", entry.m_nodes.at(i).m_globalNodeId, " is empty, skipping it..."));
                    continue;
                }

                auto node{entry.m_nodes.at(i)};
                auto globalNodeId{node.m_globalNodeId};

                double value{getScalarFieldValueFromX(globalNodeId - 1)};
                posFile << "SP("
                        << node.m_nodeCoords.x() << ", "
                        << node.m_nodeCoords.y() << ", "
                        << node.m_nodeCoords.z() << "){"
                        << value << "};\n";
            }
        }
        posFile << "};\n";
        posFile.close();

        LOGMSG(util::stringify("File \'", filepath, "\' was successfully created"));
    }
    catch (std::exception const &ex)
    {
        std::string errorMessage{util::stringify("Error was occured while trying to write electric potentials to the .pos file: ", ex.what())};
        ERRMSG(errorMessage);
        START_THROW_EXCEPTION(SolversWritingElectricPotentialsToPosFileException, errorMessage);
    }
    catch (...)
    {
        std::string errorMessage{"Unknown error was occured while writing results to the .pos file"};
        ERRMSG(errorMessage);
        START_THROW_EXCEPTION(SolversUnknownException, errorMessage);
    }
}

void MatrixEquationSolver::writeElectricFieldVectorsToPosFile(double time)
{
    if (time < 0)
    {
        START_THROW_EXCEPTION(SolversTimeNegativeException,
                              util::stringify("Time can't be negative. Passed time is ",
                                              time,
                                              "."));
    }

    if (m_x.is_null())
    {
        WARNINGMSG("Can't write electric field vectors to the .pos file. Solution vector is empty.");
        return;
    }

    try
    {
        std::filesystem::path resultsDir{"results"};

        // Check if the directory exists, and create it if it does not.
        if (!std::filesystem::exists(resultsDir))
            std::filesystem::create_directories(resultsDir);

        std::string filepath = (time == 0.0)
                                   ? (resultsDir / "electricField_time_0.pos").string()
                                   : (resultsDir.string() + "/electricField_time_" + std::to_string(time) + ".pos");
        std::ofstream posFile(filepath);

        posFile << "View \"Vector Field\" {\n";
        for (auto const &entry : m_assembler->getMeshManager().getMeshComponents())
        {
            if (!entry.m_electricField.has_value())
            {
                WARNINGMSG(util::stringify("Electic field for the tetrahedron ", entry.m_globalTetraId, " is empty"));
                continue;
            }

            auto x{entry.getTetrahedronCenter().x()},
                y{entry.getTetrahedronCenter().y()},
                z{entry.getTetrahedronCenter().z()};

            auto fieldVector{entry.m_electricField.value()};
            posFile << "VP("
                    << x << ", "
                    << y << ", "
                    << z << "){"
                    << fieldVector.x() << ", "
                    << fieldVector.y() << ", "
                    << fieldVector.z() << "};\n";
        }

        posFile << "};\n";
        posFile.close();

        LOGMSG(util::stringify("File \'", filepath, "\' was successfully created"));
    }
    catch (std::exception const &ex)
    {
        std::string errorMessage{util::stringify("Error was occured while trying to write electric field vectors to the .pos file: ", ex.what())};
        ERRMSG(errorMessage);
        START_THROW_EXCEPTION(SolversWritingElectricFieldVectorsToPosFileException, errorMessage);
    }
    catch (...)
    {
        std::string errorMessage{"Unknown error was occured while writing results to the .pos file"};
        ERRMSG(errorMessage);
        START_THROW_EXCEPTION(SolversUnknownException, errorMessage);
    }
}

Teuchos::RCP<Teuchos::ParameterList> MatrixEquationSolver::createSolverParams(std::string_view solverName, int maxIterations,
                                                                              double convergenceTolerance, int verbosity, int outputFrequency, int numBlocks,
                                                                              int blockSize, int maxRestarts, bool flexibleGMRES, std::string_view orthogonalization,
                                                                              bool adaptiveBlockSize, int convergenceTestFrequency)
{
    Teuchos::RCP<Teuchos::ParameterList> params{Teuchos::parameterList()};
    try
    {
        params->set("Maximum Iterations", maxIterations);
        params->set("Convergence Tolerance", convergenceTolerance);
        params->set("Verbosity", verbosity);
        params->set("Output Frequency", outputFrequency);

        if (solverName == "GMRES" || solverName == "Block GMRES")
        {
            params->set("Num Blocks", numBlocks);
            params->set("Block Size", blockSize);
            params->set("Maximum Restarts", maxRestarts);
            params->set("Flexible GMRES", flexibleGMRES);
            params->set("Orthogonalization", orthogonalization.data());
            params->set("Adaptive Block Size", adaptiveBlockSize);
            if (convergenceTestFrequency >= 0)
                params->set("Convergence Test Frequency", convergenceTestFrequency);
        }
        else if (solverName == "CG" || solverName == "Block CG")
        {
            params->set("Block Size", blockSize);
            params->set("Convergence Tolerance", convergenceTolerance);
            params->set("Maximum Iterations", maxIterations);
        }
        else if (solverName == "LSQR")
        {
            params->set("Convergence Tolerance", convergenceTolerance);
            params->set("Maximum Iterations", maxIterations);
        }
        else if (solverName == "MINRES")
        {
            params->set("Convergence Tolerance", convergenceTolerance);
            params->set("Maximum Iterations", maxIterations);
        }
        else
            START_THROW_EXCEPTION(SolversUnsupportedSolverNameException,
                                  util::stringify("Unsupported solver name: ", solverName));
    }
    catch (std::exception const &ex)
    {
        std::string errorMessage{util::stringify("Error was occured while setting parameters for the solver: ", ex.what())};
        ERRMSG(errorMessage);
        START_THROW_EXCEPTION(SolversSettingSolverParametersException, errorMessage);
    }
    catch (...)
    {
        std::string errorMessage{"Unknown error occurred while setting parameters for the solver."};
        ERRMSG(errorMessage);
        START_THROW_EXCEPTION(SolversUnknownException, errorMessage);
    }
    return params;
}

std::pair<std::string, Teuchos::RCP<Teuchos::ParameterList>> MatrixEquationSolver::parseSolverParamsFromJson(std::string_view filename)
{
    if (!std::filesystem::exists(filename))
    {
        START_THROW_EXCEPTION(SolversFileDoesNotExistException,
                              util::stringify("File does not exist: ", filename));
    }

    if (std::filesystem::path(filename).extension() != ".json")
    {
        START_THROW_EXCEPTION(SolversFileIsNotJSONException,
                              util::stringify("File is not a JSON file: ", filename));
    }

    std::ifstream file(filename.data());
    if (!file.is_open())
    {
        START_THROW_EXCEPTION(SolversUnableToOpenFileException,
                              util::stringify("Unable to open file: ", filename));
    }

    json j;
    try
    {
        file >> j;
    }
    catch (json::parse_error const &e)
    {
        START_THROW_EXCEPTION(SolversFailedToParseJSONFileException,
                              util::stringify("Failed to parse JSON file: ", filename, ". Error: ", e.what()));
    }

    Teuchos::RCP<Teuchos::ParameterList> params{Teuchos::rcp(new Teuchos::ParameterList())};
    std::string solverName;

    try
    {
        if (j.contains("solverName"))
            solverName = j.at("solverName").get<std::string>();
        if (j.contains("maxIterations"))
            params->set("Maximum Iterations", std::stoi(j.at("maxIterations").get<std::string>()));
        if (j.contains("convergenceTolerance"))
            params->set("Convergence Tolerance", std::stod(j.at("convergenceTolerance").get<std::string>()));
        if (j.contains("verbosity"))
            params->set("Verbosity", std::stoi(j.at("verbosity").get<std::string>()));
        if (j.contains("outputFrequency"))
            params->set("Output Frequency", std::stoi(j.at("outputFrequency").get<std::string>()));
        if (j.contains("numBlocks"))
            params->set("Num Blocks", std::stoi(j.at("numBlocks").get<std::string>()));
        if (j.contains("blockSize"))
            params->set("Block Size", std::stoi(j.at("blockSize").get<std::string>()));
        if (j.contains("maxRestarts"))
            params->set("Maximum Restarts", std::stoi(j.at("maxRestarts").get<std::string>()));
        if (j.contains("flexibleGMRES"))
            params->set("Flexible GMRES", j.at("flexibleGMRES").get<std::string>() == "true");
        if (j.contains("orthogonalization"))
            params->set("Orthogonalization", j.at("orthogonalization").get<std::string>());
        if (j.contains("adaptiveBlockSize"))
            params->set("Adaptive Block Size", j.at("adaptiveBlockSize").get<std::string>() == "true");
        if (j.contains("convergenceTestFrequency"))
            params->set("Convergence Test Frequency", std::stoi(j.at("convergenceTestFrequency").get<std::string>()));
    }
    catch (json::type_error const &e)
    {
        START_THROW_EXCEPTION(SolversTypeJSONFileException,
                              util::stringify("Type error in JSON file: ", filename, ". Error: ", e.what()));
    }
    return std::make_pair(solverName, params);
}

void MatrixEquationSolver::solve(std::string_view solverName, Teuchos::RCP<Teuchos::ParameterList> solverParams)
{
    try
    {
#ifdef USE_MPI
        Teuchos::ParameterList mueluParams;
        mueluParams.set("verbosity", "high");

        // Use an MPI-capable smoother.
        mueluParams.set("smoother: type", "RELAXATION");
        mueluParams.sublist("smoother: params").set("relaxation: type", "Symmetric Gauss-Seidel");
        mueluParams.sublist("smoother: params").set("relaxation: sweeps", 1);
        mueluParams.sublist("smoother: params").set("relaxation: damping factor", 1.0);

        // Use an appropriate coarse solver.
        mueluParams.set("coarse: type", "RELAXATION");
        mueluParams.sublist("coarse: params").set("relaxation: type", "Symmetric Gauss-Seidel");
        mueluParams.sublist("coarse: params").set("relaxation: sweeps", 1);
        mueluParams.sublist("coarse: params").set("relaxation: damping factor", 1.0);
#endif
        // Initialize the preconditioner with parameters.
        Teuchos::RCP<MueLu::TpetraOperator<Scalar, LocalOrdinal, GlobalOrdinal, Node>> M;
#ifdef USE_MPI
        M = MueLu::CreateTpetraPreconditioner<Scalar, LocalOrdinal, GlobalOrdinal, Node>(m_A, mueluParams);
#else
        M = MueLu::CreateTpetraPreconditioner<Scalar, LocalOrdinal, GlobalOrdinal, Node>(m_A);
#endif

        Teuchos::RCP<Belos::LinearProblem<Scalar, TpetraMultiVector, TpetraOperator>> problem{
            Teuchos::rcp(new Belos::LinearProblem<Scalar, TpetraMultiVector, TpetraOperator>(m_A, m_x, m_rhs))};
        problem->setOperator(m_A);
        problem->setLHS(m_x);
        problem->setRHS(m_rhs);
        problem->setLeftPrec(M);

        if (!problem->setProblem())
        {
            START_THROW_EXCEPTION(SolversFailedToSetUpLinearProblemException,
                                  "Failed to set up the linear problem.");
        }

        Belos::SolverFactory<Scalar, TpetraMultiVector, TpetraOperator> factory;
        Teuchos::RCP<Belos::SolverManager<Scalar, TpetraMultiVector, TpetraOperator>> solver{
            factory.create(solverName.data(), solverParams)};
        solver->setProblem(problem);

        Belos::ReturnType result{solver->solve()};
        if (result == Belos::Converged)
        {
            SUCCESSMSG(util::stringify("Belos solver successfully converged in ", solver->getNumIters(), " iterations"));
        }
        else
        {
            START_THROW_EXCEPTION(SolversFailedToConvergeException,
                                  "Belos solver failed to converge.");
        }
    }
    catch (std::exception const &ex)
    {
        std::string errorMessage{util::stringify("Solver failed while solving the equation: ", ex.what())};
        ERRMSG(errorMessage);
        START_THROW_EXCEPTION(SolversFailedToSolveEquationException, errorMessage);
    }
    catch (...)
    {
        std::string errorMessage{"Solver failed while solving the equation: Unknown error"};
        ERRMSG(errorMessage);
        START_THROW_EXCEPTION(SolversUnknownException, errorMessage);
    }
}
