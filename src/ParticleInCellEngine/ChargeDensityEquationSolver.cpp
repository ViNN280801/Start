#include "ParticleInCellEngine/ChargeDensityEquationSolver.hpp"
#include "FiniteElementMethod/FEMLimits.hpp"
#include "FiniteElementMethod/MatrixEquationSolver.hpp"

void ChargeDensityEquationSolver::solve(double timeMoment,
                                        std::string_view configFilename,
                                        NodeChargeDensitiesMap &nodeChargeDensityMap,
                                        std::shared_ptr<GSMAssembler> &gsmAssembler,
                                        std::shared_ptr<VectorManager> &solutionVector,
                                        BoundaryConditionsMap &boundaryConditions)
{
    try
    {
        ConfigParser configParser(configFilename);

        auto nonChangebleNodes{configParser.getNonChangeableNodes()};
        for (auto const &[nodeId, nodeChargeDensity] : nodeChargeDensityMap)
#if __cplusplus >= 202002L
            if (std::ranges::find(nonChangebleNodes, nodeId) ==
                nonChangebleNodes.cend())
#else
            if (std::find(nonChangebleNodes.cbegin(), nonChangebleNodes.cend(),
                          nodeId) == nonChangebleNodes.cend())
#endif
                boundaryConditions[nodeId] = nodeChargeDensity;
        BoundaryConditionsManager::set(solutionVector->get(),
                                       FEM_LIMITS_DEFAULT_POLYNOMIAL_ORDER,
                                       boundaryConditions);

        MatrixEquationSolver solver(gsmAssembler, solutionVector);
        auto solverParams{solver.createSolverParams(
            configParser.getSolverName(), configParser.getMaxIterations(),
            configParser.getConvergenceTolerance(), configParser.getVerbosity(),
            configParser.getOutputFrequency(), configParser.getNumBlocks(),
            configParser.getBlockSize(), configParser.getMaxRestarts(),
            configParser.getFlexibleGMRES(),
            configParser.getOrthogonalization(),
            configParser.getAdaptiveBlockSize(),
            configParser.getConvergenceTestFrequency())};
        solver.solve(configParser.getSolverName(), solverParams);
        solver.calculateElectricField(); // Getting electric field for the
                                         // each cell.

        solver.writeElectricPotentialsToPosFile(timeMoment);
        solver.writeElectricFieldVectorsToPosFile(timeMoment);
    }
    catch (std::exception const &ex)
    {
        ERRMSG(util::stringify("Can't solve the equation: ", ex.what()));
    }
    catch (...)
    {
        ERRMSG("Some error occured while solving the matrix equation Ax=b");
    }
}
