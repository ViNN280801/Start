#ifndef CHARGEDNSITYEQUATIONSOLVER_HPP
#define CHARGEDNSITYEQUATIONSOLVER_HPP

#include "FiniteElementMethod/BoundaryConditions/BoundaryConditionsManager.hpp"
#include "FiniteElementMethod/GSMAssembler.hpp"
#include "FiniteElementMethod/LinearAlgebraManagers/VectorManager.hpp"
#include "PICTypes.hpp"
#include "Utilities/ConfigParser.hpp"

/**
 * @class ChargeDensityEquationSolver
 * @brief A static class providing methods to solve charge density equations.
 *
 * This class offers static methods to process charge density maps, apply boundary conditions,
 * and solve the resulting equations using a configurable matrix equation solver. The solution
 * updates a vector manager, reflecting changes in node potentials and fields.
 */
class ChargeDensityEquationSolver
{
public:
	/**
	 * @brief Solves the charge density equation system at a given time moment.
	 *
	 * This method processes the provided node charge density map and boundary conditions,
	 * constructs the system of equations using the specified assembler, and solves the system.
	 * It also handles writing outputs such as electric potentials and fields to files.
	 *
	 * @param timeMoment The simulation time for which the system is solved.
	 * @param configFilename The configuration file specifying solver parameters and FEM settings.
	 * @param nodeChargeDensityMap A reference to a map containing charge density values per node.
	 * @param gsmAssembler A shared pointer to the GSM assembler that constructs the system matrix.
	 * @param solutionVector A shared pointer to the vector manager storing the computed solution.
	 * @param boundaryConditions A reference to a map defining boundary conditions for specific nodes.
	 *
	 * @throws std::runtime_error If a critical error occurs while parsing the configuration or solving equations.
	 * @throws std::exception For any generic error during processing.
	 */
	static void solve(double timeMoment, std::string_view configFilename,
					  NodeChargeDensitiesMap &nodeChargeDensityMap,
					  std::shared_ptr<GSMAssembler> &gsmAssembler,
					  std::shared_ptr<VectorManager> &solutionVector,
					  BoundaryConditionsMap &boundaryConditions);
};

#endif // !CHARGEDNSITYEQUATIONSOLVER_HPP
