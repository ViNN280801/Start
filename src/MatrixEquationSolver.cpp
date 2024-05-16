#include "../include/FiniteElementMethod/MatrixEquationSolver.hpp"
#include "../include/Utilities/Utilities.hpp"

void MatrixEquationSolver::initialize() {
  m_A = m_assemblier.getGlobalStiffnessMatrix();
  m_x = Teuchos::rcp(new TpetraVectorType(m_A->getRowMap()));
  m_rhs = m_solutionVector.getSolutionVector();
  m_x->putScalar(0.0); // Initialize solution vector `x` with zeros.
}

MatrixEquationSolver::MatrixEquationSolver(GSMatrixAssemblier const &assemblier,
                                           SolutionVector const &solutionVector)
    : m_assemblier(assemblier), m_solutionVector(solutionVector) {
  initialize();
}

void MatrixEquationSolver::setRHS(const Teuchos::RCP<TpetraVectorType> &rhs) {
  m_rhs = rhs;
}

Scalar MatrixEquationSolver::getScalarFieldValueFromX(size_t nodeID) const {
  if (m_x.is_null())
    throw std::runtime_error("Solution vector is not initialized");

  // 1. Calculating initial index for `nodeID` node.
  short polynomOrder{m_assemblier.getPolynomOrder()};
  size_t actualIndex{nodeID * polynomOrder};
  if (actualIndex >= m_x->getLocalLength())
    throw std::runtime_error(
        util::stringify("Node index ", actualIndex,
                        " is out of range in the solution vector."));

  Teuchos::ArrayRCP<Scalar const> data{m_x->getData(0)};
  Scalar value{};

  // 2. Accumulating values for all DOFs.
  if (polynomOrder == 1)
    value = data[actualIndex];
  if (polynomOrder > 1) {
    for (int i{}; i < polynomOrder; ++i)
      value += data[actualIndex + i];
    value /= polynomOrder;
  }

  return value;
}

std::vector<Scalar> MatrixEquationSolver::getValuesFromX() const {
  if (m_x.is_null())
    throw std::runtime_error("Solution vector is not initialized");

  Teuchos::ArrayRCP<Scalar const> data{m_x->getData(0)};
  return std::vector<Scalar>(data.begin(), data.end());
}

std::map<GlobalOrdinal, Scalar>
MatrixEquationSolver::getNodePotentialMap() const {
  if (m_x.is_null())
    throw std::runtime_error("Solution vector is not initialized");

  std::map<GlobalOrdinal, Scalar> nodePotentialMap;
  GlobalOrdinal id{1};
  for (Scalar potential : getValuesFromX())
    nodePotentialMap[id++] = potential;
  return nodePotentialMap;
}

std::map<GlobalOrdinal, MathVector>
MatrixEquationSolver::getElectricFieldMap() const {
  try {
    std::map<GlobalOrdinal, MathVector> electricFieldMap;
    auto basisFuncGradientsMap{m_assemblier.getBasisFuncGradsMap()};
    if (basisFuncGradientsMap.empty()) {
      WARNINGMSG("There is no basis function gradients");
      return electricFieldMap;
    }

    auto nodePotentialsMap{getNodePotentialMap()};
    if (nodePotentialsMap.empty()) {
      WARNINGMSG("There are no potentials in nodes. Maybe you forget to "
                 "calculate the matrix equation Ax=b");
      return electricFieldMap;
    }

    // We have map: (Tetrahedron ID | map<Node ID | Basis function gradient math
    // vector (3 components)>). To get electric field of the cell we just need
    // to accumulate all the basis func grads for each node for each
    // tetrahedron: E_cell = -Σ(φi⋅∇φi), where i - global index of the node.
    for (auto const &[tetraId, basisFuncGrads] : basisFuncGradientsMap) {
      MathVector E_cell;
      for (auto const &[nodeId, basisFuncGrad] : basisFuncGrads)
        E_cell += nodePotentialsMap.at(nodeId) * basisFuncGrad;
      electricFieldMap[tetraId] = -E_cell;
    }

    if (electricFieldMap.empty())
      WARNINGMSG("Something went wrong: There is no electric field in the mesh")
    return electricFieldMap;
  } catch (std::exception const &ex) {
    ERRMSG(ex.what());
  } catch (...) {
    ERRMSG("Unknown error was occured while writing results to the .pos file");
  }
  WARNINGMSG("Returning empty electric field map");
  return std::map<GlobalOrdinal, MathVector>();
}

void MatrixEquationSolver::writeElectricPotentialsToPosFile() const {
  if (m_x.is_null()) {
    WARNINGMSG("There is nothing to show. Solution vector is empty.");
    return;
  }

  try {
    std::ofstream posFile("electricPotential.pos");
    posFile << "View \"Scalar Field\" {\n";
    auto nodes{m_assemblier.getNodes()};
    for (auto const &[nodeID, coords] : nodes) {
      double value{getScalarFieldValueFromX(nodeID - 1)};
      posFile << std::format("SP({}, {}, {}){{{}}};\n", coords[0], coords[1],
                             coords[2], value);
    }
    posFile << "};\n";
    posFile.close();
    LOGMSG("File 'electricPotential.pos' was successfully created");
  } catch (std::exception const &ex) {
    ERRMSG(ex.what());
  } catch (...) {
    ERRMSG("Unknown error was occured while writing results to the .pos file");
  }
}

void MatrixEquationSolver::writeElectricFieldVectorsToPosFile() const {
  if (m_x.is_null()) {
    WARNINGMSG("There is nothing to show. Solution vector is empty.");
    return;
  }

  auto tetrahedronCentres{m_assemblier.getTetrahedronCentres()};
  if (tetrahedronCentres.empty()) {
    WARNINGMSG("There is nothing to show. Storage for the tetrahedron centres "
               "is empty.");
    return;
  }

  auto electricFieldMap{getElectricFieldMap()};
  if (electricFieldMap.empty()) {
    WARNINGMSG("There is nothing to show. Storage for the electric field "
               "values is empty.");
    return;
  }

  try {
    std::ofstream posFile("electricField.pos");
    posFile << "View \"Vector Field\" {\n";
    for (auto const &[tetraId, fieldVector] : electricFieldMap) {
      auto x{tetrahedronCentres.at(tetraId).at(0)},
          y{tetrahedronCentres.at(tetraId).at(1)},
          z{tetrahedronCentres.at(tetraId).at(2)};

      posFile << std::format("VP({}, {}, {}){{{}, {}, {}}};\n", x, y, z,
                             fieldVector.getX(), fieldVector.getY(),
                             fieldVector.getZ());
    }

    posFile << "};\n";
    posFile.close();
    LOGMSG("File 'electricField.pos' was successfully created");
  } catch (std::exception const &ex) {
    ERRMSG(ex.what());
  } catch (...) {
    ERRMSG("Unknown error was occured while writing results to the .pos file");
  }
}

bool MatrixEquationSolver::solve() {
  try {
    auto problem{Teuchos::rcp(
        new Belos::LinearProblem<Scalar, TpetraMultiVector, TpetraOperator>())};
    problem->setOperator(m_A);
    problem->setLHS(m_x);
    problem->setRHS(m_rhs);

    if (!problem->setProblem())
      return false;

    Belos::SolverFactory<Scalar, TpetraMultiVector, TpetraOperator> factory;
    auto solver{factory.create("CG", Teuchos::parameterList())};
    solver->setProblem(problem);

    Belos::ReturnType result{solver->solve()};
    return (result == Belos::Converged);
  } catch (std::exception const &ex) {
    ERRMSG(ex.what());
  } catch (...) {
    ERRMSG("Solver: Unknown error was occured while trying to solve equation");
  }
  return false;
}

void MatrixEquationSolver::solveAndPrint() {
  try {
    auto problem{Teuchos::rcp(
        new Belos::LinearProblem<Scalar, TpetraMultiVector, TpetraOperator>())};
    problem->setOperator(m_A);
    problem->setLHS(m_x);
    problem->setRHS(m_rhs);

    if (!problem->setProblem())
      ERRMSG("Can't set the problem. Belos::LinearProblem::setProblem() "
             "returned an error");

    Belos::SolverFactory<Scalar, TpetraMultiVector, TpetraOperator> factory;
    auto solver{factory.create("GMRES", Teuchos::parameterList())};
    solver->setProblem(problem);

    Belos::ReturnType result{solver->solve()};

    if (result == Belos::Converged) {
      LOGMSG("\033[1;32mSolution converged\033[0m\033[1m");
    } else {
      ERRMSG("Solution did not converge");
    }
  } catch (std::exception const &ex) {
    ERRMSG(ex.what());
  } catch (...) {
    ERRMSG("Solver: Unknown error occured");
  }
}

void MatrixEquationSolver::printRHS() const { m_solutionVector.print(); }

void MatrixEquationSolver::printLHS() const {
  auto x{SolutionVector(m_solutionVector.size())};
  x.setSolutionVector(m_x);
  x.print();
}
