#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <iostream>

#include "FiniteElementMethod/BoundaryConditions/VectorBoundaryConditionManager.hpp"
#include "FiniteElementMethod/FEMExceptions.hpp"

extern void supress_output(std::ostream &stream);
extern void restore_output(std::ostream &stream);
extern void execute_without_output(std::function<void()> const &func, std::ostream &stream);

Teuchos::RCP<TpetraVectorType> createTestVector(size_t size)
{
    Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal>> map =
        Teuchos::rcp(new Tpetra::Map<LocalOrdinal, GlobalOrdinal>(size, 0, Teuchos::DefaultComm<int>::getComm()));
    Teuchos::RCP<TpetraVectorType> vector = Teuchos::rcp(new TpetraVectorType(map));
    vector->putScalar(Scalar(0));
    return vector;
}

bool checkVectorBoundaryConditions(Teuchos::RCP<TpetraVectorType> vector,
                                   std::map<GlobalOrdinal, Scalar> const &boundary_conditions,
                                   size_t polynom_order)
{
    auto vectorData = vector->get1dView();

    for (auto const &[nodeInGmsh, value] : boundary_conditions)
    {
        for (size_t j = 0; j < polynom_order; ++j)
        {
            GlobalOrdinal nodeID = (nodeInGmsh - 1) * polynom_order + j;

            Scalar vectorValue = vectorData[nodeID];

            if (vectorValue != value)
            {
                return false;
            }
        }
    }
    return true;
}

// Clean test: checking the setting of boundary conditions
TEST(VectorBoundaryConditionsManagerTest, ValidBoundaryCondition)
{
    Teuchos::RCP<TpetraVectorType> vector = createTestVector(5e5);
    std::map<GlobalOrdinal, Scalar> boundary_conditions{{1, 1.0}, {2, 0.0}};

    VectorBoundaryConditionsManager manager;
    manager.set(vector, 1, boundary_conditions);

    ASSERT_TRUE(checkVectorBoundaryConditions(vector, boundary_conditions, 1));
}

// Dirty Test: Empty vector
TEST(VectorBoundaryConditionsManagerTest, EmptyVector)
{
    Teuchos::RCP<TpetraVectorType> vector;
    std::map<GlobalOrdinal, Scalar> boundary_conditions{{1, 1.0}, {2, 0.0}};

    execute_without_output([&]()
                           {
VectorBoundaryConditionsManager manager;
    EXPECT_NO_THROW(manager.set(vector, 1, boundary_conditions)); }, std::cerr);
}

// Dirty Test: empty boundary conditions
TEST(VectorBoundaryConditionsManagerTest, EmptyBoundaryConditions)
{
    Teuchos::RCP<TpetraVectorType> vector = createTestVector(5);
    std::map<GlobalOrdinal, Scalar> boundary_conditions;

    execute_without_output([&]()
                           {
VectorBoundaryConditionsManager manager;
    EXPECT_NO_THROW(manager.set(vector, 1, boundary_conditions)); }, std::cerr);
}

// Dirty test: node is out of range
TEST(VectorBoundaryConditionsManagerTest, OutOfBoundsNodeID)
{
    Teuchos::RCP<TpetraVectorType> vector = createTestVector(5);
    std::map<GlobalOrdinal, Scalar> boundary_conditions{{10, 1.0}, {11, 0.0}};

    execute_without_output([&]()
                           {
    VectorBoundaryConditionsManager manager;
    EXPECT_THROW(manager.set(vector, 1, boundary_conditions),
        VectorBoundaryConditionsSettingException); }, std::cerr);
}

// Dirty test: boundary conditions with a polynomial of order 2
TEST(VectorBoundaryConditionsManagerTest, PolynomialOrderTest)
{
    Teuchos::RCP<TpetraVectorType> vector = createTestVector(10e5);
    std::map<GlobalOrdinal, Scalar> boundary_conditions{{1, 1.0}, {2, 0.0}};

    VectorBoundaryConditionsManager manager;
    manager.set(vector, 2, boundary_conditions);

    ASSERT_TRUE(checkVectorBoundaryConditions(vector, boundary_conditions, 2));
}
