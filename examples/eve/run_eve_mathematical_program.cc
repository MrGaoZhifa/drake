/// @file
///
/// This file formulate a mathematical program and solves it.

#include <gflags/gflags.h>

#include "drake/solvers/linear_system_solver.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace examples {
namespace eve {

void DoMain() {
  // linear programming in symbolic form.
  solvers::MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<2>();
  prog.AddLinearConstraint(3 * x(0) == 1 && 2 * x(0) + x(1) == 2 &&
      x(0) - x(1) == 0);
  const solvers::MathematicalProgramResult result = Solve(prog);
  DRAKE_THROW_UNLESS(result.is_success() == false);
  Eigen::Vector2d x_expected(12.0 / 27, 21.0 / 27);
  DRAKE_THROW_UNLESS((result.GetSolution(x)-x_expected).norm() < 1e-10);
  DRAKE_THROW_UNLESS(result.get_optimal_cost() == solvers::MathematicalProgram::kGlobalInfeasibleCost);
}

void DoMain2() {
  // Solve A*x=b using the mathematical programming.
  solvers::MathematicalProgram prog;
  Eigen::MatrixXd A(3,2);
  A << 1, 0,
       0, 1,
       0, 0;
  auto X_ = prog.NewContinuousVariables<2>("X");
  Eigen::Vector3d b(4,5,6);
  prog.AddLinearEqualityConstraint(A * X_, b);

  // Add quadratic cost.
  Eigen::Matrix<double, 2, 2> Q =
      100 * Eigen::Matrix<double, 2, 2>::Identity();
  Eigen::Matrix<double, 2, 1> c;
  c << 0, 0;
  prog.AddQuadraticCost(Q, c, X_);

  //Solve
  const solvers::MathematicalProgramResult result = Solve(prog);

  // Check result
  auto X_value = result.GetSolution(X_);
  DRAKE_THROW_UNLESS(result.is_success() == false);
  DRAKE_THROW_UNLESS((X_value-b.head(2)).norm() < 1e-12);
}

}  // namespace eve
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "A simple mathematical program example, for later used on eve.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::eve::DoMain2();
  return 0;
}
