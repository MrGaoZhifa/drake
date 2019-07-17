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

}  // namespace eve
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "A simple mathematical program example, for later used on eve.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::eve::DoMain();
  return 0;
}
