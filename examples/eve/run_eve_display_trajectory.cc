/// @file
///
/// This file creates a simple trajectory and visualize it.

/* Examples

PublishFramesToLcm("DRAKE_DRAW_TRAJECTORY", {
    {"X_WF", Eigen::Isometry3d::Identity()},
    {"X_WG", Eigen::Isometry3d::Identity()},
   }, &lcm);
*/

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/text_logging_gflags.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/lcmt_viewer_draw.hpp"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/examples/eve/eve_common.h"

namespace drake {
namespace examples {
namespace eve {

void DoMain() {
  // Design the trajectory to follow.
  const std::vector<double> kTimes{0.0, 2.0, 5.0, 10.0};
  std::vector<Eigen::MatrixXd> knots(kTimes.size());
  Eigen::VectorXd tmp1(3);
  tmp1 << 0, 0, 0;
  knots[0] = tmp1;
  Eigen::VectorXd tmp2(3);
  tmp2 << 1, 1, 0;
  knots[1] = tmp2;
  Eigen::VectorXd tmp3(3);
  tmp3 << 2, -1, 0;
  knots[2] = tmp3;
  Eigen::VectorXd tmp4(3);
  tmp4 << 3, 0, 0;
  knots[3] = tmp4;
//  Eigen::VectorXd knot_dot_start = Eigen::VectorXd::Zero(3);
//  Eigen::MatrixXd knot_dot_end = Eigen::VectorXd::Zero(3);
//  trajectories::PiecewisePolynomial<double> trajectory =
//      trajectories::PiecewisePolynomial<double>::Cubic(
//          kTimes, knots, knot_dot_start, knot_dot_end);
  trajectories::PiecewisePolynomial<double> trajectory =
      trajectories::PiecewisePolynomial<double>::FirstOrderHold(
          kTimes, knots);

  std::vector<std::string> names;
  std::vector<Eigen::Isometry3d> poses;
  for (double t = 0.0; t < 10.0; t += 0.1) {
    names.push_back("X" + std::to_string(int(t * 100)));
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = trajectory.value(t);
    poses.push_back(pose);
  }
  lcm::DrakeLcm lcm;

  //  std::vector<std::string> names = {"X_WF", "X_WG"};
  //  Eigen::Isometry3d pose1 = Eigen::Isometry3d::Identity();
  //  pose1.translation() = Eigen::Vector3d::Ones()*0.5;
  //  Eigen::Isometry3d pose2 = Eigen::Isometry3d::Identity();
  //  Eigen::Vector3d translation2; translation2 << 1,2,3;
  //  pose1.translation() = translation2;
  //  std::vector<Eigen::Isometry3d> poses = {pose1, pose2};

  PublishFramesToLcm("DRAKE_DRAW_FRAMES", poses, names, &lcm);
}

}  // namespace eve
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "A simple dynamic simulation for the Allegro hand moving under constant"
      " torques.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::eve::DoMain();
  return 0;
}
