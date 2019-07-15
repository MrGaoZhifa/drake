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

#include <chrono>
#include <thread>

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
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/systems/framework/event_status.h"
#include "drake/systems/controllers/test/zmp_test_util.h"

DEFINE_double(target_realtime_rate, 1.0,
              "Rate at which to run the simulation, relative to realtime");
DEFINE_double(simulation_time, 10, "How long to simulate.");
DEFINE_double(max_time_step, 1.0e-3,
              "Simulation time step used for integrator.");

namespace drake {
namespace examples {
namespace eve {

class DisplayTrajectoryInSim : public systems::LeafSystem<double> {
 public:
  DisplayTrajectoryInSim(lcm::DrakeLcm& lcm) : lcm_(lcm) {
    this->DeclareVectorInputPort(
        "Trajectory", systems::BasicVector<double>(6));

    DeclarePerStepPublishEvent(
        &DisplayTrajectoryInSim::MyPublishHandler);
  }

  void remap_output(const systems::Context<double>& context,
                    systems::BasicVector<double>* output_vector) const {
    auto input_value = this->EvalVectorInput(context, 0)->get_value();
    drake::log()->info(input_value.transpose());

    std::vector<std::string> names;
    std::vector<Eigen::Isometry3d> poses;
    names.push_back("Time_" + std::to_string(context.get_time()));

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = input_value.segment<3>(0);
    poses.push_back(pose);

    PublishFramesToLcm("DRAKE_DRAW_FRAMES", poses, names, &lcm_);
//    PublishContactToLcm(contact_points, contact_forces, &lcm);

    auto output_value = output_vector->get_mutable_value();
    output_value = input_value;
  }

 private:
  lcm::DrakeLcm& lcm_;

  systems::EventStatus MyPublishHandler(const systems::Context<double>& context) const {
    MySuccessfulPublishHandler(context);
    return systems::EventStatus::Succeeded();
  }

  void MySuccessfulPublishHandler(const systems::Context<double>& context) const {
    auto input_value = this->EvalVectorInput(context, 0)->get_value();
    drake::log()->info(input_value.transpose());

    std::vector<std::string> names;
    std::vector<Eigen::Isometry3d> poses;
    names.push_back("Time_" + std::to_string(context.get_time()));

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = input_value.segment<3>(0);
    poses.push_back(pose);

    PublishFramesToLcm("DRAKE_DRAW_FRAMES", poses, names, &lcm_);

    std::vector<Eigen::VectorXd> contact_points;
    std::vector<Eigen::VectorXd> contact_forces;
    contact_points.push_back(input_value.segment(0,3));
    contact_forces.push_back(input_value.segment(3,3));

    PublishContactToLcm(contact_points, contact_forces, &lcm_);
  }
};

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

  // Send Trajectory frame to viz.
  //  std::vector<std::string> names = {"X_WF", "X_WG"};
  //  Eigen::Isometry3d pose1 = Eigen::Isometry3d::Identity();
  //  pose1.translation() = Eigen::Vector3d::Ones()*0.5;
  //  Eigen::Isometry3d pose2 = Eigen::Isometry3d::Identity();
  //  Eigen::Vector3d translation2; translation2 << 1,2,3;
  //  pose1.translation() = translation2;
  //  std::vector<Eigen::Isometry3d> poses = {pose1, pose2};

  PublishFramesToLcm("DRAKE_DRAW_FRAMES", poses, names, &lcm);

  // Send Trajectory arrow to viz.
  std::vector<Eigen::VectorXd> contact_points;
  std::vector<Eigen::VectorXd> contact_forces;
  contact_points.push_back(Eigen::VectorXd::Zero(3));
  contact_points.push_back(Eigen::VectorXd::Ones(3));
  contact_forces.push_back(Eigen::VectorXd::Ones(3));
  contact_forces.push_back(Eigen::VectorXd::Ones(3)*-1);

  PublishContactToLcm(contact_points, contact_forces, &lcm);
}

void DoMain2() {
  logging::HandleSpdlogGflags();

  systems::DiagramBuilder<double> builder;

  geometry::SceneGraph<double>& scene_graph =
      *builder.AddSystem<geometry::SceneGraph>();
  scene_graph.set_name("scene_graph");

  // Design the trajectory to follow.
  const std::vector<double> kTimes{0.0, 2.0, 7.0, 10.0};
  std::vector<Eigen::MatrixXd> knots(kTimes.size());
  knots[0] = Eigen::Vector3d(0,0,0);
  knots[1] = Eigen::Vector3d(1,1,0);
  knots[2] = Eigen::Vector3d(2,-1,0);
  knots[3] = Eigen::Vector3d(3,0,0);
//  trajectories::PiecewisePolynomial<double> trajectory =
//      trajectories::PiecewisePolynomial<double>::FirstOrderHold(kTimes, knots);
  Eigen::VectorXd knot_dot_start = Eigen::VectorXd::Zero(3);
  Eigen::MatrixXd knot_dot_end = Eigen::VectorXd::Zero(3);
  trajectories::PiecewisePolynomial<double> trajectory =
      trajectories::PiecewisePolynomial<double>::Cubic(
          kTimes, knots, knot_dot_start, knot_dot_end);

  // Adds a trajectory source for desired state.
  auto traj_src = builder.AddSystem<systems::TrajectorySource<double>>(
      trajectory, 1 /* outputs q + v */);
  traj_src->set_name("trajectory_source");

  lcm::DrakeLcm lcm;
  auto displayer = builder.AddSystem<DisplayTrajectoryInSim>(lcm);
  builder.Connect(traj_src->get_output_port(), displayer->get_input_port(0));


  auto diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_publish_every_time_step(true);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
//  simulator.AdvanceTo(FLAGS_simulation_time);
  for (int i = 0; i<100; i++)
    simulator.AdvanceTo(0.1*i);
}

// Visualize the
void DoMain3() {
  std::vector<Eigen::Vector2d> footsteps = {
      Eigen::Vector2d(0, 0),    Eigen::Vector2d(0.5, 0.1),
      Eigen::Vector2d(1, -0.1), Eigen::Vector2d(1.5, 0.1),
      Eigen::Vector2d(2, -0.1), Eigen::Vector2d(2.5, 0)};

  std::vector<trajectories::PiecewisePolynomial<double>> zmp_trajs =
      systems::controllers::GenerateDesiredZMPTrajs(footsteps, 0.5, 1);

  Eigen::Vector4d x0(0, 0, 0, 0);
  double z = 1;

  systems::controllers::ZMPPlanner zmp_planner;
  zmp_planner.Plan(zmp_trajs[1], x0, z);

  double sample_dt = 0.01;

  // Perturb the initial state a bit.
  x0 << 0, 0, 0.2, -0.1;
  systems::controllers::ZMPTestTraj result =
      systems::controllers::SimulateZMPPolicy(zmp_planner, x0, sample_dt, 2);

  lcm::DrakeLcm lcm;

  std::vector<std::string> names;
  std::vector<Eigen::Isometry3d> poses;
  const int N = result.time.size();
  for (int t = 0; t < N; t++) {
    names.push_back("CoM" + std::to_string(int(t)));
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = Eigen::Vector3d(result.nominal_com(0, t), result.nominal_com(1, t), z);
    poses.push_back(pose);
  }
  for (int t = 0; t < N; t++) {
    names.push_back("ZMP" + std::to_string(int(t)));
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = Eigen::Vector3d(result.desired_zmp(0, t), result.desired_zmp(1, t), 0);
    poses.push_back(pose);
  }
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
  drake::examples::eve::DoMain3();
  return 0;
}
