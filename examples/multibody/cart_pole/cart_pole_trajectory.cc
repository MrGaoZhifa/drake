///
/// This file use a fully actuated cart pole model to track a trajectory.
///

#include <gflags/gflags.h>

#include <drake/systems/framework/event.h>
#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/common/text_logging_gflags.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/controllers/pid_controller.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/framework_common.h"
#include "drake/systems/primitives/affine_system.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/linear_system.h"
#include "drake/systems/primitives/trajectory_source.h"

DEFINE_double(target_realtime_rate, 1.0,
              "Rate at which to run the simulation, relative to realtime");
DEFINE_double(simulation_time, 10, "How long to simulate the pendulum");
DEFINE_double(max_time_step, 1.0e-3,
              "Simulation time step used for integrator.");

namespace drake {
namespace examples {
namespace multibody {
namespace cart_pole {
namespace {

// Fixed path to double pendulum SDF model.
static const char* const kCartPoleSdfPath =
    "drake/examples/multibody/cart_pole/cart_pole.sdf";

//
// Main function for demo.
//
void DoMain() {
  DRAKE_DEMAND(FLAGS_simulation_time > 0);
  logging::HandleSpdlogGflags();

  systems::DiagramBuilder<double> builder;

  geometry::SceneGraph<double>& scene_graph =
      *builder.AddSystem<geometry::SceneGraph>();
  scene_graph.set_name("scene_graph");

  // Load and parse double pendulum SDF from file into a tree.
  drake::multibody::MultibodyPlant<double>* cp =
      builder.AddSystem<drake::multibody::MultibodyPlant<double>>(
          FLAGS_max_time_step);
  cp->set_name("cart_pole");
  cp->RegisterAsSourceForSceneGraph(&scene_graph);

  drake::multibody::Parser parser(cp);
  const std::string sdf_path = FindResourceOrThrow(kCartPoleSdfPath);
  drake::multibody::ModelInstanceIndex plant_model_instance_index =
      parser.AddModelFromFile(sdf_path);
  (void)plant_model_instance_index;

  // Now the plant is complete.
  cp->Finalize();

  // Create PID Controller.
  const Eigen::VectorXd Kp_base = Eigen::VectorXd::Ones(2) * 200.0;
  const Eigen::VectorXd Ki_base = Eigen::VectorXd::Ones(2) * 0.0;
  const Eigen::VectorXd Kd_base = Eigen::VectorXd::Ones(2) * 10.0;
  systems::controllers::PidController<double>* pid_controller =
      builder.AddSystem<systems::controllers::PidController<double>>(
          Kp_base, Ki_base, Kd_base);
  builder.Connect(cp->get_state_output_port(),
                  pid_controller->get_input_port_estimated_state());
  builder.Connect(pid_controller->get_output_port_control(),
                  cp->get_actuation_input_port());

//  // Set desired position [q,v]' for PID as feedback reference.
//  auto desired_base_source =
//      builder.AddSystem<systems::ConstantVectorSource<double>>(
//          Eigen::VectorXd::Zero(2 * 2));
//  builder.Connect(desired_base_source->get_output_port(),
//                  pid_controller->get_input_port_desired_state());

  // Design the trajectory to follow.
  const std::vector<double> kTimes{0.0, 7.0, 10.0};
  std::vector<Eigen::MatrixXd> knots(kTimes.size());
  knots[0] = Eigen::VectorXd::Zero(2);
  knots[1] = Eigen::VectorXd::Ones(2);
  knots[2] = Eigen::VectorXd::Ones(2)*2;

  trajectories::PiecewisePolynomial<double> trajectory =
      trajectories::PiecewisePolynomial<double>::FirstOrderHold(kTimes, knots);
  // Adds a trajectory source for desired state.
  auto traj_src = builder.AddSystem<systems::TrajectorySource<double>>(
      trajectory, 1 /* outputs q + v */);
  traj_src->set_name("trajectory_source");

  builder.Connect(traj_src->get_output_port(),
                  pid_controller->get_input_port_desired_state());

  // Connect plant with scene_graph to get collision information.
  DRAKE_DEMAND(!!cp->get_source_id());
  builder.Connect(
      cp->get_geometry_poses_output_port(),
      scene_graph.get_source_pose_port(cp->get_source_id().value()));
  builder.Connect(scene_graph.get_query_output_port(),
                  cp->get_geometry_query_input_port());

  geometry::ConnectDrakeVisualizer(&builder, scene_graph);

  auto diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();

  // Create plant_context to set velocity.
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(*cp, diagram_context.get());
  // Set init position.
  Eigen::VectorXd positions = Eigen::VectorXd::Zero(2);
  positions[0] = 0.0;
  positions[1] = 0.0;
  cp->SetPositions(&plant_context, positions);

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_publish_every_time_step(true);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);
}

}  // namespace
}  // namespace cart_pole
}  // namespace multibody
}  // namespace examples
}  // namespace drake

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Using LQR controller to control "
      "the fully actuated cart pole model!");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::multibody::cart_pole::DoMain();
  return 0;
}