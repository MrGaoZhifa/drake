/// @file
///
/// This demo sets up a humanoid robot eve from halodi robotics. The file shows
/// how to use inverse dynamics controller and pid controller to balance the
/// robot.

#include <gflags/gflags.h>

#include <drake/multibody/tree/prismatic_joint.h>
#include <drake/multibody/tree/revolute_spring.h>
#include <drake/systems/controllers/inverse_dynamics_controller.h>
#include <drake/systems/controllers/pid_controlled_system.h>
#include <drake/systems/primitives/multiplexer.h>
#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/common/text_logging_gflags.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/benchmarks/inclined_plane/inclined_plane_plant.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/multibody/tree/uniform_gravity_field_element.h"
#include "drake/multibody/tree/weld_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/matrix_gain.h"

namespace drake {
namespace examples {
namespace eve {
using drake::multibody::BodyIndex;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;

DEFINE_double(constant_pos, 0.0,
              "the constant load on each joint, Unit [Nm]."
              "Suggested load is in the order of 0.01 Nm. When input value"
              "equals to 0 (default), the program runs a passive simulation.");

DEFINE_double(simulation_time, 3,
              "Desired duration of the simulation in seconds");

DEFINE_bool(use_right_hand, true,
            "Which hand to model: true for right hand or false for left hand");

DEFINE_double(max_time_step, 1.0e-3,
              "Simulation time step used for integrator.");

DEFINE_double(gravity, 9.8, "Value of gravity in the direction of -z.");

DEFINE_double(target_realtime_rate, 1,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");

DEFINE_double(integration_accuracy, 1.0E-6,
              "When time_step = 0 (plant is modeled as a continuous system), "
              "this is the desired integration accuracy.  This value is not "
              "used if time_step > 0 (fixed-time step).");
DEFINE_double(penetration_allowance, 1.0E-5, "Allowable penetration (meters).");
DEFINE_double(stiction_tolerance, 1.0E-5,
              "Allowable drift speed during stiction (m/s).");
DEFINE_double(inclined_plane_angle_degrees, 15.0,
              "Inclined plane angle (degrees), i.e., angle from Wx to Ax.");
DEFINE_double(inclined_plane_coef_static_friction, 0.3,
              "Inclined plane's coefficient of static friction (no units).");
DEFINE_double(inclined_plane_coef_kinetic_friction, 0.3,
              "Inclined plane's coefficient of kinetic friction (no units).  "
              "When time_step > 0, this value is ignored.  Only the "
              "coefficient of static friction is used in fixed-time step.");
DEFINE_double(bodyB_coef_static_friction, 0.3,
              "Body B's coefficient of static friction (no units).");
DEFINE_double(bodyB_coef_kinetic_friction, 0.3,
              "Body B's coefficient of kinetic friction (no units).  "
              "When time_step > 0, this value is ignored.  Only the "
              "coefficient of static friction is used in fixed-time step.");
DEFINE_bool(is_inclined_plane_half_space, true,
            "Is inclined plane a half-space (true) or box (false).");
DEFINE_string(bodyB_type, "sphere",
              "Valid body types are "
              "'sphere', 'block', or 'block_with_4Spheres'");

void DoMain() {
  DRAKE_DEMAND(FLAGS_simulation_time > 0);

  systems::DiagramBuilder<double> builder;

  geometry::SceneGraph<double>& scene_graph =
      *builder.AddSystem<geometry::SceneGraph>();
  scene_graph.set_name("scene_graph");

  // Create real model for simulation and control
  MultibodyPlant<double>& plant =
      *builder.AddSystem<MultibodyPlant>(FLAGS_max_time_step);
  plant.set_name("plant");

  plant.RegisterAsSourceForSceneGraph(&scene_graph);

  multibody::Parser parser(&plant);

  const std::string full_name = FindResourceOrThrow(
      "drake/manipulation/models/eve/"
      "urdf/eve_7dof_arms_relative_base_collision.urdf");

  ModelInstanceIndex plant_model_instance_index =
      parser.AddModelFromFile(full_name);
  (void)plant_model_instance_index;

  // Add half space plane and gravity.
  const drake::multibody::CoulombFriction<double>
    coef_friction_inclined_plane(
        FLAGS_inclined_plane_coef_static_friction,
        FLAGS_inclined_plane_coef_kinetic_friction);
  multibody::benchmarks::inclined_plane::AddInclinedPlaneAndGravityToPlant(
        FLAGS_gravity, 0.0, drake::nullopt, coef_friction_inclined_plane,
        &plant);

  // Now the plant is complete.
  plant.Finalize();

  // Create fake model for InverseDynamicsController
  MultibodyPlant<double> fake_plant(FLAGS_max_time_step);
  fake_plant.set_name("fake_plant");
  multibody::Parser fake_parser(&fake_plant);

  const std::string fake_full_name = FindResourceOrThrow(
      "drake/manipulation/models/eve/"
      "urdf/eve_7dof_arms_relative_base_collision.urdf");

  ModelInstanceIndex fake_plant_model_instance_index =
      fake_parser.AddModelFromFile(fake_full_name);
  (void)fake_plant_model_instance_index;

  // Weld the fake plant to the world frame
  const auto& fake_joint_eve_root = fake_plant.GetBodyByName("base");
  fake_plant.AddJoint<multibody::WeldJoint>(
      "weld_base", fake_plant.world_body(), nullopt, fake_joint_eve_root,
      nullopt, Isometry3<double>::Identity());

  // Now the fake_plant is complete.
  fake_plant.Finalize();

  // Plot and Test the port dimension and numbering.
  drake::log()->info(
      "num_joints: " + std::to_string(plant.num_joints()) +
      ", num_positions: " + std::to_string(plant.num_positions()) +
      ", num_velocities: " + std::to_string(plant.num_velocities()) +
      ", num_actuators: " + std::to_string(plant.num_actuators()));
  drake::log()->info(
      "num_joints: " + std::to_string(fake_plant.num_joints()) +
      ", num_positions: " + std::to_string(fake_plant.num_positions()) +
      ", num_velocities: " + std::to_string(fake_plant.num_velocities()) +
      ", num_actuators: " + std::to_string(fake_plant.num_actuators()));
  int index = 0;
  for (multibody::JointActuatorIndex a(0); a < plant.num_actuators(); ++a) {
    drake::log()->info(std::to_string(index++));
    drake::log()->info(
        "PLANT JOINT: " + plant.get_joint_actuator(a).joint().name() +
        " has actuator " + plant.get_joint_actuator(a).name());

    //    Eigen::VectorXd u_instance(1);
    //    u_instance << 100;
    //    Eigen::VectorXd u = Eigen::VectorXd::Zero(plant.num_actuators());
    //    plant.get_joint_actuator(a).set_actuation_vector(u_instance, &u);
    //    drake::log()->info(u.transpose());

    //    drake::log()->info(
    //        "FAKE PLANT JOINT: " +
    //        fake_plant.get_joint_actuator(a).joint().name() + " has actuator "
    //        + fake_plant.get_joint_actuator(a).name());
  }
  index = 0;
  for (multibody::JointIndex j(0); j < plant.num_joints(); ++j) {
    drake::log()->info(std::to_string(index++));

    drake::log()->info(
        "PLANT JOINT: " + plant.get_joint(j).name() + ", position@ " +
        std::to_string(plant.get_joint(j).position_start()) + ", velocity@ " +
        std::to_string(plant.get_joint(j).velocity_start()));
    if (index <= fake_plant.num_joints())
      drake::log()->info(
          "FAKE PLANT JOINT: " + fake_plant.get_joint(j).name() +
          ", position@" +
          std::to_string(fake_plant.get_joint(j).position_start()) +
          ", velocity@" +
          std::to_string(fake_plant.get_joint(j).velocity_start()));
  }
  drake::log()->info(plant.MakeActuationMatrix());

  // Create InverseDynamicsController using fake_plant.
  const int Q = plant.num_positions();
  const int V = plant.num_velocities();
  const int U = fake_plant.num_actuators();
  const Eigen::VectorXd Kp_ = Eigen::VectorXd::Ones(U) * 0.0;
  const Eigen::VectorXd Ki_ = Eigen::VectorXd::Ones(U) * 0.0;
  const Eigen::VectorXd Kd_ = Eigen::VectorXd::Ones(U) * 0.0;
  auto feed_forward_controller =
      builder
          .AddSystem<systems::controllers::InverseDynamicsController<double>>(
              fake_plant, Kp_, Ki_, Kd_, false);

  // Set desired position [q,v]' for IDC as feedback reference.
  VectorX<double> constant_pos_value =
      VectorX<double>::Ones(2 * U) * FLAGS_constant_pos;
  auto desired_constant_source =
      builder.AddSystem<systems::ConstantVectorSource<double>>(
          constant_pos_value);
  desired_constant_source->set_name("desired_constant_source");
  builder.Connect(desired_constant_source->get_output_port(),
                  feed_forward_controller->get_input_port_desired_state());

  // Select plant states and feed into controller with fake_plant.
  Eigen::MatrixXd feedback_joints_selector =
      Eigen::MatrixXd::Zero(2 * U, Q + V);
  for (multibody::JointIndex j(0); j < plant.num_joints(); ++j) {
    if (plant.get_joint(j).name() == "j_l_wheel_y" || plant.get_joint(j).name() == "j_r_wheel_y")
      continue;
    feedback_joints_selector(fake_plant.get_joint(j).position_start(),
                             plant.get_joint(j).position_start()) = 1;
    feedback_joints_selector(
        fake_plant.get_joint(j).velocity_start() + fake_plant.num_positions(),
        plant.get_joint(j).velocity_start() + plant.num_positions()) = 1;
  }
  drake::log()->info(feedback_joints_selector);
  // Use Gain system to convert plant output to IDC state input
  systems::MatrixGain<double>& select_IDC_states =
      *builder.AddSystem<systems::MatrixGain<double>>(feedback_joints_selector);
  //  builder.Connect(plant.get_state_output_port(),
  //                  feed_forward_controller->get_input_port_estimated_state());
  builder.Connect(plant.get_state_output_port(),
                  select_IDC_states.get_input_port());
  builder.Connect(select_IDC_states.get_output_port(),
                  feed_forward_controller->get_input_port_estimated_state());

  // Select generalized control signal and feed into plant.
  Eigen::MatrixXd generalized_actuation_selector =
      Eigen::MatrixXd::Zero(plant.num_velocities(), U);
  generalized_actuation_selector.bottomRightCorner(U, U) =
      Eigen::MatrixXd::Identity(U, U);
  drake::log()->info(generalized_actuation_selector);
  systems::MatrixGain<double>* select_generalized_actuation_states =
      builder.AddSystem<systems::MatrixGain<double>>(
          generalized_actuation_selector);
  //  builder.Connect(feed_forward_controller->get_output_port_control(),
  //                  plant.get_applied_generalized_force_input_port());
  builder.Connect(feed_forward_controller->get_output_port_control(),
                  select_generalized_actuation_states->get_input_port());
  builder.Connect(select_generalized_actuation_states->get_output_port(),
                  plant.get_applied_generalized_force_input_port());

//  // Create the PID controller for the base.
//  const Eigen::VectorXd Kp_base = Eigen::VectorXd::Ones(2) * 10.0;
//  const Eigen::VectorXd Ki_base = Eigen::VectorXd::Ones(2) * 0.0;
//  const Eigen::VectorXd Kd_base = Eigen::VectorXd::Ones(2) * 0.0;
//  systems::controllers::PidController<double>* pid_controller =
//      builder.AddSystem<systems::controllers::PidController<double>>(
//          Kp_base, Ki_base, Kd_base);
//
//  // Set desired position [q,v]' for PID as feedback reference.
//  auto desired_base_source =
//      builder.AddSystem<systems::ConstantVectorSource<double>>(
//          Eigen::VectorXd::Zero(2 * 2));
//  builder.Connect(desired_base_source->get_output_port(),
//                  pid_controller->get_input_port_desired_state());
//
//  // Select plant states and feed into PID controller.
//  Eigen::MatrixXd feedback_base_selector = Eigen::MatrixXd::Zero(2 * 2, Q + V);
//  feedback_base_selector.topLeftCorner(2, 2) = Eigen::MatrixXd::Identity(2, 2);
//  feedback_base_selector.block<2, 2>(2, Q) = Eigen::MatrixXd::Identity(2, 2);
//  drake::log()->info(feedback_base_selector);
//  systems::MatrixGain<double>& select_PID_states =
//      *builder.AddSystem<systems::MatrixGain<double>>(feedback_base_selector);
//  builder.Connect(plant.get_state_output_port(),
//                  select_PID_states.get_input_port());
//  builder.Connect(select_PID_states.get_output_port(),
//                  pid_controller->get_input_port_estimated_state());
//
//  // Select control signal and feed into plant.
//  Eigen::MatrixXd actuation_selector =
//      Eigen::MatrixXd::Zero(plant.num_actuators(), 2);
//  actuation_selector.bottomRightCorner(2, 2) = Eigen::MatrixXd::Identity(2, 2);
//  drake::log()->info(actuation_selector);
//  systems::MatrixGain<double>* select_actuation_states =
//      builder.AddSystem<systems::MatrixGain<double>>(actuation_selector);
//  builder.Connect(pid_controller->get_output_port_control(),
//                  select_actuation_states->get_input_port());
//  builder.Connect(select_actuation_states->get_output_port(),
//                  plant.get_actuation_input_port());

  // Set zero to plant actuation, we are using generalized actuation instead.
    auto zero_actuation =
        builder.AddSystem<systems::ConstantVectorSource<double>>(
            VectorX<double>::Zero(plant.num_actuators()));
    builder.Connect(zero_actuation->get_output_port(),
                    plant.get_actuation_input_port());

  // Connect plant with scene_graph to get collision information.
  DRAKE_DEMAND(!!plant.get_source_id());
  builder.Connect(
      plant.get_geometry_poses_output_port(),
      scene_graph.get_source_pose_port(plant.get_source_id().value()));
  builder.Connect(scene_graph.get_query_output_port(),
                  plant.get_geometry_query_input_port());

  geometry::ConnectDrakeVisualizer(&builder, scene_graph);

  // Create a context for this diagram and plant.
  std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());
  // Create plant_context to set velocity.
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  // Set the robot COM position, make sure the robot base is off the ground.
    drake::VectorX<double> positions =
        plant.GetPositions(plant_context, plant_model_instance_index);
    positions[6] = 0.6;
    plant.SetPositions(&plant_context, positions);

  // Set robot init velocity for every joint.
  drake::VectorX<double> velocities =
      Eigen::VectorXd::Ones(plant.num_velocities()) * -0.0;
  plant.SetVelocities(&plant_context, velocities);

  // Set up simulator.
  drake::log()->info("\nNow starts Simulation\n");
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_publish_every_time_step(true);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);
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
