/// @file
///
/// This demo sets up a simple dynamic simulation for the Allegro hand using
/// the multi-body library. A single, constant torque is applied to all joints
/// and defined by a command-line parameter. This demo also allows to specify
/// whether the right or left hand is simulated.

#include <gflags/gflags.h>

#include <drake/multibody/tree/revolute_spring.h>
#include <drake/systems/controllers/inverse_dynamics_controller.h>
#include <drake/systems/controllers/pid_controlled_system.h>
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
#include <drake/multibody/tree/prismatic_joint.h>
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/matrix_gain.h"

namespace drake {
namespace examples {
namespace eve {

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

DEFINE_bool(add_gravity, false,
            "Indicator for whether terrestrial gravity"
            " (9.81 m/s²) is included or not.");
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
      "sdf/eve_7dof_arms_relative_base_no_collision.sdf");
//      "sdf/eve_2dof_base_no_collision.sdf");
//      "urdf/eve_7dof_arms_relative_base_no_collision.urdf");

  ModelInstanceIndex plant_model_instance_index =
      parser.AddModelFromFile(full_name);
  (void)plant_model_instance_index;

//  // Add half space plane and gravity.
//  const drake::multibody::CoulombFriction<double> coef_friction_inclined_plane(
//      FLAGS_inclined_plane_coef_static_friction,
//      FLAGS_inclined_plane_coef_kinetic_friction);
//  multibody::benchmarks::inclined_plane::AddInclinedPlaneAndGravityToPlant(
//      FLAGS_gravity, 0.0, drake::nullopt, coef_friction_inclined_plane, &plant);
  const Vector3<double> gravity_vector_W(0, 0, -FLAGS_gravity);
  plant.mutable_gravity_field().set_gravity_vector(gravity_vector_W);

  // Weld the plant to the world frame
  Eigen::VectorXd axis = Eigen::Vector3d::UnitX();
  const auto& joint_eve_root = plant.GetBodyByName("base");
  plant.AddJoint<multibody::PrismaticJoint>(
      "slider", plant.world_body(), nullopt, joint_eve_root, nullopt, axis, -1.0, 1.0, 3);

  // Now the model is complete.
  plant.Finalize();

  // Create fake model for InverseDynamicsController
  MultibodyPlant<double> fake_plant(FLAGS_max_time_step);
  fake_plant.set_name("fake_plant");
  multibody::Parser fake_parser(&fake_plant);

  const std::string fake_full_name = FindResourceOrThrow(
      "drake/manipulation/models/eve/"
      "sdf/eve_7dof_arms_relative_base_no_collision.sdf");

  ModelInstanceIndex fake_plant_model_instance_index =
      fake_parser.AddModelFromFile(fake_full_name);
  (void)fake_plant_model_instance_index;

  // Weld the fake plant to the world frame
  const auto& fake_joint_eve_root = fake_plant.GetBodyByName("base");
  fake_plant.AddJoint<multibody::WeldJoint>(
      "weld_base", fake_plant.world_body(), nullopt, fake_joint_eve_root, nullopt,
      Isometry3<double>::Identity());

  // Now the model is complete.
  fake_plant.Finalize();

  // Test the port dimension and numbering.
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
  for (multibody::JointActuatorIndex a(0); a < plant.num_actuators();
       ++a) {
    drake::log()->info(
        "PLANT JOINT: " + plant.get_joint_actuator(a).joint().name() +
        " has actuator " + plant.get_joint_actuator(a).name());
    Eigen::VectorXd u_instance(1);
    u_instance << 100;
    Eigen::VectorXd u = Eigen::VectorXd::Zero(plant.num_actuators());
    plant.get_joint_actuator(a).set_actuation_vector(u_instance, &u);
    drake::log()->info(u.transpose());
//    drake::log()->info(
//        "FAKE PLANT JOINT: " + fake_plant.get_joint_actuator(a).joint().name() +
//        " has actuator " + fake_plant.get_joint_actuator(a).name());
  }
  int index = 0;
  for (multibody::JointIndex j(0); j < plant.num_joints(); ++j) {

    drake::log()->info(std::to_string(index++));

    drake::log()->info(
        "PLANT JOINT: " + plant.get_joint(j).name() + ", position@ " +
        std::to_string(plant.get_joint(j).position_start()) + ", velocity@ " +
        std::to_string(plant.get_joint(j).velocity_start()));
    if (index <= fake_plant.num_joints())
      drake::log()->info(
        "FAKE PLANT JOINT: " + fake_plant.get_joint(j).name() + ", position@" +
        std::to_string(fake_plant.get_joint(j).position_start()) +
        ", velocity@" +
        std::to_string(fake_plant.get_joint(j).velocity_start()));
  }
  drake::log()->info(plant.MakeActuationMatrix());

  const int Q = plant.num_positions();
  const int V = plant.num_velocities();
  const int U = plant.num_actuators();
  const Eigen::VectorXd Kp_ = Eigen::VectorXd::Ones(U) * 0.0;
  const Eigen::VectorXd Ki_ = Eigen::VectorXd::Ones(U) * 0.0;
  const Eigen::VectorXd Kd_ = Eigen::VectorXd::Ones(U) * 0.0;
  Eigen::MatrixXd feedback_selector = Eigen::MatrixXd::Zero(2 * U, Q + V);
  // Try to find mapping between joints of plant and fake_plant.
  for (multibody::JointIndex j(0); j < fake_plant.num_actuators(); ++j) {
    feedback_selector(fake_plant.get_joint(j).position_start(),
                      plant.get_joint(j).position_start()) = 1;
    feedback_selector(fake_plant.get_joint(j).velocity_start() + fake_plant.num_positions(),
                      plant.get_joint(j).velocity_start() + plant.num_positions()) = 1;
  }
  drake::log()->info(feedback_selector);


  // Create InverseDynamicsController
  auto feed_forward_controller = builder
           .AddSystem<systems::controllers::InverseDynamicsController<double>>(
               fake_plant, Kp_, Ki_, Kd_, false);

  // Set desired position [q,v]' for both IDC
  VectorX<double> constant_pos_value =
      VectorX<double>::Ones(2 * U) * FLAGS_constant_pos;
  auto desired_constant_source =
      builder.AddSystem<systems::ConstantVectorSource<double>>(
          constant_pos_value);
  desired_constant_source->set_name("desired_constant_source");

  // Use Gain system to convert plant output to IDC state input
  systems::MatrixGain<double>& select_controlled_states =
      *builder.AddSystem<systems::MatrixGain<double>>(feedback_selector);
  builder.Connect(plant.get_state_output_port(),
                  select_controlled_states.get_input_port());
  builder.Connect(select_controlled_states.get_output_port(),
                  feed_forward_controller->get_input_port_estimated_state());

  Eigen::MatrixXd actuation_selector =
      Eigen::MatrixXd::Zero(plant.num_velocities(), fake_plant.num_actuators());
  actuation_selector.bottomRightCorner(fake_plant.num_actuators(),
                                       fake_plant.num_actuators()) =
      Eigen::MatrixXd::Identity(fake_plant.num_actuators(),
                           fake_plant.num_actuators());
  drake::log()->info(actuation_selector);
  systems::MatrixGain<double>* select_actuation_states =
      builder.AddSystem<systems::MatrixGain<double>>(actuation_selector);
  (void) select_actuation_states;
  builder.Connect(feed_forward_controller->get_output_port_control(),
                  select_actuation_states->get_input_port());
  builder.Connect(select_actuation_states->get_output_port(),
                  plant.get_applied_generalized_force_input_port());

  auto zero_actuation =
      builder.AddSystem<systems::ConstantVectorSource<double>>(
          VectorX<double>::Zero(plant.num_actuators()));
  zero_actuation->set_name("zero_actuation");

  builder.Connect(zero_actuation->get_output_port(),
                  plant.get_actuation_input_port());
//  builder.Connect(feed_forward_controller->get_output_port_control(),
//                  plant.get_applied_generalized_force_input_port());
//  builder.Connect(plant.get_state_output_port(),
//                  feed_forward_controller->get_input_port_estimated_state());
  builder.Connect(desired_constant_source->get_output_port(),
                  feed_forward_controller->get_input_port_desired_state());


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

  // Try set init velocity to zero, but the robot still moves with no torque
  // applied.
//  VectorX<double> constant_vel_value =
//      VectorX<double>::Zero(plant.num_velocities());
//  auto constant_vel_source =
//      builder.AddSystem<systems::ConstantVectorSource<double>>(
//          constant_vel_value);
//  constant_vel_source->set_name("constant_vel_source");
//  plant.SetVelocities(&plant_context, constant_vel_value);

  // Set the robot COM position, make sure the robot base is off the ground.
//  drake::VectorX<double> positions =
//      plant.GetPositions(plant_context, plant_model_instance_index);
//  positions[0] = 0.4;
//  plant.SetPositions(&plant_context, positions);
  drake::VectorX<double> velocities = Eigen::VectorXd::Ones(plant.num_velocities()) * -0.1;
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
