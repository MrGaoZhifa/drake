/// @file
///
/// This demo sets up a simple dynamic simulation for the Allegro hand using
/// the multi-body library. A single, constant torque is applied to all joints
/// and defined by a command-line parameter. This demo also allows to specify
/// whether the right or left hand is simulated.

#include <gflags/gflags.h>

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
#include <drake/multibody/tree/revolute_spring.h>
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_vector_source.h"
//#include "drake/systems/controllers/inverse_dynamics_controller.h"

namespace drake {
namespace examples {
namespace eve {

using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;

DEFINE_double(constant_load, 0,
              "the constant load on each joint, Unit [Nm]."
              "Suggested load is in the order of 0.01 Nm. When input value"
              "equals to 0 (default), the program runs a passive simulation.");

DEFINE_double(simulation_time, 10,
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

  MultibodyPlant<double>& plant =
      *builder.AddSystem<MultibodyPlant>(FLAGS_max_time_step);
  plant.RegisterAsSourceForSceneGraph(&scene_graph);

  multibody::Parser parser(&plant);
  // const std::string xml_filename =
  // FindResourceOrThrow("drake/manipulation/models/"
  //       "qb_hand_description/package.xml");
  // parser.package_map().AddPackageXml(xml_filename);

  const std::string full_name = FindResourceOrThrow(
      "drake/manipulation/models/eve/"
      "urdf/eve_7dof_arms_relative_base.urdf");

  ModelInstanceIndex plant_model_instance_index =
      parser.AddModelFromFile(full_name);

  // Weld the hand to the world frame
  // const auto& joint_hand_root = plant.GetBodyByName("base");
  // plant.AddJoint<multibody::WeldJoint>("weld_base", plant.world_body(),
  // nullopt,
  //     joint_hand_root, nullopt, Isometry3<double>::Identity());

  // if (!FLAGS_add_gravity) {
  //   plant.mutable_gravity_field().set_gravity_vector(
  //       Eigen::Vector3d::Zero());
  // }

  // Add half space plane and gravity.
  const drake::multibody::CoulombFriction<double> coef_friction_inclined_plane(
      FLAGS_inclined_plane_coef_static_friction,
      FLAGS_inclined_plane_coef_kinetic_friction);
  multibody::benchmarks::inclined_plane::AddInclinedPlaneAndGravityToPlant(
      FLAGS_gravity, 0.0, drake::nullopt, coef_friction_inclined_plane, &plant);

  // Attach Spring at each joint.
  plant.AddForceElement<multibody::RevoluteSpring>(
      dynamic_cast<const drake::multibody::RevoluteJoint<double>&>(
          plant.GetJointByName("j_ankle_x")),
      0, 1500);
  plant.AddForceElement<multibody::RevoluteSpring>(
      dynamic_cast<const drake::multibody::RevoluteJoint<double>&>(
          plant.GetJointByName("j_ankle_y")),
      0, 1500);
  plant.AddForceElement<multibody::RevoluteSpring>(
      dynamic_cast<const drake::multibody::RevoluteJoint<double>&>(
          plant.GetJointByName("j_knee_y")),
      0, 1500);
  plant.AddForceElement<multibody::RevoluteSpring>(
      dynamic_cast<const drake::multibody::RevoluteJoint<double>&>(
          plant.GetJointByName("j_hip_y")),
      0, 1500);
  plant.AddForceElement<multibody::RevoluteSpring>(
      dynamic_cast<const drake::multibody::RevoluteJoint<double>&>(
          plant.GetJointByName("j_hip_x")),
      0, 1500);

  // Now the model is complete.
  plant.Finalize();

//  drake::log()->info(std::to_string(plant.num_actuators()) + " - " +
//                     std::to_string(plant.num_actuated_dofs()));

  // constant force input
  VectorX<double> constant_load_value =
      VectorX<double>::Ones(plant.num_actuators()) * FLAGS_constant_load;

  auto constant_source =
      builder.AddSystem<systems::ConstantVectorSource<double>>(
          constant_load_value);
  constant_source->set_name("constant_source");
  builder.Connect(constant_source->get_output_port(),
                  plant.get_actuation_input_port());

  // Connect plant with scene_graph to get collision information
  DRAKE_DEMAND(!!plant.get_source_id());
  builder.Connect(
      plant.get_geometry_poses_output_port(),
      scene_graph.get_source_pose_port(plant.get_source_id().value()));
  builder.Connect(scene_graph.get_query_output_port(),
                  plant.get_geometry_query_input_port());

  geometry::ConnectDrakeVisualizer(&builder, scene_graph);
  std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();

  // Create a context for this system:
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  // Try set init velocity to zero, but the robot still moves with no torque
  // applied.
  VectorX<double> constant_vel_value =
      VectorX<double>::Zero(plant.num_velocities());
  auto constant_vel_source =
      builder.AddSystem<systems::ConstantVectorSource<double>>(
          constant_vel_value);
  constant_vel_source->set_name("constant_vel_source");
  plant.SetVelocities(&plant_context, constant_vel_value);

  // Set the robot COM position, make sure the robot base is off the ground.
  drake::VectorX<double> positions =
      plant.GetPositions(plant_context, plant_model_instance_index);
  positions[6] = 2;
  plant.SetPositions(&plant_context, positions);

  // Set up simulator.
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