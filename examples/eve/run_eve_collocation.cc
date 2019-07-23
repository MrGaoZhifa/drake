/// @file
///
/// This file create a eve base plant and try to using 
/// direct collocation to get the trajectory and track the trajectory.

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
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include <drake/multibody/plant/externally_applied_spatial_force.h>
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
DEFINE_bool(is_inclined_plane_half_space, true,
            "Is inclined plane a half-space (true) or box (false).");
DEFINE_double(init_height, 0.2, "Initial height for base.");


class VelocityController : public systems::LeafSystem<double> {
 public:
  VelocityController(MultibodyPlant<double>& plant,
                  ModelInstanceIndex plant_instance)
      : plant_(plant), plant_instance_(plant_instance) {
    this->DeclareVectorInputPort(
        "input1", systems::BasicVector<double>(17));
    this->DeclareVectorOutputPort(
        "output1", systems::BasicVector<double>(plant.num_actuators()),
        &VelocityController::remap_output);
  }

  void remap_output(const systems::Context<double>& context,
                    systems::BasicVector<double>* output_vector) const {
    auto output_value = output_vector->get_mutable_value();
    auto input_value = this->EvalVectorInput(context, 0)->get_value();
    (void) input_value;
    output_value[0] = FLAGS_constant_pos;
    output_value[1] = FLAGS_constant_pos;

    drake::log()->info(input_value.transpose());
    drake::log()->info(output_value.transpose());
    drake::log()->info("\n");

//    plant_.SetVelocitiesInArray(plant_instance_, input_value, &output_value);
  }

 private:
  MultibodyPlant<double>& plant_;
  ModelInstanceIndex plant_instance_;
};

void DoMain() {
  DRAKE_DEMAND(FLAGS_simulation_time > 0);

  systems::DiagramBuilder<double> builder;

  geometry::SceneGraph<double>& scene_graph =
      *builder.AddSystem<geometry::SceneGraph>();
  scene_graph.set_name("scene_graph");

  // Create real model for simulation and control
  MultibodyPlant<double>* plant =
      builder.AddSystem<MultibodyPlant>(FLAGS_max_time_step);
  plant->set_name("plant");

  plant->RegisterAsSourceForSceneGraph(&scene_graph);

  multibody::Parser parser(plant);

  const std::string full_name = FindResourceOrThrow(
      "drake/manipulation/models/eve/"
      "urdf/eve_0dof_base.urdf");

  ModelInstanceIndex plant_model_instance_index =
      parser.AddModelFromFile(full_name);
  (void)plant_model_instance_index;

  // Add half space plane and gravity.
  const drake::multibody::CoulombFriction<double> coef_friction_inclined_plane(
      FLAGS_inclined_plane_coef_static_friction,
      FLAGS_inclined_plane_coef_kinetic_friction);
  multibody::benchmarks::inclined_plane::AddInclinedPlaneAndGravityToPlant(
      FLAGS_gravity, 0.0, drake::nullopt, coef_friction_inclined_plane, plant);
  // Now the plant is complete.
  plant->Finalize();
  // Publish contact results for visualization.
  ConnectContactResultsToDrakeVisualizer(&builder, *plant);

  int index = 0;
  for (multibody::JointIndex j(0); j < plant->num_joints(); ++j) {
    drake::log()->info(std::to_string(index++));
    drake::log()->info(
        "PLANT JOINT: " + plant->get_joint(j).name() + ", position@ " +
            std::to_string(plant->get_joint(j).position_start()) + ", velocity@ " +
            std::to_string(plant->get_joint(j).velocity_start()));
  }

  // Create the WheelController.
  auto vc = builder.AddSystem<VelocityController>(*plant, plant_model_instance_index);
  builder.Connect(plant->get_state_output_port(), vc->get_input_port(0));
  builder.Connect(vc->get_output_port(0), plant->get_actuation_input_port());


  // Connect plant with scene_graph to get collision information.
  DRAKE_DEMAND(!!plant->get_source_id());
  builder.Connect(
      plant->get_geometry_poses_output_port(),
      scene_graph.get_source_pose_port(plant->get_source_id().value()));
  builder.Connect(scene_graph.get_query_output_port(),
                  plant->get_geometry_query_input_port());

  geometry::ConnectDrakeVisualizer(&builder, scene_graph);

  // Create a context for this diagram and plant.
  std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
      
  // Create plant_context to set velocity.
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(*plant, diagram_context.get());
  // Set the robot COM position, make sure the robot base is off the ground.
  Eigen::VectorXd positions = Eigen::VectorXd::Zero(plant->num_positions());
  positions[0] = 1.0;
  positions[6] = 0.130256;
  plant->SetPositions(&plant_context, positions);

//  // Set robot init velocity for every joint.
//  drake::VectorX<double> velocities =
//      Eigen::VectorXd::Ones(plant->num_velocities()) * -0.1;
//  plant->SetVelocities(&plant_context, velocities);

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
      "A simple differential wheel demo.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::eve::DoMain();
  return 0;
}
