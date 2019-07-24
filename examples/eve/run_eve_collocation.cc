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
#include "drake/examples/eve/eve_common.h"
#include "drake/lcmt_viewer_draw.hpp"
#include "drake/systems/primitives/trajectory_source.h"

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

DEFINE_double(K1, 1, "The feedback for forward velocity.");
DEFINE_double(K2, 1, "The feedback for rotational velocity.");
DEFINE_double(K3, 1, "The feedback for rotational velocity");

class VelocitySource : public systems::LeafSystem<double> {
 public:
  VelocitySource(MultibodyPlant<double>& plant,
                  ModelInstanceIndex plant_instance, lcm::DrakeLcm& lcm)
      : plant_(plant), plant_instance_(plant_instance), lcm_(lcm)  {
    this->DeclareVectorInputPort(
        "base_state", systems::BasicVector<double>(plant_.num_multibody_states()));
    this->DeclareVectorInputPort(
        "desired_traj", systems::BasicVector<double>(6));
    this->DeclareVectorOutputPort(
        "output1", systems::BasicVector<double>(4),
        &VelocitySource::remap_output);
  }

  void remap_output(const systems::Context<double>& context,
                    systems::BasicVector<double>* output_vector) const {
    auto output_value = output_vector->get_mutable_value();
    auto state_value = this->EvalVectorInput(context, 0)->get_value();
    auto desired_traj_value = this->EvalVectorInput(context, 1)->get_value();

    // Visualize frame attached to base.
    std::vector<std::string> names;
    std::vector<Eigen::Isometry3d> poses;
    names.push_back("base_state");
    math::RollPitchYawd rpy_base(Eigen::Quaterniond(state_value[0], state_value[1], state_value[2], state_value[3]));

    Eigen::Isometry3d pose = Eigen::Translation3d(Eigen::Vector3d(state_value[4], state_value[5], 0.130256)) * 
                             Eigen::AngleAxisd(rpy_base.yaw_angle(), Eigen::Vector3d::UnitZ());
    poses.push_back(pose);

    PublishFramesToLcm("DRAKE_DRAW_FRAMES", poses, names, &lcm_);
    
    Eigen::Vector3d state{rpy_base.yaw_angle(), state_value[4], state_value[5]};
    Eigen::Vector3d desired_state{std::atan2(desired_traj_value[3], desired_traj_value[2]), desired_traj_value[0], desired_traj_value[1]};
    
    Eigen::Matrix3d kinematic_constraint_matrix;
    kinematic_constraint_matrix << 1, 0, 0,
                                   0, std::cos(desired_state[0]), std::sin(desired_state[0]),
                                   0, -std::sin(desired_state[0]), std::cos(desired_state[0]);

    Eigen::Vector3d state_error = kinematic_constraint_matrix * (state - desired_state); (void)state_error;
    Eigen::Vector2d feedforward_velocity{desired_traj_value.segment<2>(2).norm(), desired_traj_value.tail(2).dot(desired_traj_value.segment<2>(2).normalized())};
    
    // Modern Robotics P468 Eq.13.31
    Eigen::Vector2d actual_velocity_input = feedforward_velocity - Eigen::Vector2d{
      FLAGS_K1 * feedforward_velocity[0] * (state_error[1] + state_error[2] * std::tanh(state_error[0])) / std::cos(state_error[0]),
      (FLAGS_K2 * feedforward_velocity[0] * state_error[2] + FLAGS_K3 * feedforward_velocity[0] * std::tanh(state_error[0])) * std::pow(std::cos(state_error[0]),2)
    };
    // Eigen::Vector2d actual_velocity_input = feedforward_velocity;
    const double l = 0.26983;
    const double r = 0.15;
    Eigen::Vector2d actual_wheel_velocity{actual_velocity_input[0] - actual_velocity_input[1] * l, // left wheel velocity
      actual_velocity_input[0] + actual_velocity_input[1] * l};// right wheel velocity

    output_value.setZero();
    output_value.head(2) = actual_wheel_velocity / r;

    drake::log()->info("Desired velocity and acceleration:");
    drake::log()->info(output_value.transpose());
  }

 private:
  MultibodyPlant<double>& plant_;
  ModelInstanceIndex plant_instance_;
  lcm::DrakeLcm& lcm_;
};

// TODO: consider get rid of the output logic system.
class WheelControllerLogic : public systems::LeafSystem<double> {
 public:
  WheelControllerLogic(MultibodyPlant<double>& plant,
                  ModelInstanceIndex plant_instance)
      : plant_(plant), plant_instance_(plant_instance) {
    this->DeclareVectorInputPort(
        "input1", systems::BasicVector<double>(2));
    this->DeclareVectorOutputPort(
        "output1", systems::BasicVector<double>(plant.num_actuators()),
        &WheelControllerLogic::remap_output);
  }

  void remap_output(const systems::Context<double>& context,
                    systems::BasicVector<double>* output_vector) const {
    auto output_value = output_vector->get_mutable_value();
    auto input_value = this->EvalVectorInput(context, 0)->get_value();
    
    output_value = input_value;
    drake::log()->info("Actual torque:");
    drake::log()->info(output_value.transpose());
  }

 private:
  MultibodyPlant<double>& plant_;
  ModelInstanceIndex plant_instance_;
};

class WheelStateSelector : public systems::LeafSystem<double> {
 public:
  WheelStateSelector(MultibodyPlant<double>& plant,
                  ModelInstanceIndex plant_instance)
      : plant_(plant), plant_instance_(plant_instance) {
    this->DeclareVectorInputPort(
        "input1", systems::BasicVector<double>(plant_.num_multibody_states()));
    this->DeclareVectorOutputPort(
        "output1", systems::BasicVector<double>(4),
        &WheelStateSelector::remap_output);
  }

  void remap_output(const systems::Context<double>& context,
                    systems::BasicVector<double>* output_vector) const {
    auto output_value = output_vector->get_mutable_value();
    auto input_value = this->EvalVectorInput(context, 0)->get_value();

    output_value = Eigen::Vector4d::Zero();
    output_value[0] = input_value[15];
    output_value[1] = input_value[16];
    
    drake::log()->info(input_value.transpose());
  }

 private:

  MultibodyPlant<double>& plant_;
  ModelInstanceIndex plant_instance_;
};

class WheelVelocityController : public systems::Diagram<double> {
 public:
  WheelVelocityController(MultibodyPlant<double>& plant,
                  ModelInstanceIndex plant_instance)
    : plant_(plant), plant_instance_(plant_instance) {
    systems::DiagramBuilder<double> builder;

    // Add wheel state selector.
    const auto* const wss = builder.AddSystem<WheelStateSelector>(plant, plant_instance);

    // Add PID controller.
    const Eigen::VectorXd Kp = Eigen::VectorXd::Ones(2) * 8.0;
    const Eigen::VectorXd Ki = Eigen::VectorXd::Ones(2) * 0.0;
    const Eigen::VectorXd Kd = Eigen::VectorXd::Ones(2) * 0.0;
    const auto* const wc = builder.AddSystem<systems::controllers::PidController<double>>(Kp, Ki, Kd);

    // Add wheel control logic.
    const auto* const wcl = builder.AddSystem<WheelControllerLogic>(plant, plant_instance);

    // Expose Input and Output port.
    builder.ExportInput(wss->get_input_port(0), "wheel_state");
    builder.ExportInput(wc->get_input_port_desired_state(), "desired_wheel_state");
    builder.ExportOutput(wcl->get_output_port(0), "wheel_control");

    // Connect internal ports
    builder.Connect(wss->get_output_port(0), wc->get_input_port_estimated_state());
    builder.Connect(wc->get_output_port_control(), wcl->get_input_port(0));

    builder.BuildInto(this);
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
  lcm::DrakeLcm lcm;
  // auto wc = builder.AddSystem<VelocitySource>(*plant, plant_model_instance_index, lcm);
  auto wvc = builder.AddSystem<WheelVelocityController>(*plant, plant_model_instance_index);
  builder.Connect(plant->get_state_output_port(), wvc->get_input_port(0));
  builder.Connect(wvc->get_output_port(0), plant->get_actuation_input_port());

  // // Set PID desired states.
  // auto desired_base_source =
  //       builder.AddSystem<systems::ConstantVectorSource<double>>(
  //           Eigen::Vector4d(10,10,0,0)); // v_L, v_R, a_L, a_R.
  // builder.Connect(desired_base_source->get_output_port(), wvc->get_input_port(1));

  // // Design the straight trajectory to follow.
  const std::vector<double> kTimes{0.0, 5.0, 10.0};
  std::vector<Eigen::MatrixXd> knots(kTimes.size());
  knots[0] = Eigen::Vector2d(0,0); // x, y;
  knots[1] = Eigen::Vector2d(10,0);
  knots[2] = Eigen::Vector2d(20,0);

  // Design a curvy trajectory to follow.
  // const std::vector<double> kTimes{0.0, 0.8, 2.0, 3.2, 4.0};
  // std::vector<Eigen::MatrixXd> knots(kTimes.size());
  // knots[0] = Eigen::Vector2d(0,   0);
  // knots[1] = Eigen::Vector2d(0.5, 0);
  // knots[2] = Eigen::Vector2d(3,   0);
  // knots[3] = Eigen::Vector2d(5.5, 0);
  // knots[4] = Eigen::Vector2d(6,   0);

  trajectories::PiecewisePolynomial<double> trajectory =
      trajectories::PiecewisePolynomial<double>::Pchip(kTimes, knots);

  // The feedback controller that map trajectory tracking error to velocity input.
  auto vs = builder.AddSystem<VelocitySource>(*plant, plant_model_instance_index, lcm);
  builder.Connect(plant->get_state_output_port(), vs->get_input_port(0));
  builder.Connect(vs->get_output_port(0), wvc->get_input_port(1));

  // Adds a trajectory source for desired state.
  auto traj_src = builder.AddSystem<systems::TrajectorySource<double>>(
      trajectory, 2 /* outputs q + v + a*/);
  traj_src->set_name("trajectory_source");
  builder.Connect(traj_src->get_output_port(),
                  vs->get_input_port(1));



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
