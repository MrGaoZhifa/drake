/// @file
///
/// This file create a eve plant for simulation and another one for control.
/// The simulation plant if connected to ground with a prismatic and a rovolute
/// joint, the control plant is welded to ground.

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
#include "drake/systems/controllers/linear_quadratic_regulator.h"
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

class JInverse : public systems::LeafSystem<double> {
 public:
  JInverse(MultibodyPlant<double>* plant) : plant_(plant) {
    this->DeclareVectorInputPort(
        "COM_Acceleration", systems::BasicVector<double>(3));
    this->DeclareVectorOutputPort(
        "Generalized_Acceleration", systems::BasicVector<double>(plant_->num_velocities()),
        &JInverse::remap_output);
  }

  void remap_output(const systems::Context<double>& context,
                    systems::BasicVector<double>* output_vector) const {
    auto output_value = output_vector->get_mutable_value();
    auto input_value = this->EvalVectorInput(context, 0)->get_value();
    drake::log()->info(output_value.transpose());

//    output_value = plant_.CalcJacobianAngularVelocity().inverse() * input_value;
    output_value = Eigen::VectorXd::Zero(plant_->num_velocities());
  }

 private:
  MultibodyPlant<double>* plant_;
};

class COP2COM : public systems::LeafSystem<double> {
 public:
  COP2COM() {
    this->DeclareVectorInputPort(
        "COM_Acceleration", systems::BasicVector<double>(3));
    this->DeclareVectorOutputPort(
        "Generalized_Acceleration", systems::BasicVector<double>(3)),
        &COP2COM::remap_output);
  }

  void remap_output(const systems::Context<double>& context,
                    systems::BasicVector<double>* output_vector) const {
    auto output_value = output_vector->get_mutable_value();
    auto input_value = this->EvalVectorInput(context, 0)->get_value();
    drake::log()->info(output_value.transpose());

//    output_value = plant_.CalcJacobianAngularVelocity().inverse() * input_value;
    output_value = input_value;
  }
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
      "sdf/eve_7dof_arms_relative_base_no_collision.sdf");
//        "sdf/eve_2dof_base_no_collision.sdf");
  //      "urdf/eve_7dof_arms_relative_base_no_collision.urdf");

  ModelInstanceIndex plant_model_instance_index =
      parser.AddModelFromFile(full_name);
  (void)plant_model_instance_index;

  // Add gravity.
  const Vector3<double> gravity_vector_W(0, 0, -FLAGS_gravity);
  plant->mutable_gravity_field().set_gravity_vector(gravity_vector_W);

  // Connect the plant to the world frame with 1 prismatic joint.
  const multibody::Body<double>& eve_root = plant->GetBodyByName("base");

  const multibody::PrismaticJoint<double>& pris_x =
      plant->AddJoint<multibody::PrismaticJoint>(
          "pris_x", plant->world_body(), nullopt, eve_root, nullopt,
          Eigen::Vector3d::UnitX());

  plant->AddJointActuator("a_pris_x", pris_x);

  // Now the plant is complete.
  plant->Finalize();

  int index = 0;
  for (multibody::JointIndex j(0); j < plant->num_joints(); ++j) {
    drake::log()->info(std::to_string(index++));
    drake::log()->info(
        "PLANT JOINT: " + plant->get_joint(j).name() + ", position@ " +
            std::to_string(plant->get_joint(j).position_start()) + ", velocity@ " +
            std::to_string(plant->get_joint(j).velocity_start()));
  }

//  // Create InverseDynamicsController using fake_plant.
////  const int Q = plant->num_positions();
////  const int V = plant->num_velocities();
//  const int U = plant->num_actuators();
//  const Eigen::VectorXd Kp_ = Eigen::VectorXd::Ones(U) * 0.0;
//  const Eigen::VectorXd Ki_ = Eigen::VectorXd::Ones(U) * 0.0;
//  const Eigen::VectorXd Kd_ = Eigen::VectorXd::Ones(U) * 0.0;
//  auto feed_forward_controller =
//      builder
//          .AddSystem<systems::controllers::InverseDynamicsController<double>>(
//              *plant, Kp_, Ki_, Kd_, false);
//  builder.Connect(plant->get_state_output_port(), feed_forward_controller->get_input_port_estimated_state());
//  builder.Connect(feed_forward_controller->get_output_port_control(), plant->get_applied_generalized_force_input_port());
//
//  // Set desired position [q,v]' for IDC as feedback reference.
//  VectorX<double> constant_pos_value =
//      VectorX<double>::Ones(plant->num_multibody_states()) * FLAGS_constant_pos;
//  auto desired_constant_source =
//      builder.AddSystem<systems::ConstantVectorSource<double>>(
//          constant_pos_value);
//  desired_constant_source->set_name("desired_constant_source");
//  builder.Connect(desired_constant_source->get_output_port(),
//                  feed_forward_controller->get_input_port_desired_state());


  // Zero to the actuation port
  auto zero_force =
      builder.AddSystem<systems::ConstantVectorSource<double>>(Eigen::VectorXd::Zero(plant->num_actuators()));
  builder.Connect(zero_force->get_output_port(),
                  plant->get_actuation_input_port());

  // Create IDC to convert theta_ddot to Bu.
  const int U = plant->num_actuators();
  const Eigen::VectorXd Kp = Eigen::VectorXd::Ones(U) * 5.0;
  const Eigen::VectorXd Ki = Eigen::VectorXd::Ones(U) * 0.0;
  const Eigen::VectorXd Kd = Eigen::VectorXd::Ones(U) * 0.0;
  auto IDC = builder.AddSystem<systems::controllers::InverseDynamicsController<double>>(plant, Kp, Ki, Kd, true);
  builder.Connect(plant->get_state_output_port(), IDC->get_input_port_estimated_state());
  builder.Connect(IDC->get_output_port_control(), plant->get_applied_generalized_force_input_port());

  // Create a system to transform COM acceleration to Joint acceleration.
  auto j_inverse = builder.AddSystem<JInverse>(plant);
  builder.Connect(j_inverse->get_output_port(0), IDC->get_input_port_desired_acceleration());

  // TODO: Create a trajectory desired state for IDC.
  // Transform trajectory of base to whole body state and upper body state to zero.

  // TODO: Create a COP to COM transform
  auto cop2cpm = builder.AddSystem<COP2COM>();
  builder.Connect(cop2cpm->get_output_port(0), j_inverse->get_input_port(0));

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
  trajectories::PiecewisePolynomial<double> trajectory =
      trajectories::PiecewisePolynomial<double>::FirstOrderHold(
          kTimes, knots);
  // Adds a trajectory source for desired state.
  auto traj_src = builder.AddSystem<systems::TrajectorySource<double>>(
      trajectory, 1 /* outputs q + v */);
  traj_src->set_name("trajectory_source");

  builder.Connect(traj_src->get_output_port(),
                  cop2cpm->get_input_port(0));



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

//  // Set the robot COM position, make sure the robot base is off the ground.
//  Eigen::VectorXd positions = Eigen::VectorXd::Zero(plant->num_positions());
//  positions[0] = 30;
//  plant->SetPositions(plant_context.get(), positions);

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
      "A simple dynamic simulation for the Allegro hand moving under constant"
      " torques.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::eve::DoMain();
  return 0;
}
