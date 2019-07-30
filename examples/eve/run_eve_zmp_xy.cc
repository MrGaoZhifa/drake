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
#include "drake/systems/framework/leaf_system.h"
#include "drake/examples/eve/eve_common.h"
#include "drake/lcmt_viewer_draw.hpp"
#include "drake/systems/controllers/test/zmp_test_util.h"
#include "drake/solvers/linear_system_solver.h"
#include "drake/solvers/solve.h"

#include <chrono>
#include <thread>

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

DEFINE_double(simulation_time, 4,
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
DEFINE_double(com_kp, 2.0, "Used on feedback of com position to track com acceleration.");
DEFINE_double(com_kd, 0.1, "Used on feedback of com velocity to track com acceleration.");
DEFINE_double(base_kp, 2.0, "Used on feedback of com position to track com acceleration.");
DEFINE_double(base_kd, 0.1, "Used on feedback of com velocity to track com acceleration.");

class JInverse : public systems::LeafSystem<double> {
 public:
  systems::InputPortIndex com_acceleration_port_index;
  systems::InputPortIndex base_trajectory_port_index;
  systems::Context<double>* plant_context_;

  JInverse(MultibodyPlant<double>& plant,
           ModelInstanceIndex plant_instance, lcm::DrakeLcm& lcm)
      : plant_(plant), plant_instance_(plant_instance), lcm_(lcm) {
    // Reference acceleration of CoM from cop2com.
    com_acceleration_port_index = this->DeclareVectorInputPort(
        "COM_Acceleration", systems::BasicVector<double>(9)).get_index();
    // Reference acceleration of base from trajectory.
    base_trajectory_port_index = this->DeclareVectorInputPort(
        "base_trajectory", systems::BasicVector<double>(6)).get_index();
    this->DeclareVectorOutputPort(
        "Generalized_Acceleration", systems::BasicVector<double>(plant_.num_velocities()),
        &JInverse::remap_output);
    DeclarePeriodicPublishEvent(0.05, 0, &JInverse::MyPublishHandler);
  }

  void remap_output(const systems::Context<double>& context,
                    systems::BasicVector<double>* output_vector) const {
    auto output_value = output_vector->get_mutable_value();
    auto com_trajectory_value = this->EvalVectorInput(context, com_acceleration_port_index)->get_value();
    auto com_acceleration_value = com_trajectory_value.tail(3);
//    drake::log()->info(com_acceleration_value.transpose());
    auto base_trajectory_value = this->EvalVectorInput(context, base_trajectory_port_index)->get_value();
//    drake::log()->info(base_trajectory_value.transpose());

    // Compute CoM position and velocity.
    Eigen::Vector3d p_WBcm, v_WBcm;
    plant_.CalcCenterOfMassPosition(*plant_context_, &p_WBcm);
    plant_.CalcCenterOfMassVelocity(*plant_context_, &v_WBcm);
    Eigen::Vector3d a_cm = com_acceleration_value
        + FLAGS_com_kp * (com_trajectory_value.head(3) - p_WBcm)
        + FLAGS_com_kd * (com_trajectory_value.segment<3>(3) - v_WBcm);

    // Calculate Jacobian.
//    const Eigen::MatrixXd Jcm = plant_.CalcCenterOfMassJacobian(context);
    MatrixX<double> Jcm(3, plant_.num_velocities());
    plant_.CalcCenterOfMassJacobian(*plant_context_, &Jcm);
    int pris_x_position_index = plant_.GetJointByName("pris_x").position_start();
    int pris_y_position_index = plant_.GetJointByName("pris_y").position_start();
    auto positions = plant_.GetPositions(*plant_context_, plant_instance_);
    auto velocities = plant_.GetVelocities(*plant_context_, plant_instance_);

    // Feedback on base acceleration to follow base trajectory.
    double base_acc_x = base_trajectory_value(4)
        + FLAGS_base_kp * (base_trajectory_value(0) - positions(pris_x_position_index))
        + FLAGS_base_kd * (base_trajectory_value(2) - velocities(pris_x_position_index));
    double base_acc_y = base_trajectory_value(5)
        + FLAGS_base_kp * (base_trajectory_value(1) - positions(pris_y_position_index))
        + FLAGS_base_kd * (base_trajectory_value(3) - velocities(pris_y_position_index));

    // Calculate acceleration bias term.
    Eigen::Vector3d a_bias;
    plant_.CalcBiasForCenterOfMassJacobianTranslationalVelocity(*plant_context_, &a_bias);

    // Get A and b by get rid of known base acceleration.
    Eigen::VectorXd b = a_cm - a_bias
        - Jcm.col(pris_x_position_index) * base_acc_x
        - Jcm.col(pris_y_position_index) * base_acc_y;
//    removeColumn(Jcm, pris_x_position_index);
    MatrixX<double> Jcm_trimed(3, plant_.num_velocities()-2);
    int index = 0;
    for (int i = 0; i < plant_.num_velocities(); ++i) {
      if (i == pris_x_position_index || i == pris_y_position_index)
        continue;
      Jcm_trimed.col(index++) = Jcm.col(i);
    }

    // TODO(Zhaoyuan): Verify the correctness of the svd.
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Jcm_trimed, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd theta_ddot_without_base = svd.solve(b);


    // Solve using the mathematical programming.
    solvers::MathematicalProgram prog;
    const int X_size = plant_.num_velocities()-2;
    auto X_ = prog.NewContinuousVariables(X_size, "X");
    // Add quadratic cost.
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(X_size, X_size);
    Q.block<6,6>(0,0) = 10 * Eigen::MatrixXd::Identity(6, 6);
    Eigen::VectorXd c = Eigen::VectorXd::Zero(X_size);
    prog.AddQuadraticCost(Q, c, X_);
    prog.AddLinearEqualityConstraint(Jcm_trimed * X_, b);
    // Allow the upper body to keep up straight.
    prog.AddLinearEqualityConstraint(X_(1) + X_(2) + X_(3), 0);
    const solvers::MathematicalProgramResult result = Solve(prog);
    // Check result
    auto X_value = result.GetSolution(X_);
    DRAKE_THROW_UNLESS(result.is_success() == true);
//    DRAKE_THROW_UNLESS((X_value-theta_ddot_without_base).norm() < 1e-12);
    DRAKE_THROW_UNLESS((Jcm_trimed*X_value - b).norm() < 1e-12);
//    drake::log()->info(Jcm_trimed*X_value - b);

    // Insert the xd_dot and yd_dot back to thetad_dot. Skip first for compile error.
    output_value = Eigen::VectorXd::Zero(plant_.num_velocities());
    index = 0;
    for (int i = 0; i < plant_.num_velocities(); ++i) {
      if (i == pris_x_position_index) {
        index++;
        output_value[i] = base_acc_x;
        continue;
      }
      if (i == pris_y_position_index) {
        index++;
        output_value[i] = base_acc_y;
        continue;
      }
//      output_value[i] = theta_ddot_without_base[i-index];
      output_value[i] = X_value[i-index];
    }
    DRAKE_THROW_UNLESS(output_value[pris_x_position_index] == base_acc_x);
    DRAKE_THROW_UNLESS(output_value[pris_y_position_index] == base_acc_y);
//    const int second_half_size = theta_ddot_without_base.size() - pris_x_position_index;
//    output_value.head(pris_x_position_index) = theta_ddot_without_base.head(pris_x_position_index);
//    output_value(pris_x_position_index) = base_trajectory_value(4);
//    output_value.tail(second_half_size) = theta_ddot_without_base.tail(second_half_size);
//    drake::log()->info(output_value.transpose());

    // Verify svd.
    MatrixX<double> Jcm_tmp(3, plant_.num_velocities());
    plant_.CalcCenterOfMassJacobian(*plant_context_, &Jcm_tmp);
    DRAKE_THROW_UNLESS((Jcm_trimed*theta_ddot_without_base - b).norm() < 1e-10);
    DRAKE_THROW_UNLESS((Jcm_tmp*output_value - (a_cm-a_bias)).norm() < 1e-10);
  }

  void removeColumn(Eigen::MatrixXd& matrix, int colToRemove) const {
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if ( colToRemove < static_cast<int>(numCols) )
      matrix.block(0,colToRemove,numRows,numCols-colToRemove) =
          matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
  }

 private:
  MultibodyPlant<double>& plant_;
  ModelInstanceIndex plant_instance_;
  lcm::DrakeLcm& lcm_;

  systems::EventStatus MyPublishHandler(const systems::Context<double>& context) const {
    Plot(context);
    return systems::EventStatus::Succeeded();
  }

  void Plot(const systems::Context<double>& context) const {
    auto cop_trajectory_value = this->EvalVectorInput(context, base_trajectory_port_index)->get_value();
    auto com_trajectory_value = this->EvalVectorInput(context, com_acceleration_port_index)->get_value();

    Eigen::Vector3d p_WBcm;
    plant_.CalcCenterOfMassPosition(*plant_context_, &p_WBcm);

    std::vector<std::string> names;
    std::vector<Eigen::Isometry3d> poses;

    // Visualize reference ZMP position.
    names.push_back("Ref_ZMP_" + std::to_string(context.get_time()));
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    Eigen::Vector3d translation = Eigen::VectorXd::Zero(3);
    translation.head(2) = cop_trajectory_value.head(2);
    pose.translation() = translation;
    poses.push_back(pose);

    // Visualize CoM position.
    names.push_back("COM_" + std::to_string(context.get_time()));
    pose.translation() = p_WBcm;
    poses.push_back(pose);

    PublishFramesToLcm("DRAKE_DRAW_FRAMES", poses, names, &lcm_);


    // Draw arrows.
    std::vector<Eigen::VectorXd> contact_points;
    std::vector<Eigen::VectorXd> contact_forces;

    // Visualize CoM position error.
    Eigen::Vector3d a_cm_expected;
    a_cm_expected(0) = (p_WBcm(0) - cop_trajectory_value(0))
        * plant_.gravity_field().gravity_vector().norm() / p_WBcm(2);
    a_cm_expected(1) = (p_WBcm(1) - cop_trajectory_value(1))
        * plant_.gravity_field().gravity_vector().norm() / p_WBcm(2);
    a_cm_expected(2) = 0;
    contact_points.push_back(p_WBcm);
    contact_forces.push_back(com_trajectory_value.head(3)-p_WBcm);

    // Visualize base position error.
    Eigen::Vector3d e_base = Eigen::Vector3d::Zero();
    int pris_x_position_index = plant_.GetJointByName("pris_x").position_start();
    int pris_y_position_index = plant_.GetJointByName("pris_y").position_start();
    auto positions = plant_.GetPositions(*plant_context_, plant_instance_);
    Eigen::Vector3d p_base(positions[pris_x_position_index], positions[pris_y_position_index], 0);
    e_base.head(2) = cop_trajectory_value.head(2) - p_base.head(2);
    contact_points.push_back(p_base);
    contact_forces.push_back(e_base);

    PublishContactToLcm(contact_points, contact_forces, &lcm_);
  }
};

class COP2COM : public systems::LeafSystem<double> {
 public:
  systems::InputPortIndex cop_trajectory_port_index;
  systems::InputPortIndex mbp_state_port_index;
  systems::Context<double>* plant_context_;

  COP2COM(MultibodyPlant<double>& plant,
          ModelInstanceIndex plant_instance, lcm::DrakeLcm& lcm)
      : plant_(plant), plant_instance_(plant_instance), lcm_(lcm) {
    cop_trajectory_port_index = this->DeclareVectorInputPort(
        "COP_Trajectory", systems::BasicVector<double>(6)).get_index();
    mbp_state_port_index = this->DeclareVectorInputPort(
        "MBP_state", systems::BasicVector<double>(plant_.num_multibody_states())).get_index();
    this->DeclareVectorOutputPort(
        "COM_Acceleration", systems::BasicVector<double>(3),
        &COP2COM::remap_output);
    DeclarePeriodicPublishEvent(0.05, 0, &COP2COM::MyPublishHandler);
  }

  void remap_output(const systems::Context<double>& context,
                    systems::BasicVector<double>* output_vector) const {
    auto output_value = output_vector->get_mutable_value();
    auto cop_trajectory_value = this->EvalVectorInput(context, cop_trajectory_port_index)->get_value();
//    auto mbp_state_value = this->EvalVectorInput(context, mbp_state_port_index)->get_value();

    // Compute the current CoM position.
//    auto q_WCcm = plant_.CalcCenterOfMassPosition(context);
    Eigen::Vector3d p_WBcm;
    plant_.CalcCenterOfMassPosition(*plant_context_, &p_WBcm);
//    drake::log()->info(p_WBcm.transpose());

    output_value(0) = (p_WBcm(0) - cop_trajectory_value(0))
        * plant_.gravity_field().gravity_vector().norm() / p_WBcm(2);
    output_value(1) = (p_WBcm(1) - cop_trajectory_value(1))
        * plant_.gravity_field().gravity_vector().norm() / p_WBcm(2);
    output_value(2) = 0;
  }

 private:
  MultibodyPlant<double>& plant_;
  ModelInstanceIndex plant_instance_;
  lcm::DrakeLcm& lcm_;

  systems::EventStatus MyPublishHandler(const systems::Context<double>& context) const {
    Plot(context);
    return systems::EventStatus::Succeeded();
  }

  void Plot(const systems::Context<double>& context) const {
    auto cop_trajectory_value = this->EvalVectorInput(context, cop_trajectory_port_index)->get_value();

    Eigen::Vector3d p_WBcm;
    plant_.CalcCenterOfMassPosition(*plant_context_, &p_WBcm);

    std::vector<std::string> names;
    std::vector<Eigen::Isometry3d> poses;

    // Visualize reference ZMP position.
    names.push_back("Ref_ZMP_" + std::to_string(context.get_time()));
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = cop_trajectory_value.head(3);
    poses.push_back(pose);


//    names.push_back("Real_ZMP_" + std::to_string(context.get_time()));
//    pose.translation() = Eigen::Vector3d(p_WBcm(0)-  / plant_.gravity_field().gravity_vector().norm() * p_WBcm(2) , , 0);

    // Visualize CoM position.
    names.push_back("COM_" + std::to_string(context.get_time()));
    pose.translation() = p_WBcm;
    poses.push_back(pose);

    PublishFramesToLcm("DRAKE_DRAW_FRAMES", poses, names, &lcm_);
    // Visualize expected CoM acceleration.
    Eigen::Vector3d a_cm_expected;
    a_cm_expected(0) = (p_WBcm(0) - cop_trajectory_value(0))
        * plant_.gravity_field().gravity_vector().norm() / p_WBcm(2);
    a_cm_expected(1) = (p_WBcm(1) - cop_trajectory_value(1))
        * plant_.gravity_field().gravity_vector().norm() / p_WBcm(2);;
    a_cm_expected(2) = 0;

    std::vector<Eigen::VectorXd> contact_points;
    std::vector<Eigen::VectorXd> contact_forces;
    contact_points.push_back(p_WBcm);
    contact_forces.push_back(a_cm_expected);

    PublishContactToLcm(contact_points, contact_forces, &lcm_);
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
  const multibody::RigidBody<double>& adapter_body = plant->AddRigidBody(
      "adapter", plant_model_instance_index,
      multibody::SpatialInertia<double>::MakeFromCentralInertia(
          1e-8, Eigen::Vector3d::Ones() * 1e-8,
          multibody::RotationalInertia<double>(1e-8, 1e-8, 1e-8)));

  const multibody::PrismaticJoint<double>& pris_x =
      plant->AddJoint<multibody::PrismaticJoint>(
          "pris_x", plant->world_body(), nullopt, adapter_body, nullopt,
          Eigen::Vector3d::UnitX());
  const multibody::PrismaticJoint<double>& pris_y =
      plant->AddJoint<multibody::PrismaticJoint>(
          "pris_y", adapter_body, nullopt, eve_root, nullopt,
          Eigen::Vector3d::UnitY());
  const multibody::JointActuator<double>& a_pris_x = plant->AddJointActuator("a_pris_x", pris_x);
  const multibody::JointActuator<double>& a_pris_y = plant->AddJointActuator("a_pris_y", pris_y);
  (void)a_pris_x;
  (void)a_pris_y;


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

//  // Create 1 dimensional PID controller for base moving backward and forward.
//  const int q_desired_dimension = 1;
//  Eigen::MatrixXd state_projection = Eigen::MatrixXd::Zero(2 * q_desired_dimension, plant->num_multibody_states());
//  state_projection(0, pris_x.position_start()) = 1;
//  state_projection(1, pris_x.velocity_start()) = 1;
//
//  const Eigen::VectorXd u_instance = Eigen::VectorXd::Ones(1);
//  Eigen::MatrixXd output_projection = Eigen::MatrixXd::Zero(plant->num_actuators(), 1 * q_desired_dimension);
//  a_pris_x.set_actuation_vector(u_instance, &output_projection);
//
//  const Eigen::VectorXd Kp_base = Eigen::VectorXd::Ones(1) * 10.0;
//  const Eigen::VectorXd Ki_base = Eigen::VectorXd::Ones(1) * 0.0;
//  const Eigen::VectorXd Kd_base = Eigen::VectorXd::Ones(1) * 1.0;
//  auto base_controller = builder.AddSystem<systems::controllers::PidController>(state_projection, output_projection, Kp_base, Ki_base, Kd_base);
//  builder.Connect(plant->get_state_output_port(), base_controller->get_input_port_estimated_state());
//  builder.Connect(base_controller->get_output_port_control(), plant->get_actuation_input_port());

  // Zero to the actuation port
  auto zero_force =
      builder.AddSystem<systems::ConstantVectorSource<double>>(Eigen::VectorXd::Zero(plant->num_actuators()));
  builder.Connect(zero_force->get_output_port(),
                  plant->get_actuation_input_port());

  // Create InverseDynamicsController to convert theta_ddot to Bu using plant.
  const int U = plant->num_actuators();
  // No feed back for IDC for now.
  const Eigen::VectorXd Kp = Eigen::VectorXd::Ones(U) * 0.0;
  const Eigen::VectorXd Ki = Eigen::VectorXd::Ones(U) * 0.0;
  const Eigen::VectorXd Kd = Eigen::VectorXd::Ones(U) * 0.0;
  auto IDC = builder.AddSystem<systems::controllers::InverseDynamicsController<double>>(*plant, Kp, Ki, Kd, true);
  builder.Connect(plant->get_state_output_port(), IDC->get_input_port_estimated_state());
  builder.Connect(IDC->get_output_port_control(), plant->get_applied_generalized_force_input_port());


  // Create a system to transform COM acceleration to Joint acceleration.
  lcm::DrakeLcm lcm;
  auto j_inverse = builder.AddSystem<JInverse>(*plant, plant_model_instance_index, lcm);
  builder.Connect(j_inverse->get_output_port(0), IDC->get_input_port_desired_acceleration());

  // Create a COP to COM transform

//  auto cop2com = builder.AddSystem<COP2COM>(*plant, plant_model_instance_index, lcm);
//  builder.Connect(cop2com->get_output_port(0), j_inverse->get_input_port(j_inverse->com_acceleration_port_index));
//  builder.Connect(plant->get_state_output_port(), cop2com->get_input_port(cop2com->mbp_state_port_index));

//  // Design the trajectory to follow.
//  const std::vector<double> kTimes{0.0, 0.8, 2.0, 3.2, 4.0};
//  std::vector<Eigen::MatrixXd> knots(kTimes.size());
//  knots[0] = Eigen::Vector2d(0,0);
//  knots[1] = Eigen::Vector2d(0.5,0);
//  knots[2] = Eigen::Vector2d(3,0.5);
//  knots[3] = Eigen::Vector2d(5.5,0);
//  knots[4] = Eigen::Vector2d(6,0);

  // Design the trajectory to follow.
  const std::vector<double> kTimes{0.0, 1.0, 2.0, 3.0, 4.0};
  std::vector<Eigen::MatrixXd> knots(kTimes.size());
  knots[0] = Eigen::Vector2d(0,0);
  knots[1] = Eigen::Vector2d(0,0);
  knots[2] = Eigen::Vector2d(0,0);
  knots[3] = Eigen::Vector2d(0,0);
  knots[4] = Eigen::Vector2d(0,0);

//  const std::vector<double> kTimes{0.0, 1.0, 2.0, 3.0};
//  std::vector<Eigen::MatrixXd> knots(kTimes.size());
//  knots[0] = Eigen::Vector2d(0,0);
//  knots[1] = Eigen::Vector2d(0.5,-0.0);
//  knots[2] = Eigen::Vector2d(2.5,0.0);
//  knots[3] = Eigen::Vector2d(3,0);

//  trajectories::PiecewisePolynomial<double> trajectory =
//      trajectories::PiecewisePolynomial<double>::FirstOrderHold(kTimes, knots);
  trajectories::PiecewisePolynomial<double> trajectory =
      trajectories::PiecewisePolynomial<double>::Pchip(kTimes, knots);

//  // Adds a trajectory source for desired state.
//  auto traj_src = builder.AddSystem<systems::TrajectorySource<double>>(
//      trajectory, 1 /* outputs q + v */);
//  traj_src->set_name("trajectory_source");
//  builder.Connect(traj_src->get_output_port(),
//                  cop2com->get_input_port(cop2com->cop_trajectory_port_index));

  // Adds a trajectory source for desired base acceleration.
  auto traj_acc_src = builder.AddSystem<systems::TrajectorySource<double>>(
      trajectory, 2 /* outputs q + v + a*/);
  traj_acc_src->set_name("trajectory_acceleration_source");
  builder.Connect(traj_acc_src->get_output_port(),
                  j_inverse->get_input_port(j_inverse->base_trajectory_port_index));


  // TODO: Create a trajectory desired state for IDC.
  // For now, just use home configuration as desired state for IDC, the PID parameter is all zero.
  auto desired_theta =
      builder.AddSystem<systems::ConstantVectorSource<double>>(Eigen::VectorXd::Zero(plant->num_multibody_states()));
  builder.Connect(desired_theta->get_output_port(), IDC->get_input_port_desired_state());

  // TODO: Create the prioritized controller that optimize the desired
  //  acceleration in order to keep the qd_ddot while make the robot state
  //  close to home.
//  // Create a trajectory desired state for PID controller.
//  // This source is the 1 dimension base state.
//  builder.Connect(traj_src->get_output_port(), base_controller->get_input_port_desired_state());

  // TODO: Compute the com height using multibody plant function.
  // Given the trajectory of reference ZMP, we get the CoM trajectory.
  const double z_cm = 0.592203;
  Eigen::Vector4d x0(0, 0, 0, 0);
  systems::controllers::ZMPPlanner zmp_planner;
  zmp_planner.Plan(trajectory, x0, z_cm);
  double sample_dt = 0.01;
  systems::controllers::ZMPTestTraj result =
      systems::controllers::SimulateZMPPolicy(zmp_planner, x0, sample_dt, 0.02);

  const int N = result.time.size();
  std::vector<double> com_times;
  std::vector<Eigen::MatrixXd> com_knots;
  for (int i = 0; i < N; i++) {
    com_times.push_back(result.time[i]);
    Eigen::Vector3d pi_com;
    pi_com.head(2) = result.nominal_com.col(i).head(2);
    pi_com(2) = z_cm;
    com_knots.push_back(pi_com);
  }
  trajectories::PiecewisePolynomial<double> com_trajectory =
      trajectories::PiecewisePolynomial<double>::Pchip(com_times, com_knots);
  // Adds a trajectory source for desired base acceleration.
  auto com_traj_acc_src = builder.AddSystem<systems::TrajectorySource<double>>(
      com_trajectory, 2 /* outputs q + v + a*/);
  com_traj_acc_src->set_name("com_trajectory_acceleration_source");
  builder.Connect(com_traj_acc_src->get_output_port(),
                  j_inverse->get_input_port(j_inverse->com_acceleration_port_index));

  // Visualize the CoM and ZMP trajectory.
  std::vector<std::string> names;
  std::vector<Eigen::Isometry3d> poses;
  for (int t = 0; t < N; t=t+5) {
    names.push_back("CoM" + std::to_string(int(t)));
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = Eigen::Vector3d(result.nominal_com(0, t), result.nominal_com(1, t), z_cm);
    poses.push_back(pose);
  }
  for (int t = 0; t < N; t=t+5) {
    names.push_back("ZMP" + std::to_string(int(t)));
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = Eigen::Vector3d(result.desired_zmp(0, t), result.desired_zmp(1, t), 0);
    poses.push_back(pose);
  }
  PublishFramesToLcm("DRAKE_DRAW_FRAMES2", poses, names, &lcm);
  std::this_thread::sleep_for(std::chrono::seconds(2));

  // Connect plant with scene_graph to get collision information.
  DRAKE_DEMAND(!!plant->get_source_id());
  builder.Connect(
      plant->get_geometry_poses_output_port(),
      scene_graph.get_source_pose_port(plant->get_source_id().value()));
  builder.Connect(scene_graph.get_query_output_port(),
                  plant->get_geometry_query_input_port());

  geometry::ConnectDrakeVisualizer(&builder, scene_graph);

//  // Create a tmp context for this diagram and plant.
//  std::unique_ptr<systems::Diagram<double>> diagram_tmp = builder.Build();
//  std::unique_ptr<systems::Context<double>> diagram_context_tmp =
//      diagram_tmp->CreateDefaultContext();
//
//  systems::Context<double>& plant_context_tmp = diagram_tmp->GetMutableSubsystemContext(*plant, diagram_context_tmp.get());
////  cop2com->plant_context_ = &plant_context_real;
//
//  // Set the robot COM position, make sure the robot base is off the ground.
//  Eigen::VectorXd positions_tmp = Eigen::VectorXd::Zero(plant->num_positions());
//  positions_tmp[plant->GetJointByName("j_hip_y").position_start()] = 0.43;
//  positions_tmp[plant->GetJointByName("j_knee_y").position_start()] = -0.91;
//  positions_tmp[plant->GetJointByName("j_ankle_y").position_start()] = 0.47;
//  plant->SetPositions(&plant_context_tmp, positions_tmp);
//
//  // Set robot init velocity for every joint.
//  drake::VectorX<double> velocities_tmp = Eigen::VectorXd::Zero(plant->num_velocities());
//  plant->SetVelocities(&plant_context_tmp, velocities_tmp);

//  // Compute the CoM position and extract height.
//  Eigen::Vector3d p_cm;
//  plant->CalcCenterOfMassPosition(plant_context_tmp, &p_cm);
//  const double z_cm = p_cm(2);
//  drake::log()->info(p_cm.transpose());


  // Create a real context for this diagram and plant.
  std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();

  systems::Context<double>& plant_context_real = diagram->GetMutableSubsystemContext(*plant, diagram_context.get());
  j_inverse->plant_context_ = &plant_context_real;

  // Set the robot COM position, make sure the robot base is off the ground.
  Eigen::VectorXd positions = Eigen::VectorXd::Zero(plant->num_positions());
  positions[plant->GetJointByName("j_hip_y").position_start()] = 0.43;
  positions[plant->GetJointByName("j_knee_y").position_start()] = -0.91;
  positions[plant->GetJointByName("j_ankle_y").position_start()] = 0.47;
  plant->SetPositions(&plant_context_real, positions);

  // Set robot init velocity for every joint.
  drake::VectorX<double> velocities = Eigen::VectorXd::Zero(plant->num_velocities());
//  velocities[plant->GetJointByName("pris_y").velocity_start()] = -1.25;
  plant->SetVelocities(&plant_context_real, velocities);

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
