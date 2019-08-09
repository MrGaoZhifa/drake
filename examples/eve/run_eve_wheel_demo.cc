/// @file
///
/// This demo sets up a humanoid robot eve from halodi robotics. The file shows
/// how to use inverse dynamics controller and pid controller to balance the
/// robot.
/// Currently the pid controller does not able to solve the deviation problem.

#include <gflags/gflags.h>

#include <drake/multibody/plant/externally_applied_spatial_force.h>
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
#include "drake/multibody/plant/contact_results_to_lcm.h"
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
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/examples/eve/eve_common.h"
#include "drake/lcmt_viewer_draw.hpp"
#include "drake/systems/controllers/test/zmp_test_util.h"
#include "drake/solvers/linear_system_solver.h"
#include "drake/solvers/solve.h"
#include "drake/common/proto/call_python.h"

#include <memory>
#include <algorithm>

namespace drake {
namespace examples {
namespace eve {
using drake::multibody::BodyIndex;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using common::CallPython;

DEFINE_double(constant_pos, 0.0,
              "the constant load on each joint, Unit [Nm]."
              "Suggested load is in the order of 0.01 Nm. When input value"
              "equals to 0 (default), the program runs a passive simulation.");

DEFINE_double(simulation_time, 10,
              "Desired duration of the simulation in seconds");

DEFINE_double(max_time_step, 2.0e-3,
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
DEFINE_bool(is_inclined_plane_half_space, true,
            "Is inclined plane a half-space (true) or box (false).");
DEFINE_double(init_height, 0.13, "Initial height for base.");

DEFINE_double(K1, 30, "The feedback control for base error of x and y.");
DEFINE_double(K2, 30, "The feedback control for base rotational velocity.");
DEFINE_double(K3, 30, "The feedback control for base rotational velocity");
DEFINE_double(com_kp, 30, "Used on feedback of com position to track com acceleration.");
DEFINE_double(com_kd, 10, "Used on feedback of com velocity to track com acceleration.");
DEFINE_double(base_kp, 2.0, "Used on feedback of base position to track base acceleration.");
DEFINE_double(base_kd, 0.1, "Used on feedback of base velocity to track base acceleration.");
DEFINE_double(circle_radius, 1.0, "Used on feedback of base velocity to track base acceleration.");
DEFINE_double(inteval, 0.4, "Used on feedback of base velocity to track base acceleration.");
DEFINE_double(precision, 16, "Used on feedback of base velocity to track base acceleration.");

class JInverse : public systems::LeafSystem<double> {
 public:
  systems::InputPortIndex com_acceleration_port_index;
  systems::InputPortIndex base_trajectory_port_index;
  systems::Context<double>* plant_context_;

  JInverse(MultibodyPlant<double>& plant,
           MultibodyPlant<double>& fake_plant,
           ModelInstanceIndex plant_instance,
           ModelInstanceIndex fake_plant_instance,
           lcm::DrakeLcm& lcm)
      : plant_(plant), fake_plant_(fake_plant),
        plant_instance_(plant_instance), fake_plant_instance_(fake_plant_instance),
        lcm_(lcm) {
    // Reference acceleration of CoM from cop2com.
    com_acceleration_port_index = this->DeclareVectorInputPort(
        "COM_Acceleration", systems::BasicVector<double>(9)).get_index();
    // Reference acceleration of base from trajectory.
    base_trajectory_port_index = this->DeclareVectorInputPort(
        "base_trajectory", systems::BasicVector<double>(6)).get_index();
    this->DeclareVectorOutputPort(
        "Generalized_Acceleration", systems::BasicVector<double>(fake_plant_.num_velocities()),
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

    // The is the first try.
    // Here we assume the base is able to track the zmp trajectory perfectly.
    // So the base acceleration is the same feedforward acceleration on the zmp trajectory.
    // The base acceleration does not care about the direction of the base, it only care about the acceleration of x y direction in world.
    // We could get rid of the base x y acceleration and then use the fixed base fake model to compute the jacobian.

    const systems::DiagramContext<double>* diagram_context = dynamic_cast<const systems::DiagramContext<double>*>(context.get_parent_base());
    const systems::Context<double>& plant_context = diagram_context->GetSubsystemContext(systems::SubsystemIndex{1});

    // Compute CoM position and velocity.
    Eigen::Vector3d p_WBcm, v_WBcm;
    plant_.CalcCenterOfMassPosition(plant_context, &p_WBcm);
    plant_.CalcCenterOfMassVelocity(plant_context, &v_WBcm);
    Eigen::Vector3d a_cm = com_acceleration_value
        + FLAGS_com_kp * (com_trajectory_value.head(3) - p_WBcm)
        + FLAGS_com_kd * (com_trajectory_value.segment<3>(3) - v_WBcm);

    // Calculate Jacobian.
//    const Eigen::MatrixXd Jcm = plant_.CalcCenterOfMassJacobian(context);
    MatrixX<double> Jcm(3, plant_.num_velocities());
    plant_.CalcCenterOfMassJacobian(*plant_context_, &Jcm);


    // Compute base acceleration to define.
//    int pris_x_position_index = plant_.GetJointByName("pris_x").position_start();
//    int pris_y_position_index = plant_.GetJointByName("pris_y").position_start();
    int pris_x_position_index = 3;
    int pris_y_position_index = 4;
    auto positions = plant_.GetPositions(*plant_context_, plant_instance_);
    auto velocities = plant_.GetVelocities(*plant_context_, plant_instance_);

    // Feedback on base acceleration to follow base trajectory.
    double base_acc_x = base_trajectory_value(4);
//        + FLAGS_base_kp * (base_trajectory_value(0) - positions(pris_x_position_index))
//        + FLAGS_base_kd * (base_trajectory_value(2) - velocities(pris_x_position_index));
    double base_acc_y = base_trajectory_value(5);
//        + FLAGS_base_kp * (base_trajectory_value(1) - positions(pris_y_position_index))
//        + FLAGS_base_kd * (base_trajectory_value(3) - velocities(pris_y_position_index));

    // Calculate acceleration bias term.
    Eigen::Vector3d a_bias;
    plant_.CalcBiasForCenterOfMassJacobianTranslationalVelocity(*plant_context_, &a_bias);

    // Get A and b by get rid of known base acceleration.
    Eigen::VectorXd b = a_cm - a_bias
        - Jcm.col(pris_x_position_index) * base_acc_x
        - Jcm.col(pris_y_position_index) * base_acc_y;

    // Remove the first 6 column of velocity of floating base, and remove 2 velocities for wheels.
    const int num_floating_velocities = 6;
    const int num_wheel_velocities = 2;
    MatrixX<double> Jcm_trimed =
        Jcm.block(0, num_floating_velocities + num_wheel_velocities,
            3, fake_plant_.num_velocities());

//    // TODO(Zhaoyuan): Verify the correctness of the svd.
//    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Jcm_trimed, Eigen::ComputeThinU | Eigen::ComputeThinV);
//    Eigen::VectorXd theta_ddot_without_base = svd.solve(b);
////////////////////////////////////////////////////////////////////////////////
    // Form the QP problem to solve the weigghted theta double dot.
    // Solve using the mathematical programming.
    solvers::MathematicalProgram prog;
    const int X_size = Jcm_trimed.cols();
    auto X_ = prog.NewContinuousVariables(X_size, "X");
    // Add quadratic cost.
    Eigen::VectorXd Xd = Eigen::VectorXd::Zero(X_size);
    drake::log()->info(positions.segment<3>(10));
    Xd.segment<3>(1) = 0 * (Eigen::Vector3d{0.43, -0.91, 0.47} - positions.segment<3>(10));
        + 0 * (Eigen::Vector3d{0, 0, 0} - velocities.segment<3>(39));
    Eigen::MatrixXd Q = 1 * Eigen::MatrixXd::Identity(X_size, X_size);
    Q.block<6,6>(0,0) = 100 * Eigen::MatrixXd::Identity(6, 6); // Joints on the leg.
    Q(0,0) = 10; Q(1,1) = 10; // ankle x and y.
    Q(5,5) = 1000; // pelvis Z.
    Q.block<10,10>(11,11) = 1000 * Eigen::MatrixXd::Identity(6, 6); // Joints on arms except shoulder.
    Q(7,7) = 1000; // Head not move.
    Eigen::VectorXd c = -2 * Q * Xd; // Penalize the (X_ - Xd)' * Q * (X_ - Xd)
    prog.AddQuadraticCost(Q, c, X_);
    prog.AddLinearEqualityConstraint(Jcm_trimed * X_, b);
    // Allow the upper body to keep up straight.
    prog.AddLinearEqualityConstraint(X_(1) + X_(2) + X_(3), 0);
    const solvers::MathematicalProgramResult result = Solve(prog);
    // Check result
    auto X_value = result.GetSolution(X_);
    DRAKE_THROW_UNLESS(result.is_success() == true);
//    DRAKE_THROW_UNLESS((X_value-theta_ddot_without_base).norm() < 1e-12);
    DRAKE_THROW_UNLESS((Jcm_trimed*X_value - b).norm() < 1e-10);
//    drake::log()->info(Jcm_trimed*X_value - b);

    // TODO: Make sure the order for the two accleration is correct,
    //  aka: verify last N element of V from floating base robot is the same order as the fake robot.
    DRAKE_DEMAND(output_value.size() == X_size);
    drake::log()->info(X_value.head(6).transpose());
    output_value = X_value;

////////////////////////////////////////////////////////////////////////////////
    // Verify svd.
//    MatrixX<double> Jcm_tmp(3, plant_.num_velocities());
//    plant_.CalcCenterOfMassJacobian(*plant_context_, &Jcm_tmp);
//    DRAKE_THROW_UNLESS((Jcm_trimed*theta_ddot_without_base - b).norm() < 1e-10);
    DRAKE_THROW_UNLESS((Jcm_trimed*X_value - b).norm() < 1e-10);
//    DRAKE_THROW_UNLESS((Jcm_tmp*output_value - (a_cm-a_bias)).norm() < 1e-10);

    // Draw arrows.
    std::vector<Eigen::VectorXd> contact_points;
    std::vector<Eigen::VectorXd> contact_forces;

    // Visualize CoM position error.
    contact_points.emplace_back(p_WBcm);
//    contact_forces.emplace_back(com_trajectory_value.head(3)-p_WBcm);
    contact_forces.emplace_back(a_cm);

    PublishContactToLcm(contact_points, contact_forces, &lcm_);
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
  MultibodyPlant<double>& fake_plant_;
  ModelInstanceIndex plant_instance_;
  ModelInstanceIndex fake_plant_instance_;
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

    PublishFramesToLcm("DRAKE_DRAW_FRAMES_ONLINE", poses, names, &lcm_);


    // Draw arrows.
    std::vector<Eigen::VectorXd> contact_points;
    std::vector<Eigen::VectorXd> contact_forces;

    // Visualize CoM position error.
//    Eigen::Vector3d a_cm_expected;
//    a_cm_expected(0) = (p_WBcm(0) - cop_trajectory_value(0))
//        * plant_.gravity_field().gravity_vector().norm() / p_WBcm(2);
//    a_cm_expected(1) = (p_WBcm(1) - cop_trajectory_value(1))
//        * plant_.gravity_field().gravity_vector().norm() / p_WBcm(2);
//    a_cm_expected(2) = 0;
    contact_points.emplace_back(p_WBcm);
    contact_forces.emplace_back(com_trajectory_value.head(3)-p_WBcm);

    // Visualize base position error.
//    Eigen::Vector3d e_base = Eigen::Vector3d::Zero();
//    int pris_x_position_index = plant_.GetJointByName("pris_x").position_start();
//    int pris_y_position_index = plant_.GetJointByName("pris_y").position_start();
//    auto positions = plant_.GetPositions(*plant_context_, plant_instance_);
//    Eigen::Vector3d p_base(positions[pris_x_position_index], positions[pris_y_position_index], 0);
//    e_base.head(2) = cop_trajectory_value.head(2) - p_base.head(2);
//    contact_points.push_back(p_base);
//    contact_forces.push_back(e_base);

    PublishContactToLcm(contact_points, contact_forces, &lcm_);
  }
};

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
//    drake::log()->info(desired_traj_value.transpose());

    math::RollPitchYawd rpy_base(Eigen::Quaterniond(state_value[0], state_value[1], state_value[2], state_value[3]));
//    math::RollPitchYawd rpy_desired_base(Eigen::Quaterniond(state_value[0], state_value[1], state_value[2], state_value[3]));

    Eigen::Vector3d state{rpy_base.yaw_angle(), state_value[4], state_value[5]};
    Eigen::Vector3d desired_state{
        std::atan2(desired_traj_value[3], desired_traj_value[2]),
        desired_traj_value[0], desired_traj_value[1]};


    // Visualize frame attached to base.
    std::vector<std::string> names;
    std::vector<Eigen::Isometry3d> poses;
    names.push_back("base_state");

    Eigen::Isometry3d pose = Eigen::Translation3d(Eigen::Vector3d(state_value[4], state_value[5], state_value[6])) *
        Eigen::AngleAxisd(rpy_base.yaw_angle(), Eigen::Vector3d::UnitZ());
    poses.push_back(pose);

    // Draw frame of where the desired base is.
    names.push_back("desired_base_position");
    Eigen::Isometry3d desired_pose = Eigen::Translation3d(Eigen::Vector3d(desired_traj_value[0], desired_traj_value[1], 0.13))
        * Eigen::AngleAxisd(desired_state[0], Eigen::Vector3d::UnitZ());
    poses.push_back(desired_pose);

    PublishFramesToLcm("DRAKE_DRAW_FRAMES", poses, names, &lcm_);

    // Draw arrows.
    std::vector<Eigen::VectorXd> contact_points;
    std::vector<Eigen::VectorXd> contact_forces;

    // Visualize base position error.
    Eigen::Vector3d e_base = Eigen::Vector3d::Zero();
    Eigen::Vector3d p_base(state_value[4], state_value[5], state_value[6]);
    e_base.head(2) = desired_traj_value.head(2) - p_base.head(2);
    contact_points.push_back(p_base);
    contact_forces.push_back(e_base);

//    PublishContactToLcm(contact_points, contact_forces, &lcm_);


    // Error dynamics controller design.
    Eigen::Matrix3d kinematic_constraint_matrix;
    kinematic_constraint_matrix << 1, 0, 0,
        0, std::cos(desired_state[0]), std::sin(desired_state[0]),
        0, -std::sin(desired_state[0]), std::cos(desired_state[0]);

    Eigen::Vector3d state_error = kinematic_constraint_matrix * (state - desired_state);
    state_error[0] = std::max(-M_PI*0.4, std::min(M_PI*0.4, state_error[0]));
//    state_error = Eigen::Vector3d::Zero();
    drake::log()->info(state_error.transpose());

    Eigen::Vector2d feedforward_velocity{desired_traj_value.segment<2>(2).norm(),
                                         (desired_traj_value.tail(2) -
                                             desired_traj_value.tail(2).dot(desired_traj_value.segment<2>(2).normalized()) * desired_traj_value.segment<2>(2).normalized()).norm()};

    // Modern Robotics P468 Eq.13.31
    Eigen::Vector2d actual_velocity_input = Eigen::Vector2d{
        (feedforward_velocity[0] - FLAGS_K1 * feedforward_velocity[0] * (state_error[1] + state_error[2] * std::tanh(state_error[0]))) / std::cos(state_error[0]),
        feedforward_velocity[1] - (FLAGS_K2 * feedforward_velocity[0] * state_error[2] + FLAGS_K3 * feedforward_velocity[0] * std::tanh(state_error[0])) * std::pow(std::cos(state_error[0]),2)
    };

    // Eigen::Vector2d actual_velocity_input = feedforward_velocity;
    const double l = 0.26983;
    const double r = 0.15;
    Eigen::Vector2d actual_wheel_velocity{actual_velocity_input[0] - actual_velocity_input[1] * l, // left wheel velocity
                                          actual_velocity_input[0] + actual_velocity_input[1] * l};// right wheel velocity

    output_value.setZero();
    output_value.head(2) = actual_wheel_velocity / r;

    drake::log()->info("feedback base velocity:");
    drake::log()->info(actual_velocity_input.transpose());
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
        "output1", systems::BasicVector<double>(plant_.num_actuators()),
        &WheelControllerLogic::remap_output);
  }

  void remap_output(const systems::Context<double>& context,
                    systems::BasicVector<double>* output_vector) const {
    auto output_value = output_vector->get_mutable_value();
    auto input_value = this->EvalVectorInput(context, 0)->get_value();

    output_value.setZero();
    output_value.head(2) = input_value.cwiseMax(Eigen::Vector2d::Ones()*-20).cwiseMin(Eigen::Vector2d::Ones()*20);
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
    output_value[0] = input_value[plant_.num_positions() + plant_.GetJointByName("j_l_wheel_y").velocity_start()];
    output_value[1] = input_value[plant_.num_positions() + plant_.GetJointByName("j_r_wheel_y").velocity_start()];

    drake::log()->info("\nActual velocity and acceleration");
    drake::log()->info(output_value.transpose());
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
    const Eigen::VectorXd Kp = Eigen::VectorXd::Ones(2) * 5.0;
    const Eigen::VectorXd Ki = Eigen::VectorXd::Ones(2) * 0.0;
    const Eigen::VectorXd Kd = Eigen::VectorXd::Ones(2) * 2.0;
    const auto* const wc = builder.AddSystem<systems::controllers::PidController<double>>(Kp, Ki, Kd);

    // Add wheel control logic.
    const auto* const wcl = builder.AddSystem<WheelControllerLogic>(plant_, plant_instance);

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
  logging::HandleSpdlogGflags();
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
      //      "urdf/eve_7dof_arms_relative_base_collision.urdf");
      "sdf/eve_7dof_arms_relative_base_sphere_collision.sdf");

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

  // Create fake model for InverseDynamicsController
  MultibodyPlant<double> fake_plant(FLAGS_max_time_step);
  fake_plant.set_name("fake_plant");
  multibody::Parser fake_parser(&fake_plant);

  const std::string fake_full_name = FindResourceOrThrow(
      "drake/manipulation/models/eve/"
      //      "urdf/eve_7dof_arms_relative_base_collision.urdf");
      "sdf/eve_7dof_arms_relative_base_no_collision.sdf");

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
      "num_joints: " + std::to_string(plant->num_joints()) +
      ", num_positions: " + std::to_string(plant->num_positions()) +
      ", num_velocities: " + std::to_string(plant->num_velocities()) +
      ", num_actuators: " + std::to_string(plant->num_actuators()));
  drake::log()->info(
      "num_joints: " + std::to_string(fake_plant.num_joints()) +
      ", num_positions: " + std::to_string(fake_plant.num_positions()) +
      ", num_velocities: " + std::to_string(fake_plant.num_velocities()) +
      ", num_actuators: " + std::to_string(fake_plant.num_actuators()));
  int index = 0;
  for (multibody::JointActuatorIndex a(0); a < plant->num_actuators(); ++a) {
    drake::log()->info(std::to_string(index++));
    drake::log()->info(
        "PLANT JOINT: " + plant->get_joint_actuator(a).joint().name() +
        " has actuator " + plant->get_joint_actuator(a).name());

    //    Eigen::VectorXd u_instance(1);
    //    u_instance << 100;
    //    Eigen::VectorXd u = Eigen::VectorXd::Zero(plant->num_actuators());
    //    plant->get_joint_actuator(a).set_actuation_vector(u_instance, &u);
    //    drake::log()->info(u.transpose());
    if (index < fake_plant.num_joints())
      drake::log()->info("FAKE PLANT JOINT: " +
                         fake_plant.get_joint_actuator(a).joint().name() +
                         " has actuator " +
                         fake_plant.get_joint_actuator(a).name());
  }
  index = 0;
  for (multibody::JointIndex j(0); j < plant->num_joints(); ++j) {
    drake::log()->info(std::to_string(index++));

    drake::log()->info(
        "PLANT JOINT: " + plant->get_joint(j).name() + ", position@ " +
        std::to_string(plant->get_joint(j).position_start()) + ", velocity@ " +
        std::to_string(plant->get_joint(j).velocity_start()));
    if (index < fake_plant.num_joints())
      drake::log()->info(
          "FAKE PLANT JOINT: " + fake_plant.get_joint(j).name() +
          ", position@" +
          std::to_string(fake_plant.get_joint(j).position_start()) +
          ", velocity@" +
          std::to_string(fake_plant.get_joint(j).velocity_start()));
  }
  drake::log()->info("MakeActuationMatrix() = B matrix");
  drake::log()->info(plant->MakeActuationMatrix());

  //////////////////////////////////////////////////////////////////////////////
  // Diagram build starts.

  // Create InverseDynamicsController using fake_plant.
  const int Q = plant->num_positions();
  const int V = plant->num_velocities();
  const int U = fake_plant.num_actuators();
  Eigen::VectorXd Kp_ = Eigen::VectorXd::Ones(U) * 20.0;
//  Kp_.head(6) = Eigen::VectorXd::Ones(6) * 100.0;
  Eigen::VectorXd Ki_ = Eigen::VectorXd::Ones(U) * 0.0;
  Eigen::VectorXd Kd_ = Eigen::VectorXd::Ones(U) * 2.0;
//  Kp_.setZero();
//  Kd_.setZero();

  auto feed_forward_controller =
      builder
          .AddSystem<systems::controllers::InverseDynamicsController<double>>(
              fake_plant, Kp_, Ki_, Kd_, true);

  // Set desired position [q,v]' for IDC as feedback reference.
  VectorX<double> constant_pos_value =
      VectorX<double>::Ones(2 * U) * FLAGS_constant_pos;
  constant_pos_value[plant->GetJointByName("j_hip_y").position_start()-9] = 0.43;
  constant_pos_value[plant->GetJointByName("j_knee_y").position_start()-9] = -0.91;
  constant_pos_value[plant->GetJointByName("j_ankle_y").position_start()-9] = 0.47;
//  constant_pos_value[plant->GetJointByName("j_hip_y").position_start()-9] = 0.76;
//  constant_pos_value[plant->GetJointByName("j_knee_y").position_start()-9] = -1.59;
//  constant_pos_value[plant->GetJointByName("j_ankle_y").position_start()-9] = 0.82;
  constant_pos_value[plant->GetJointByName("j_hip_y").position_start()-9] = 0.96;
  constant_pos_value[plant->GetJointByName("j_knee_y").position_start()-9] = -2.00;
  constant_pos_value[plant->GetJointByName("j_ankle_y").position_start()-9] = 1.04;
  constant_pos_value[plant->GetJointByName("j_l_shoulder_x").position_start()-9] = 1.57;
  constant_pos_value[plant->GetJointByName("j_r_shoulder_x").position_start()-9] = -1.57;
  auto desired_constant_source =
      builder.AddSystem<systems::ConstantVectorSource<double>>(
          constant_pos_value);
  desired_constant_source->set_name("desired_constant_source");
  builder.Connect(desired_constant_source->get_output_port(),
                  feed_forward_controller->get_input_port_desired_state());

  // Select plant states and feed into controller with fake_plant.
  Eigen::MatrixXd feedback_joints_selector =
      Eigen::MatrixXd::Zero(2 * U, Q + V);
  for (multibody::JointActuatorIndex a(0); a < fake_plant.num_actuators();
       ++a) {
    std::string fake_joint_name =
        fake_plant.get_joint_actuator(a).joint().name();
    feedback_joints_selector(
        fake_plant.get_joint_actuator(a).joint().position_start(),
        plant->GetJointByName(fake_joint_name).position_start()) = 1;
    feedback_joints_selector(
        fake_plant.get_joint_actuator(a).joint().velocity_start() +
            fake_plant.num_positions(),
        plant->GetJointByName(fake_joint_name).velocity_start() +
            plant->num_positions()) = 1;
  }
  drake::log()->info("feedback_joints_selector");
  drake::log()->info(feedback_joints_selector);
  // Use Gain system to convert plant output to IDC state input
  systems::MatrixGain<double>& select_IDC_states =
      *builder.AddSystem<systems::MatrixGain<double>>(feedback_joints_selector);
  //  builder.Connect(plant->get_state_output_port(),
  //                  feed_forward_controller->get_input_port_estimated_state());
  builder.Connect(plant->get_state_output_port(),
                  select_IDC_states.get_input_port());
  builder.Connect(select_IDC_states.get_output_port(),
                  feed_forward_controller->get_input_port_estimated_state());

  // Select generalized control signal and feed into plant->
  Eigen::MatrixXd generalized_actuation_selector =
      Eigen::MatrixXd::Zero(plant->num_velocities(), U);
  //  generalized_actuation_selector.bottomRightCorner(U, U) =
  //      Eigen::MatrixXd::Identity(U, U);
  for (multibody::JointIndex j(0); j < fake_plant.num_velocities(); ++j) {
    std::string fake_joint_name = fake_plant.get_joint(j).name();
    generalized_actuation_selector(
        plant->GetJointByName(fake_joint_name).velocity_start(),
        fake_plant.get_joint(j).velocity_start()) = 1;
  }
  drake::log()->info("generalized_actuation_selector");
  drake::log()->info(generalized_actuation_selector);
  systems::MatrixGain<double>* select_generalized_actuation_states =
      builder.AddSystem<systems::MatrixGain<double>>(
          generalized_actuation_selector);
  //  builder.Connect(feed_forward_controller->get_output_port_control(),
  //                  plant->get_applied_generalized_force_input_port());
  builder.Connect(feed_forward_controller->get_output_port_control(),
                  select_generalized_actuation_states->get_input_port());
  builder.Connect(select_generalized_actuation_states->get_output_port(),
                  plant->get_applied_generalized_force_input_port());

//  // Create the WheelController.
//  auto wc =
//      builder.AddSystem<WheelController>(plant, plant_model_instance_index);
//  builder.Connect(plant->get_state_output_port(), wc->get_input_port(0));
//  builder.Connect(wc->get_output_port(0), plant->get_actuation_input_port());

  // Create the WheelController.
  lcm::DrakeLcm lcm;
  auto wvc = builder.AddSystem<WheelVelocityController>(*plant, plant_model_instance_index);
  builder.Connect(plant->get_state_output_port(), wvc->get_input_port(0));
  builder.Connect(wvc->get_output_port(0), plant->get_actuation_input_port());

  // The feedback controller that map trajectory tracking error to velocity input.
  auto vs = builder.AddSystem<VelocitySource>(*plant, plant_model_instance_index, lcm);
  builder.Connect(plant->get_state_output_port(), vs->get_input_port(0));
  builder.Connect(vs->get_output_port(0), wvc->get_input_port(1));

//  // Design the fix point to follow.
//  const std::vector<double> kTimes{0.0, 2.5, 5.0, 7.5, 10.0};
//  std::vector<Eigen::MatrixXd> knots(kTimes.size());
//  knots[0] = Eigen::Vector2d(0.0,0);
//  knots[1] = Eigen::Vector2d(0.5,0);
//  knots[2] = Eigen::Vector2d(1.0,0);
//  knots[3] = Eigen::Vector2d(1.5,0);
//  knots[4] = Eigen::Vector2d(2.0,0);

  // Design the straight trajectory to follow.
  std::vector<double> kTimes{0.0, 2.0, 4.0, 6.0, 9.0};
  for (size_t i = 0; i < kTimes.size(); ++i) {
    kTimes[i] = kTimes[i] * 0.7;
  }
  std::vector<Eigen::MatrixXd> knots(kTimes.size());
  knots[0] = Eigen::Vector2d(0,0);
  knots[1] = Eigen::Vector2d(1,0);
  knots[2] = Eigen::Vector2d(4,0);
  knots[3] = Eigen::Vector2d(7,0);
  knots[4] = Eigen::Vector2d(8,0);

//  // Design a curvy trajectory to follow.
//  std::vector<double> kTimes{0.0, 2.0, 4.0, 6.0, 8.0, 10, 12};
//  for (size_t i = 0; i < kTimes.size(); ++i) {
//    kTimes[i] = kTimes[i] * 0.7;
//  }
//  std::vector<Eigen::MatrixXd> knots(kTimes.size());
//  knots[0] = Eigen::Vector2d(0, 0);
//  knots[1] = Eigen::Vector2d(1, 0);
//  knots[2] = Eigen::Vector2d(3, 0.5);
//  knots[3] = Eigen::Vector2d(5, 0);
//  knots[4] = Eigen::Vector2d(7, -0.5);
//  knots[5] = Eigen::Vector2d(9, 0);
//  knots[6] = Eigen::Vector2d(10, 0);

//  // Design circle.
//  std::vector<double> kTimes;
//  std::vector<Eigen::MatrixXd> knots;
//
//  for (int i = 0; i<FLAGS_precision; i++) {
//    kTimes.push_back(i * FLAGS_inteval);
//    double theta = M_PI*2 / FLAGS_precision * i;
//    knots.push_back(
//        Eigen::Vector2d(FLAGS_circle_radius*std::sin(theta),
//        FLAGS_circle_radius*std::cos(theta)-FLAGS_circle_radius));
//  }
//  for (int i = 0; i<FLAGS_precision; i++) {
//    kTimes.push_back((i+FLAGS_precision) * FLAGS_inteval);
//    double theta = M_PI*2 / FLAGS_precision * i;
//    knots.push_back(
//        Eigen::Vector2d(FLAGS_circle_radius*std::sin(theta),
//        -FLAGS_circle_radius*std::cos(theta)+FLAGS_circle_radius));
//  }
//  kTimes.push_back(2*FLAGS_precision*FLAGS_inteval);
//  knots.push_back(Eigen::Vector2d(0,0));

  trajectories::PiecewisePolynomial<double> trajectory =
      trajectories::PiecewisePolynomial<double>::Pchip(kTimes, knots);

  // Adds a trajectory source for desired state.
  auto traj_src = builder.AddSystem<systems::TrajectorySource<double>>(
      trajectory, 2 /* outputs q + v + a*/);
  traj_src->set_name("trajectory_source");
  builder.Connect(traj_src->get_output_port(),
                  vs->get_input_port(1));

  //////////////////////////////////////////////////////////////////////////////
  // TODO: Compute the com height using multibody plant function.

  // Create a system to transform COM acceleration to Joint acceleration.
  auto j_inverse = builder.AddSystem<JInverse>(*plant, fake_plant,
      plant_model_instance_index, fake_plant_model_instance_index, lcm);
  builder.Connect(j_inverse->get_output_port(0), feed_forward_controller->get_input_port_desired_acceleration());
  builder.Connect(traj_src->get_output_port(),
                  j_inverse->get_input_port(j_inverse->base_trajectory_port_index));

  // Given the trajectory of reference ZMP, we compute the CoM trajectory.
//  const double z_cm = 0.76;
//  const double z_cm = 0.67;
  const double z_cm = 0.593561;
  Eigen::Vector4d x0(0, 0, 0, 0);
  systems::controllers::ZMPPlanner zmp_planner;
  zmp_planner.Plan(trajectory, x0, z_cm, 9.81, Eigen::Matrix2d::Identity(), 0.2 * Eigen::Matrix2d::Identity());
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
//  for (int t = 0; t < N; t=t+5) {
//    names.push_back("CoM" + std::to_string(int(t)));
//    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
//    pose.translation() = Eigen::Vector3d(result.nominal_com(0, t), result.nominal_com(1, t), z_cm);
//    poses.push_back(pose);
//  }
  for (int t = 0; t < N; t=t+5) {
    names.push_back("ZMP" + std::to_string(int(t)));
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = Eigen::Vector3d(result.desired_zmp(0, t), result.desired_zmp(1, t), 0);
    poses.push_back(pose);
  }
  PublishFramesToLcm("DRAKE_DRAW_FRAMES2", poses, names, &lcm);

  // Diagram build finish.
  //////////////////////////////////////////////////////////////////////////////

  // Connect plant with scene_graph to get collision information.
  DRAKE_DEMAND(!!plant->get_source_id());
  builder.Connect(
      plant->get_geometry_poses_output_port(),
      scene_graph.get_source_pose_port(plant->get_source_id().value()));
  builder.Connect(scene_graph.get_query_output_port(),
                  plant->get_geometry_query_input_port());

  geometry::ConnectDrakeVisualizer(&builder, scene_graph);

  // Create a context for this diagram and plant->
  std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());
  // Create plant_context to set velocity.
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(*plant, diagram_context.get());
  // TODO: Get rid of this hacky way.
  j_inverse->plant_context_ = &plant_context;

  // Set the robot COM position, make sure the robot base is off the ground.
  drake::VectorX<double> positions =
      plant->GetPositions(plant_context, plant_model_instance_index);
  positions[plant->GetJointByName("j_hip_y").position_start()] = 0.43;
  positions[plant->GetJointByName("j_knee_y").position_start()] = -0.91;
  positions[plant->GetJointByName("j_ankle_y").position_start()] = 0.47;
//  positions[plant->GetJointByName("j_hip_y").position_start()] = 0.76;
//  positions[plant->GetJointByName("j_knee_y").position_start()] = -1.59;
//  positions[plant->GetJointByName("j_ankle_y").position_start()] = 0.82;
  positions[plant->GetJointByName("j_hip_y").position_start()] = 0.96;
  positions[plant->GetJointByName("j_knee_y").position_start()] = -2.00;
  positions[plant->GetJointByName("j_ankle_y").position_start()] = 1.04;
  positions[plant->GetJointByName("j_l_shoulder_x").position_start()] = 1.57;
  positions[plant->GetJointByName("j_r_shoulder_x").position_start()] = -1.57;
  positions[6] = FLAGS_init_height;
  plant->SetPositions(&plant_context, positions);
  // Measure and report the CoM position.
  Eigen::Vector3d p_cm;
  plant->CalcCenterOfMassPosition(plant_context, &p_cm);
  drake::log()->info("Center of mass position at home stance.");
  drake::log()->info(p_cm.transpose());

  // Set robot init velocity for every joint.
  drake::VectorX<double> velocities =
      Eigen::VectorXd::Ones(plant->num_velocities()) * 0.0;
  plant->SetVelocities(&plant_context, velocities);

  std::this_thread::sleep_for(std::chrono::seconds(2));
  // Set up simulator.
  drake::log()->info("\nNow starts Simulation\n");
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_publish_every_time_step(true);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
//  simulator.AdvanceTo(FLAGS_simulation_time);
  simulator.AdvanceTo(kTimes.back());


//  CallPython("figure", 1);
//  CallPython("clf");
//  CallPython("plot", kTimes, kTimes);
//  CallPython("setvars", "x_val", kTimes, "w_val", kTimes);
//  CallPython("plt.xlabel", "x");
//  CallPython("plt.ylabel", "w, I_B");
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
