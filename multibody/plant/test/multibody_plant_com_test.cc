/* clang-format off to disable clang-format-includes */
#include "drake/multibody/plant/multibody_plant.h"
/* clang-format on */

#include <memory>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/scene_graph.h"
#include "drake/systems/framework/context.h"

namespace drake {
namespace multibody {
namespace {

class EmptyMultibodyPlantCenterOfMassTest : public ::testing::Test {
 public:
  void SetUp() override {
    plant_.Finalize();
    context_ = plant_.CreateDefaultContext();
  }

 protected:
  MultibodyPlant<double> plant_;
  std::unique_ptr<systems::Context<double>> context_;
};

TEST_F(EmptyMultibodyPlantCenterOfMassTest, GetCenterOfMassPosition) {
  Eigen::Vector3d p_WBcm;
  EXPECT_THROW(plant_.CalcCenterOfMassPosition(*context_, &p_WBcm),
               std::runtime_error);
}

class MultibodyPlantCenterOfMassTest : public ::testing::Test {
 public:
  void SetUp() override {
    cubic_instance_ = plant_.AddModelInstance("Cubics");
    sphere_instance_ = plant_.AddModelInstance("Spheres");

    mass_s1 = 10.0;
    p_SoScm_s1 = Eigen::Vector3d(1.5, 2.1, 3.9);
    plant_.AddRigidBody(
        "Sphere1", sphere_instance_,
        multibody::SpatialInertia<double>::MakeFromCentralInertia(
            mass_s1, p_SoScm_s1,
            multibody::RotationalInertia<double>(1e-8, 1e-8, 1e-8)));

    mass_c1 = 20.0;
    p_CoCcm_c1 = Eigen::Vector3d(3.3, 2.5, 0.7);
    plant_.AddRigidBody(
        "Cubic1", cubic_instance_,
        multibody::SpatialInertia<double>::MakeFromCentralInertia(
            mass_c1, p_CoCcm_c1,
            multibody::RotationalInertia<double>(1e-8, 1e-8, 1e-8)));

    plant_.Finalize();
    context_ = plant_.CreateDefaultContext();
  }

 protected:
  MultibodyPlant<double> plant_;
  std::unique_ptr<systems::Context<double>> context_;
  ModelInstanceIndex cubic_instance_;
  ModelInstanceIndex sphere_instance_;
  double mass_s1;
  double mass_c1;
  Eigen::Vector3d p_SoScm_s1;
  Eigen::Vector3d p_CoCcm_c1;
};

TEST_F(MultibodyPlantCenterOfMassTest, CenterOfMassPositionAfterTranslation) {
  Eigen::Vector3d p_WBcm;
  plant_.CalcCenterOfMassPosition(*context_, &p_WBcm);
  Eigen::Vector3d result =
      (p_SoScm_s1 * mass_s1 + p_CoCcm_c1 * mass_c1) / (mass_s1 + mass_c1);
  EXPECT_TRUE(CompareMatrices(p_WBcm, result, 1e-6));

  drake::VectorX<double> sphere_positions =
      plant_.GetPositions(*context_, sphere_instance_);
  Eigen::Vector3d p_WSo_W;
  p_WSo_W << 1.1, 2.3, 3.7;
  sphere_positions.segment(4, 3) = p_WSo_W;
  plant_.SetPositions(context_.get(), sphere_instance_, sphere_positions);

  drake::VectorX<double> cubic_positions =
      plant_.GetPositions(*context_, cubic_instance_);
  Eigen::Vector3d p_WCo_W;
  p_WCo_W << -5.2, 10.4, -6.8;
  cubic_positions.segment(4, 3) = p_WCo_W;
  plant_.SetPositions(context_.get(), cubic_instance_, cubic_positions);

  result =
      ((p_WSo_W + p_SoScm_s1) * mass_s1 + (p_WCo_W + p_CoCcm_c1) * mass_c1) /
      (mass_s1 + mass_c1);
  plant_.CalcCenterOfMassPosition(*context_, &p_WBcm);
  EXPECT_TRUE(CompareMatrices(p_WBcm, result, 1e-6));

  // Try empty model_instances.
  std::unordered_set<ModelInstanceIndex> model_instances;
  plant_.CalcCenterOfMassPosition(*context_, &p_WBcm, model_instances);
  EXPECT_TRUE(CompareMatrices(p_WBcm, result, 1e-6));

  // Try one instance in model_instances.
  model_instances.insert(cubic_instance_);
  plant_.CalcCenterOfMassPosition(*context_, &p_WBcm, model_instances);
  EXPECT_TRUE(CompareMatrices(p_WBcm, p_WCo_W + p_CoCcm_c1, 1e-6));

  // Try all instance in model_instances.
  model_instances.insert(sphere_instance_);
  plant_.CalcCenterOfMassPosition(*context_, &p_WBcm, model_instances);
  EXPECT_TRUE(CompareMatrices(p_WBcm, result, 1e-6));

  // Try error instance in model_instances.
  ModelInstanceIndex error_index(10);
  model_instances.insert(error_index);
  EXPECT_THROW(
      plant_.CalcCenterOfMassPosition(*context_, &p_WBcm, model_instances),
      std::runtime_error);

  // Try after translation and rotation.
  Eigen::Quaterniond quat_s1(1, 2, 3, 4);
  quat_s1.normalize();
  math::RotationMatrixd rot_s1(quat_s1);
  sphere_positions[0] = quat_s1.w();
  sphere_positions.segment(1, 3) = quat_s1.vec();  // Set x, y, z.
  plant_.SetPositions(context_.get(), sphere_instance_, sphere_positions);

  Eigen::Quaterniond quat_c1(-5, 6, -7, 8);
  quat_c1.normalize();
  math::RotationMatrixd rot_c1(quat_c1);
  cubic_positions[0] = quat_c1.w();
  cubic_positions.segment(1, 3) = quat_c1.vec();  // Set x, y, z.
  plant_.SetPositions(context_.get(), cubic_instance_, cubic_positions);

  Eigen::Vector3d rotation_result =
      ((p_WSo_W + rot_s1 * p_SoScm_s1) * mass_s1 +
       (p_WCo_W + rot_c1 * p_CoCcm_c1) * mass_c1) /
      (mass_s1 + mass_c1);
  plant_.CalcCenterOfMassPosition(*context_, &p_WBcm);
  EXPECT_TRUE(CompareMatrices(p_WBcm, rotation_result, 1e-6));
}

TEST_F(MultibodyPlantCenterOfMassTest, GetCenterOfMassVelocity) {

//  const Body<double>& c1 = plant_.GetBodyByName("Cubic1");
//  const Body<double>& s1 = plant_.GetBodyByName("Sphere1");

  // Test with zero velocity.
  plant_.SetVelocities(context_.get(), sphere_instance_,
                       Eigen::VectorXd::Zero(6));
  plant_.SetVelocities(context_.get(), cubic_instance_,
                       Eigen::VectorXd::Zero(6));
  Eigen::Vector3d v_WBcm;
  plant_.CalcCenterOfMassVelocity(*context_, &v_WBcm);
  EXPECT_TRUE(CompareMatrices(v_WBcm, Eigen::Vector3d::Zero(), 1e-6));

  // Test zero velocity at rotated pose.
  // Set random rotational positions.
  drake::VectorX<double> sphere_positions =
      plant_.GetPositions(*context_, sphere_instance_);
  math::RotationMatrixd rot_s1(math::RollPitchYawd(0.3, -1.5, 0.7));
  sphere_positions.segment(0, 4) = rot_s1.ToQuaternionAsVector4();
  plant_.SetPositions(context_.get(), sphere_instance_, sphere_positions);
  drake::VectorX<double> cubic_positions =
      plant_.GetPositions(*context_, cubic_instance_);
  math::RotationMatrixd rot_c1(math::RollPitchYawd(-2.3, -3.5, 1.2));
  cubic_positions.segment(0, 4) = rot_c1.ToQuaternionAsVector4();
  plant_.SetPositions(context_.get(), cubic_instance_, cubic_positions);

  // Test bodies with translational velocity at rotated pose.
  Eigen::VectorXd vel_s1(6);
  vel_s1 << 0,0,0,4,5,6;
  plant_.SetVelocities(context_.get(), sphere_instance_, vel_s1);
  Eigen::VectorXd vel_c1(6);
  vel_c1 << 0,0,0,5,-6,7;
  plant_.SetVelocities(context_.get(), cubic_instance_, vel_c1);
  Eigen::VectorXd result_velocity =
      (vel_s1.tail(3) * mass_s1 + vel_c1.tail(3) * mass_c1) /
      (mass_s1 + mass_c1);
  plant_.CalcCenterOfMassVelocity(*context_, &v_WBcm);
  EXPECT_TRUE(CompareMatrices(v_WBcm, result_velocity, 1e-6));

  // Test bodies with random velocity at random pose.
  // Set random velocities.
  vel_s1.head(3) = Eigen::Vector3d{1, -2, 3}; vel_s1.tail(3) = Eigen::Vector3d{-6, 7, -8};
  plant_.SetVelocities(context_.get(), sphere_instance_, vel_s1);
  vel_c1.head(3) = Eigen::Vector3d{-4, 5, -6}; vel_c1.tail(3) = Eigen::Vector3d{9, -10, 20};
  plant_.SetVelocities(context_.get(), cubic_instance_, vel_c1);
  // Set random translational positions.
  sphere_positions.segment(4, 3) = Eigen::Vector3d{5.2, -3.1, 10.9};
  plant_.SetPositions(context_.get(), sphere_instance_, sphere_positions);
  cubic_positions.segment(4, 3) = Eigen::Vector3d{-70.2, 9.8, 843.1};
  plant_.SetPositions(context_.get(), cubic_instance_, cubic_positions);
  // Compute current Center of Mass position and velocity.
  Eigen::Vector3d p_SoScm_W = rot_s1.matrix() * p_SoScm_s1;
  Eigen::Vector3d p_CoCcm_W = rot_c1.matrix() * p_CoCcm_c1;
  result_velocity =
          ((Eigen::Vector3d(vel_s1.head(3)).cross(p_SoScm_W) + vel_s1.tail(3)) * mass_s1 +
              (Eigen::Vector3d(vel_c1.head(3)).cross(p_CoCcm_W) + vel_c1.tail(3)) * mass_c1) /
          (mass_s1 + mass_c1);
  plant_.CalcCenterOfMassVelocity(*context_, &v_WBcm);
  EXPECT_TRUE(CompareMatrices(v_WBcm, result_velocity, 1e-6));
}
}  // namespace
}  // namespace multibody
}  // namespace drake
