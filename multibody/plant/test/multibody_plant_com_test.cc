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

    const multibody::RigidBody<double>& s1 = plant_.AddRigidBody(
        "Sphere1", sphere_instance_,
        multibody::SpatialInertia<double>::MakeFromCentralInertia(
            10.0, Eigen::Vector3d::Ones() * 1e-8,
            multibody::RotationalInertia<double>(1e-8, 1e-8, 1e-8)));
    (void)s1;

    const multibody::RigidBody<double>& c1 = plant_.AddRigidBody(
        "Cubic1", cubic_instance_,
        multibody::SpatialInertia<double>::MakeFromCentralInertia(
            20.0, Eigen::Vector3d::Ones() * 1e-8,
            multibody::RotationalInertia<double>(1e-8, 1e-8, 1e-8)));
    (void)c1;

    plant_.Finalize();
    context_ = plant_.CreateDefaultContext();
  }

 protected:
  MultibodyPlant<double> plant_;
  std::unique_ptr<systems::Context<double>> context_;
  ModelInstanceIndex cubic_instance_;
  ModelInstanceIndex sphere_instance_;
};

TEST_F(MultibodyPlantCenterOfMassTest, GetCenterOfMassPosition) {
  Eigen::Vector3d p_WBcm;
  plant_.CalcCenterOfMassPosition(*context_, &p_WBcm);
  EXPECT_TRUE(CompareMatrices(p_WBcm, Eigen::Vector3d::Zero(), 1e-6));
}

TEST_F(MultibodyPlantCenterOfMassTest, SetPosition) {
  drake::VectorX<double> sphere_positions =
      plant_.GetPositions(*context_, sphere_instance_);
  Eigen::Vector3d p_WBo_s1;
  p_WBo_s1 << 1.1, 2.3, 3.7;
  sphere_positions.segment(4, 3) = p_WBo_s1;
  plant_.SetPositions(context_.get(), sphere_instance_, sphere_positions);

  drake::VectorX<double> cubic_positions =
      plant_.GetPositions(*context_, cubic_instance_);
  Eigen::Vector3d p_WBo_c1;
  p_WBo_c1 << -5.2, 10.4, -6.8;
  cubic_positions.segment(4, 3) = p_WBo_c1;
  plant_.SetPositions(context_.get(), cubic_instance_, cubic_positions);

  Eigen::Vector3d p_WBcm;
  Eigen::Vector3d result;
  result << -3.1, 7.7, -3.3;
  plant_.CalcCenterOfMassPosition(*context_, &p_WBcm);
  EXPECT_TRUE(CompareMatrices(p_WBcm, result, 1e-6));

  // Try empty unordered_set.
  std::unordered_set<ModelInstanceIndex> model_instances;
  plant_.CalcCenterOfMassPosition(*context_, &p_WBcm, model_instances);
  EXPECT_TRUE(CompareMatrices(p_WBcm, result, 1e-6));

  model_instances.insert(cubic_instance_);
  plant_.CalcCenterOfMassPosition(*context_, &p_WBcm, model_instances);
  EXPECT_TRUE(CompareMatrices(p_WBcm, p_WBo_c1, 1e-6));

  ModelInstanceIndex error_index(10);
  model_instances.insert(error_index);
  EXPECT_THROW(
      plant_.CalcCenterOfMassPosition(*context_, &p_WBcm, model_instances),
      std::runtime_error);
}

}  // namespace
}  // namespace multibody
}  // namespace drake
