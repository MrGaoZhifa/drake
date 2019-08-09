//
// Created by zhaoyuangu on 6/26/19.
//

/// Usage
/*
 * PublishFramesToLcm("DRAKE_DRAW_FRAMES", poses, names, &lcm);
 * PublishTrajectoryToLcm("DRAKE_DRAW_TRAJECTORY", poses, names, &lcm);
 *
 * */

#ifndef DRAKE_DRAKE_EXAMPLES_EVE_EVE_COMMON_H_
#define DRAKE_DRAKE_EXAMPLES_EVE_EVE_COMMON_H_

#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcmt_viewer_draw.hpp"
#include "drake/lcmt_contact_results_for_viz.hpp"
#include "drake/lcmt_contact_info_for_viz.hpp"

namespace drake {
namespace examples {
namespace eve {

void PublishContactToLcm( const std::vector<Eigen::VectorXd> &contact_points,
                          const std::vector<Eigen::VectorXd> &contact_forces,
                          drake::lcm::DrakeLcmInterface *dlcm);

void PublishContactToLcm( const std::string &channel_name,
                          const std::vector<Eigen::VectorXd> &contact_points,
                          const std::vector<Eigen::VectorXd> &contact_forces,
                          drake::lcm::DrakeLcmInterface *dlcm);

void PublishTrajectoryToLcm(const std::string &channel_name,
                            const std::vector<Eigen::Isometry3d> &poses,
                            const std::vector<std::string> &names,
                            drake::lcm::DrakeLcmInterface *dlcm);

void PublishFramesToLcm(const std::string &channel_name,
                        const std::vector<Eigen::Isometry3d> &poses,
                        const std::vector<std::string> &names,
                        drake::lcm::DrakeLcmInterface *dlcm);

void PublishFramesToLcm(
    const std::string &channel_name,
    const std::unordered_map<std::string, Eigen::Isometry3d> &name_to_frame_map,
    drake::lcm::DrakeLcmInterface *lcm);

}  // namespace eve
}  // namespace examples
}  // namespace drake

#endif //DRAKE_DRAKE_EXAMPLES_EVE_EVE_COMMON_H_
