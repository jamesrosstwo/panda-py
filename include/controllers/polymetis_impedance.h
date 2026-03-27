// PolymetisImpedance: Cartesian impedance controller with explicit per-axis
// damping and per-joint nullspace gains, matching Polymetis conventions.
//
// This is a fork of panda-py's CartesianImpedance controller. The only changes
// from the original are:
//   1. K_d (6x6 damping) is specified directly, not derived from K_p via a
//      scalar damping_ratio.
//   2. Nullspace stiffness is a 7D vector (per-joint Kq), not a scalar.
//   3. Nullspace damping is a 7D vector (per-joint Kqd), not derived from
//      nullspace stiffness via 2*sqrt(k).
//
// Everything else (orientation error, nullspace projector, EMA filter, coriolis
// compensation) is identical to the original CartesianImpedance.
//
// To install: copy this header + .cpp into a panda-py clone, add to CMakeLists
// and _core.cpp bindings, then `pip install .`

#pragma once
#include <atomic>
#include <mutex>

#include "constants.h"
#include "controllers/controller.h"
#include "utils.h"

class PolymetisImpedance : public TorqueController {
 public:
  static const Eigen::Matrix<double, 6, 6> kDefaultImpedance;
  static const Eigen::Matrix<double, 6, 6> kDefaultDamping;
  static const Vector7d kDefaultNullspaceStiffness;
  static const Vector7d kDefaultNullspaceDamping;
  static const double kDefaultFilterCoeff;

  PolymetisImpedance(
      const Eigen::Matrix<double, 6, 6> &impedance = kDefaultImpedance,
      const Eigen::Matrix<double, 6, 6> &damping = kDefaultDamping,
      const Vector7d &nullspace_stiffness = kDefaultNullspaceStiffness,
      const Vector7d &nullspace_damping = kDefaultNullspaceDamping,
      const double &filter_coeff = kDefaultFilterCoeff);

  franka::Torques step(const franka::RobotState &robot_state,
                       franka::Duration &duration) override;
  void setControl(const Eigen::Vector3d &position,
                  const Eigen::Vector4d &orientation,
                  const Vector7d &q_nullspace = kJointPositionStart);
  void setImpedance(const Eigen::Matrix<double, 6, 6> &impedance);
  void setDamping(const Eigen::Matrix<double, 6, 6> &damping);
  void setNullspaceStiffness(const Vector7d &nullspace_stiffness);
  void setNullspaceDamping(const Vector7d &nullspace_damping);
  void setFilter(const double filter_coeff);
  void start(const franka::RobotState &robot_state,
             std::shared_ptr<franka::Model> model) override;
  void stop(const franka::RobotState &robot_state,
            std::shared_ptr<franka::Model> model) override;
  bool isRunning() override;
  const std::string name() override;

 private:
  // Cartesian gains
  Eigen::Matrix<double, 6, 6> K_p_, K_d_, K_p_target_, K_d_target_;
  // Nullspace gains (per-joint)
  Vector7d Kq_, Kqd_, Kq_target_, Kqd_target_;
  // Setpoints
  Eigen::Vector3d position_d_, position_d_target_;
  Eigen::Quaterniond orientation_d_, orientation_d_target_;
  Vector7d q_nullspace_d_, q_nullspace_d_target_;
  // Filter
  double filter_coeff_;
  // Synchronisation
  std::mutex mux_;
  std::atomic<bool> motion_finished_;
  std::shared_ptr<franka::Model> model_;

  void _updateFilter();
};
