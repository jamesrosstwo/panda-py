// HybridJointImpedance: Joint-space impedance controller with
// Cartesian-augmented gains, matching the actual Polymetis
// start_cartesian_impedance() architecture.
//
// Control law (joint-space PD):
//   Kp_eff = J^T * Kx * J + diag(Kq)
//   Kd_eff = J^T * Kxd * J + diag(Kqd)
//   tau = Kp_eff * (q_desired - q) + Kd_eff * (0 - dq) + coriolis
//
// set_control() takes desired joint positions directly.
// Use panda_py.ik() on the Python side to convert Cartesian targets.

#pragma once
#include <atomic>
#include <mutex>

#include "constants.h"
#include "controllers/controller.h"
#include "utils.h"

class HybridJointImpedance : public TorqueController {
 public:
  static const Eigen::Matrix<double, 6, 6> kDefaultKx;
  static const Eigen::Matrix<double, 6, 6> kDefaultKxd;
  static const Vector7d kDefaultKq;
  static const Vector7d kDefaultKqd;
  static const double kDefaultFilterCoeff;

  HybridJointImpedance(
      const Eigen::Matrix<double, 6, 6> &Kx = kDefaultKx,
      const Eigen::Matrix<double, 6, 6> &Kxd = kDefaultKxd,
      const Vector7d &Kq = kDefaultKq,
      const Vector7d &Kqd = kDefaultKqd,
      const double &filter_coeff = kDefaultFilterCoeff);

  franka::Torques step(const franka::RobotState &robot_state,
                       franka::Duration &duration) override;
  void setControl(const Vector7d &q_desired);
  void setKx(const Eigen::Matrix<double, 6, 6> &Kx);
  void setKxd(const Eigen::Matrix<double, 6, 6> &Kxd);
  void setKq(const Vector7d &Kq);
  void setKqd(const Vector7d &Kqd);
  void setFilter(const double filter_coeff);
  void start(const franka::RobotState &robot_state,
             std::shared_ptr<franka::Model> model) override;
  void stop(const franka::RobotState &robot_state,
            std::shared_ptr<franka::Model> model) override;
  bool isRunning() override;
  const std::string name() override;

 private:
  // Cartesian gains (used to build effective joint gains)
  Eigen::Matrix<double, 6, 6> Kx_, Kxd_, Kx_target_, Kxd_target_;
  // Joint-space gains
  Vector7d Kq_, Kqd_, Kq_target_, Kqd_target_;
  // Desired joint positions
  Vector7d q_d_, q_d_target_;
  // Filter
  double filter_coeff_;
  // Synchronisation
  std::mutex mux_;
  std::atomic<bool> motion_finished_;
  std::shared_ptr<franka::Model> model_;

  void _updateFilter();
};
