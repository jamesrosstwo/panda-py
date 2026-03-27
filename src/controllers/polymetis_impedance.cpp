// PolymetisImpedance: fork of CartesianImpedance with explicit gains.
// See polymetis_impedance.h for the rationale.

#include "controllers/polymetis_impedance.h"

#include <iostream>

#include "panda.h"

// --- Polymetis franka_hardware.yaml defaults ---
// Kx  = [750, 750, 750, 15, 15, 15]
// Kxd = [37,  37,  37,  2,  2,  2]
// Kq  = [40, 30, 50, 25, 35, 25, 10]
// Kqd = [4,  6,  5,  5,  3,  2,  1]

// clang-format off
double pm_imp_data[36] = {750,  0,   0,  0,  0,  0,
                            0, 750,  0,  0,  0,  0,
                            0,   0, 750,  0,  0,  0,
                            0,   0,   0, 15,  0,  0,
                            0,   0,   0,  0, 15,  0,
                            0,   0,   0,  0,  0, 15};
double pm_dmp_data[36] = {37,  0,  0,  0,  0,  0,
                           0, 37,  0,  0,  0,  0,
                           0,  0, 37,  0,  0,  0,
                           0,  0,  0,  2,  0,  0,
                           0,  0,  0,  0,  2,  0,
                           0,  0,  0,  0,  0,  2};
// clang-format on

const Eigen::Matrix<double, 6, 6> PolymetisImpedance::kDefaultImpedance =
    Eigen::Matrix<double, 6, 6>(pm_imp_data);
const Eigen::Matrix<double, 6, 6> PolymetisImpedance::kDefaultDamping =
    Eigen::Matrix<double, 6, 6>(pm_dmp_data);
const Vector7d PolymetisImpedance::kDefaultNullspaceStiffness =
    (Vector7d() << 40, 30, 50, 25, 35, 25, 10).finished();
const Vector7d PolymetisImpedance::kDefaultNullspaceDamping =
    (Vector7d() << 4, 6, 5, 5, 3, 2, 1).finished();
const double PolymetisImpedance::kDefaultFilterCoeff = 1.0;

PolymetisImpedance::PolymetisImpedance(
    const Eigen::Matrix<double, 6, 6> &impedance,
    const Eigen::Matrix<double, 6, 6> &damping,
    const Vector7d &nullspace_stiffness, const Vector7d &nullspace_damping,
    const double &filter_coeff) {
  K_p_ = impedance;
  K_p_target_ = impedance;
  K_d_ = damping;
  K_d_target_ = damping;
  Kq_ = nullspace_stiffness;
  Kq_target_ = nullspace_stiffness;
  Kqd_ = nullspace_damping;
  Kqd_target_ = nullspace_damping;
  filter_coeff_ = filter_coeff;
};

// Identical to CartesianImpedance::step except:
//   - K_d is used directly (not derived from K_p)
//   - Nullspace uses per-joint Kq and Kqd vectors
franka::Torques PolymetisImpedance::step(
    const franka::RobotState &robot_state, franka::Duration &duration) {
  Eigen::Vector3d position_d;
  Eigen::Quaterniond orientation_d;
  Vector7d q_nullspace_d;
  Eigen::Matrix<double, 6, 6> K_p, K_d;
  Vector7d Kq, Kqd;

  mux_.lock();
  _updateFilter();
  K_p = K_p_;
  K_d = K_d_;
  Kq = Kq_;
  Kqd = Kqd_;
  position_d = position_d_;
  orientation_d = orientation_d_;
  q_nullspace_d = q_nullspace_d_;
  mux_.unlock();

  // get state variables
  std::array<double, 7> coriolis_array = model_->coriolis(robot_state);
  std::array<double, 42> jacobian_array =
      model_->zeroJacobian(franka::Frame::kEndEffector, robot_state);

  // convert to Eigen
  Eigen::Map<Vector7d> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Vector7d q = Eigen::Map<const Vector7d>(robot_state.q.data());
  Vector7d dq = Eigen::Map<const Vector7d>(robot_state.dq.data());
  Vector7d tau_J_d = Eigen::Map<const Vector7d>(robot_state.tau_J_d.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.rotation());

  // compute error to desired pose
  // position error
  Eigen::Matrix<double, 6, 1> error;
  error.head(3) << position - position_d;

  // orientation error
  if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  // "difference" quaternion
  Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(),
      error_quaternion.z();
  // Transform to base frame
  error.tail(3) << -transform.rotation() * error.tail(3);

  // compute control
  Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7);

  // pseudoinverse for nullspace handling
  Eigen::MatrixXd jacobian_transpose_pinv;
  pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

  // Cartesian PD control with explicit damping
  tau_task << jacobian.transpose() * (-K_p * error - K_d * (jacobian * dq));

  // Nullspace PD control with per-joint gains
  tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
                    jacobian.transpose() * jacobian_transpose_pinv) *
                       (Kq.asDiagonal() * (q_nullspace_d - q) -
                        Kqd.asDiagonal() * dq);

  // Desired torque
  tau_d << tau_task + tau_nullspace + coriolis;

  franka::Torques torques = VectorToArray<7>(tau_d);
  torques.motion_finished = motion_finished_;
  return torques;
}

void PolymetisImpedance::setControl(const Eigen::Vector3d &position,
                                    const Eigen::Vector4d &orientation,
                                    const Vector7d &q_nullspace) {
  std::lock_guard<std::mutex> lock(mux_);
  position_d_target_ = position;
  // orientation as [x, y, z, w]
  orientation_d_target_ =
      Eigen::Quaterniond(orientation[3], orientation[0], orientation[1],
                         orientation[2]);
  q_nullspace_d_target_ = q_nullspace;
}

void PolymetisImpedance::setImpedance(
    const Eigen::Matrix<double, 6, 6> &impedance) {
  std::lock_guard<std::mutex> lock(mux_);
  K_p_target_ = impedance;
}

void PolymetisImpedance::setDamping(
    const Eigen::Matrix<double, 6, 6> &damping) {
  std::lock_guard<std::mutex> lock(mux_);
  K_d_target_ = damping;
}

void PolymetisImpedance::setNullspaceStiffness(
    const Vector7d &nullspace_stiffness) {
  std::lock_guard<std::mutex> lock(mux_);
  Kq_target_ = nullspace_stiffness;
}

void PolymetisImpedance::setNullspaceDamping(
    const Vector7d &nullspace_damping) {
  std::lock_guard<std::mutex> lock(mux_);
  Kqd_target_ = nullspace_damping;
}

void PolymetisImpedance::setFilter(const double filter_coeff) {
  std::lock_guard<std::mutex> lock(mux_);
  filter_coeff_ = filter_coeff;
}

void PolymetisImpedance::_updateFilter() {
  // EMA filter on all target values — identical to CartesianImpedance
  K_p_ = ema_filter(K_p_target_, K_p_, filter_coeff_);
  K_d_ = ema_filter(K_d_target_, K_d_, filter_coeff_);
  Kq_ = ema_filter(Kq_target_, Kq_, filter_coeff_);
  Kqd_ = ema_filter(Kqd_target_, Kqd_, filter_coeff_);
  position_d_ = ema_filter(position_d_target_, position_d_, filter_coeff_);
  orientation_d_ = orientation_d_.slerp(filter_coeff_, orientation_d_target_);
  q_nullspace_d_ =
      ema_filter(q_nullspace_d_target_, q_nullspace_d_, filter_coeff_);
}

void PolymetisImpedance::start(const franka::RobotState &robot_state,
                               std::shared_ptr<franka::Model> model) {
  model_ = model;
  motion_finished_ = false;
  Eigen::Affine3d transform(
      Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  position_d_ = transform.translation();
  position_d_target_ = position_d_;
  orientation_d_ = Eigen::Quaterniond(transform.rotation());
  orientation_d_target_ = orientation_d_;
  q_nullspace_d_ = Eigen::Map<const Vector7d>(robot_state.q.data());
  q_nullspace_d_target_ = q_nullspace_d_;
}

void PolymetisImpedance::stop(const franka::RobotState &robot_state,
                              std::shared_ptr<franka::Model> model) {
  motion_finished_ = true;
}

bool PolymetisImpedance::isRunning() { return !motion_finished_; }

const std::string PolymetisImpedance::name() { return "PolymetisImpedance"; }
