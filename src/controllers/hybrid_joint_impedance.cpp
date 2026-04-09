// HybridJointImpedance: joint-space PD with Cartesian-augmented gains.
// Matches the actual Polymetis HybridJointImpedanceControl architecture.

#include "controllers/hybrid_joint_impedance.h"

#include "panda.h"

// Polymetis franka_hardware.yaml defaults
// clang-format off
double hji_kx_data[36] = {750,  0,   0,  0,  0,  0,
                            0, 750,  0,  0,  0,  0,
                            0,   0, 750,  0,  0,  0,
                            0,   0,   0, 15,  0,  0,
                            0,   0,   0,  0, 15,  0,
                            0,   0,   0,  0,  0, 15};
double hji_kxd_data[36] = {37,  0,  0,  0,  0,  0,
                             0, 37,  0,  0,  0,  0,
                             0,  0, 37,  0,  0,  0,
                             0,  0,  0,  2,  0,  0,
                             0,  0,  0,  0,  2,  0,
                             0,  0,  0,  0,  0,  2};
// clang-format on

const Eigen::Matrix<double, 6, 6> HybridJointImpedance::kDefaultKx =
    Eigen::Matrix<double, 6, 6>(hji_kx_data);
const Eigen::Matrix<double, 6, 6> HybridJointImpedance::kDefaultKxd =
    Eigen::Matrix<double, 6, 6>(hji_kxd_data);
const Vector7d HybridJointImpedance::kDefaultKq =
    (Vector7d() << 40, 30, 50, 25, 35, 25, 10).finished();
const Vector7d HybridJointImpedance::kDefaultKqd =
    (Vector7d() << 4, 6, 5, 5, 3, 2, 1).finished();
const double HybridJointImpedance::kDefaultFilterCoeff = 1.0;

HybridJointImpedance::HybridJointImpedance(
    const Eigen::Matrix<double, 6, 6> &Kx,
    const Eigen::Matrix<double, 6, 6> &Kxd,
    const Vector7d &Kq, const Vector7d &Kqd,
    const double &filter_coeff) {
  Kx_ = Kx;
  Kx_target_ = Kx;
  Kxd_ = Kxd;
  Kxd_target_ = Kxd;
  Kq_ = Kq;
  Kq_target_ = Kq;
  Kqd_ = Kqd;
  Kqd_target_ = Kqd;
  filter_coeff_ = filter_coeff;
}

franka::Torques HybridJointImpedance::step(
    const franka::RobotState &robot_state, franka::Duration &duration) {
  Eigen::Matrix<double, 6, 6> Kx, Kxd;
  Vector7d Kq, Kqd, q_d;

  mux_.lock();
  _updateFilter();
  Kx = Kx_;
  Kxd = Kxd_;
  Kq = Kq_;
  Kqd = Kqd_;
  q_d = q_d_;
  mux_.unlock();

  // Get state
  std::array<double, 7> coriolis_array = model_->coriolis(robot_state);
  std::array<double, 42> jacobian_array =
      model_->zeroJacobian(franka::Frame::kEndEffector, robot_state);

  Eigen::Map<Vector7d> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> J(jacobian_array.data());
  Vector7d q = Eigen::Map<const Vector7d>(robot_state.q.data());
  Vector7d dq = Eigen::Map<const Vector7d>(robot_state.dq.data());

  // Effective joint-space gains: J^T * Kx * J + diag(Kq)
  Eigen::Matrix<double, 7, 7> JtKxJ = J.transpose() * Kx * J;
  Eigen::Matrix<double, 7, 7> JtKxdJ = J.transpose() * Kxd * J;
  Eigen::Matrix<double, 7, 7> Kp_eff = JtKxJ + Eigen::Matrix<double, 7, 7>(Kq.asDiagonal());
  Eigen::Matrix<double, 7, 7> Kd_eff = JtKxdJ + Eigen::Matrix<double, 7, 7>(Kqd.asDiagonal());

  // Joint-space PD
  Vector7d tau_d = Kp_eff * (q_d - q) - Kd_eff * dq + coriolis;

  franka::Torques torques = VectorToArray<7>(tau_d);
  torques.motion_finished = motion_finished_;
  return torques;
}

void HybridJointImpedance::setControl(const Vector7d &q_desired) {
  std::lock_guard<std::mutex> lock(mux_);
  q_d_target_ = q_desired;
}

void HybridJointImpedance::setKx(
    const Eigen::Matrix<double, 6, 6> &Kx) {
  std::lock_guard<std::mutex> lock(mux_);
  Kx_target_ = Kx;
}

void HybridJointImpedance::setKxd(
    const Eigen::Matrix<double, 6, 6> &Kxd) {
  std::lock_guard<std::mutex> lock(mux_);
  Kxd_target_ = Kxd;
}

void HybridJointImpedance::setKq(const Vector7d &Kq) {
  std::lock_guard<std::mutex> lock(mux_);
  Kq_target_ = Kq;
}

void HybridJointImpedance::setKqd(const Vector7d &Kqd) {
  std::lock_guard<std::mutex> lock(mux_);
  Kqd_target_ = Kqd;
}

void HybridJointImpedance::setFilter(const double filter_coeff) {
  std::lock_guard<std::mutex> lock(mux_);
  filter_coeff_ = filter_coeff;
}

void HybridJointImpedance::_updateFilter() {
  // EMA filter: ema_filter(current, target, alpha) = alpha*target + (1-alpha)*current
  Kx_ = ema_filter(Kx_, Kx_target_, filter_coeff_);
  Kxd_ = ema_filter(Kxd_, Kxd_target_, filter_coeff_);
  Kq_ = ema_filter(Kq_, Kq_target_, filter_coeff_);
  Kqd_ = ema_filter(Kqd_, Kqd_target_, filter_coeff_);
  q_d_ = ema_filter(q_d_, q_d_target_, filter_coeff_);
}

void HybridJointImpedance::start(const franka::RobotState &robot_state,
                                  std::shared_ptr<franka::Model> model) {
  model_ = model;
  motion_finished_ = false;
  q_d_ = Eigen::Map<const Vector7d>(robot_state.q.data());
  q_d_target_ = q_d_;
}

void HybridJointImpedance::stop(const franka::RobotState &robot_state,
                                 std::shared_ptr<franka::Model> model) {
  motion_finished_ = true;
}

bool HybridJointImpedance::isRunning() { return !motion_finished_; }

const std::string HybridJointImpedance::name() {
  return "HybridJointImpedance";
}
