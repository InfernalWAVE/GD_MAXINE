#ifndef KALMAN_FILTER_1D_H
#define KALMAN_FILTER_1D_H

class KalmanFilter1D {
 private:
  float Q_;          // Covariance of the process noise
  float xhat_;       // Current prediction
  float xhatminus_;  // Previous prediction
  float P_;          // Estimated accuracy of xhat_
  float Pminus_;     // Previous P_
  float K_;          // Kalman gain
  float R_;          // Covariance of the observation noise
  bool bFirstUse;

 public:
  KalmanFilter1D() { reset(); }

  KalmanFilter1D(float Q, float R) { reset(Q, R); }

  void reset() {
    R_ = 0.005f * 0.005f;
    Q_ = 1e-5f;
    xhat_ = 0.0f;
    xhatminus_ = 0.0f;
    P_ = 1;
    bFirstUse = true;
    Pminus_ = 0.0f;
    K_ = 0.0f;
  }

  void reset(float Q, float R) {
    reset();
    Q_ = Q;
    R_ = R;
  }

  float update(float val) {
    if (bFirstUse) {
      xhat_ = val;
      bFirstUse = false;
    }

    xhatminus_ = xhat_;
    Pminus_ = P_ + Q_;
    K_ = Pminus_ / (Pminus_ + R_);
    xhat_ = xhatminus_ + K_ * (val - xhatminus_);
    P_ = (1 - K_) * Pminus_;

    return xhat_;
  }
};

#endif // KALMAN_FILTER_1D_H