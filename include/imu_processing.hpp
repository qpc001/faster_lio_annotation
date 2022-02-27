#ifndef FASTER_LIO_IMU_PROCESSING_H
#define FASTER_LIO_IMU_PROCESSING_H

#include <glog/logging.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <cmath>
#include <deque>
#include <fstream>

#include "common_lib.h"
#include "so3_math.h"
#include "use-ikfom.hpp"
#include "utils.h"

namespace faster_lio {

constexpr int MAX_INI_COUNT = 20;

bool time_list(const PointType &x, const PointType &y) { return (x.curvature < y.curvature); };

/// IMU Process and undistortion
class ImuProcess {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuProcess();
    ~ImuProcess();

    void Reset();
    void SetExtrinsic(const common::V3D &transl, const common::M3D &rot);
    void SetGyrCov(const common::V3D &scaler);
    void SetAccCov(const common::V3D &scaler);
    void SetGyrBiasCov(const common::V3D &b_g);
    void SetAccBiasCov(const common::V3D &b_a);
    void Process(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                 PointCloudType::Ptr pcl_un_);

    std::ofstream fout_imu_;
    Eigen::Matrix<double, 12, 12> Q_;
    common::V3D cov_acc_;
    common::V3D cov_gyr_;
    common::V3D cov_acc_scale_;
    common::V3D cov_gyr_scale_;
    common::V3D cov_bias_gyr_;
    common::V3D cov_bias_acc_;

   private:
    void IMUInit(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
    void UndistortPcl(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                      PointCloudType &pcl_out);

    PointCloudType::Ptr cur_pcl_un_;
    sensor_msgs::ImuConstPtr last_imu_;
    std::deque<sensor_msgs::ImuConstPtr> v_imu_;
    std::vector<common::Pose6D> IMUpose_;
    std::vector<common::M3D> v_rot_pcl_;
    common::M3D Lidar_R_wrt_IMU_;
    common::V3D Lidar_T_wrt_IMU_;
    common::V3D mean_acc_;
    common::V3D mean_gyr_;
    common::V3D angvel_last_;   ///< 没有实际作用，可以改为局部变量
    common::V3D acc_s_last_;
    double last_lidar_end_time_;
    int init_iter_num_ = 1;
    bool b_first_frame_ = true;
    bool imu_need_init_ = true;
};

ImuProcess::ImuProcess() : b_first_frame_(true), imu_need_init_(true) {
    init_iter_num_ = 1;
    Q_ = process_noise_cov();
    cov_acc_ = common::V3D(0.1, 0.1, 0.1);
    cov_gyr_ = common::V3D(0.1, 0.1, 0.1);
    cov_bias_gyr_ = common::V3D(0.0001, 0.0001, 0.0001);
    cov_bias_acc_ = common::V3D(0.0001, 0.0001, 0.0001);
    mean_acc_ = common::V3D(0, 0, -1.0);
    mean_gyr_ = common::V3D(0, 0, 0);
    angvel_last_ = common::Zero3d;
    Lidar_T_wrt_IMU_ = common::Zero3d;
    Lidar_R_wrt_IMU_ = common::Eye3d;
    last_imu_.reset(new sensor_msgs::Imu());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() {
    mean_acc_ = common::V3D(0, 0, -1.0);
    mean_gyr_ = common::V3D(0, 0, 0);
    angvel_last_ = common::Zero3d;
    imu_need_init_ = true;
    init_iter_num_ = 1;
    v_imu_.clear();
    IMUpose_.clear();
    last_imu_.reset(new sensor_msgs::Imu());
    cur_pcl_un_.reset(new PointCloudType());
}

void ImuProcess::SetExtrinsic(const common::V3D &transl, const common::M3D &rot) {
    Lidar_T_wrt_IMU_ = transl;
    Lidar_R_wrt_IMU_ = rot;
}

void ImuProcess::SetGyrCov(const common::V3D &scaler) { cov_gyr_scale_ = scaler; }

void ImuProcess::SetAccCov(const common::V3D &scaler) { cov_acc_scale_ = scaler; }

void ImuProcess::SetGyrBiasCov(const common::V3D &b_g) { cov_bias_gyr_ = b_g; }

void ImuProcess::SetAccBiasCov(const common::V3D &b_a) { cov_bias_acc_ = b_a; }

void ImuProcess::IMUInit(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                          int &N) {
    /** 1. initializing the gravity_, gyro bias, acc and gyro covariance
     ** 2. normalize the acceleration measurenments to unit gravity_ **/

    common::V3D cur_acc, cur_gyr;

    // 由于这个ImuProcess::IMUInit会迭代几次，所以这里又加了标志位，限制下面只运行1次
    if (b_first_frame_) {
        // 只reset一次
        Reset();
        N = 1;
        b_first_frame_ = false;
        const auto &imu_acc = meas.imu_.front()->linear_acceleration;
        const auto &gyr_acc = meas.imu_.front()->angular_velocity;
        mean_acc_ << imu_acc.x, imu_acc.y, imu_acc.z;
        mean_gyr_ << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    }
    // 遍历IMU数据，计算acc，gyro均值，协方差，这里的协方差使用了迭代方式计算
    for (const auto &imu : meas.imu_) {
        const auto &imu_acc = imu->linear_acceleration;
        const auto &gyr_acc = imu->angular_velocity;
        cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

        mean_acc_ += (cur_acc - mean_acc_) / N;
        mean_gyr_ += (cur_gyr - mean_gyr_) / N;

        cov_acc_ =
            cov_acc_ * (N - 1.0) / N + (cur_acc - mean_acc_).cwiseProduct(cur_acc - mean_acc_) * (N - 1.0) / (N * N);
        cov_gyr_ =
            cov_gyr_ * (N - 1.0) / N + (cur_gyr - mean_gyr_).cwiseProduct(cur_gyr - mean_gyr_) * (N - 1.0) / (N * N);

        N++;
    }
    // 下面是初始化kf_state的状态和协方差
    state_ikfom init_state = kf_state.get_x();
    init_state.grav = S2(-mean_acc_ / mean_acc_.norm() * common::G_m_s2);

    init_state.bg = mean_gyr_;
    init_state.offset_T_L_I = Lidar_T_wrt_IMU_;
    init_state.offset_R_L_I = Lidar_R_wrt_IMU_;
    kf_state.change_x(init_state);

    esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
    init_P.setIdentity();
    init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
    init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
    init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
    init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;
    init_P(21, 21) = init_P(22, 22) = 0.00001;
    kf_state.change_P(init_P);
    // 保存measures_中的最后一个imu，用于后面ImuProcess::UndistortPcl 去畸变
    last_imu_ = meas.imu_.back();
}

void ImuProcess::UndistortPcl(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                              PointCloudType &pcl_out) {
    /*** add the imu_ of the last frame-tail to the of current frame-head ***/
    auto v_imu = meas.imu_;
    // 取初始化阶段measures_中的最后一个imu，添加到最前面
    v_imu.push_front(last_imu_);
    const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
    const double &imu_end_time = v_imu.back()->header.stamp.toSec();
    const double &pcl_beg_time = meas.lidar_bag_time_;
    const double &pcl_end_time = meas.lidar_end_time_;

    /*** sort point clouds by offset time ***/
    // 取measures_中的雷达数据，根据每个点的时间进行排序(从小到大)
    pcl_out = *(meas.lidar_);
    sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);

    /*** Initialize IMU pose ***/
    // 取kf_的最新状态
    // 如果是刚刚完成初始化，那么取的就是初始化的状态
    // 如果是刚刚完成更新，那么取的就是更新之后的状态
    state_ikfom imu_state = kf_state.get_x();
    IMUpose_.clear();
    // 保存这个旧的状态到IMUpose_
    IMUpose_.push_back(common::set_pose6d(0.0, acc_s_last_, angvel_last_, imu_state.vel, imu_state.pos,
                                          imu_state.rot.toRotationMatrix()));

    /*** forward propagation at each imu_ point ***/
    // 所谓的forward propagation就是用IMU数据向前传播
    common::V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
    common::M3D R_imu;

    double dt = 0;

    // 遍历IMU数据，直到最后一个IMU数据的前一个数据
    input_ikfom in;
    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
        // 取当前遍历的IMU数据和下一个IMU数据
        auto &&head = *(it_imu);
        auto &&tail = *(it_imu + 1);
        // 如果下一个IMU数据比上一次去畸变的激光扫描结束时刻还早 （即tail这个IMU数据比当前帧扫描起始时刻还早，那么head这个数据需要放弃）
        // TODO: last_lidar_end_time_没有给初值阿大哥
        if (tail->header.stamp.toSec() < last_lidar_end_time_) {
            continue;
        }
        // 计算两帧IMU的平均角速度、加速度
        angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
            0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
            0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
        acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
            0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
            0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

        // note: 这里有个重力常量尺度变换，那么要求在初始化阶段，必须静止，因为mean_acc_是初始化阶段得到的
        acc_avr = acc_avr * common::G_m_s2 / mean_acc_.norm();  // - state_inout.ba;

        // 如果前面的IMU时间 < 上一次去畸变的激光扫描结束时刻(即当前帧扫描起始时间)
        if (head->header.stamp.toSec() < last_lidar_end_time_) {
            // 通常，每次去畸变的时候，只有第一帧IMU数据都会进来，即(last_imu_)这个数据
            // 需要积分的时间段为[last_lidar_end_time_, tail->header.stamp.toSec()]
            dt = tail->header.stamp.toSec() - last_lidar_end_time_;
        } else {
            // 否则，需要积分的时间段为[head, tail]
            dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
        }

        // 构造input_ikfom
        in.acc = acc_avr;
        in.gyro = angvel_avr;
        // 设置噪声矩阵Q
        Q_.block<3, 3>(0, 0).diagonal() = cov_gyr_;
        Q_.block<3, 3>(3, 3).diagonal() = cov_acc_;
        Q_.block<3, 3>(6, 6).diagonal() = cov_bias_gyr_;
        Q_.block<3, 3>(9, 9).diagonal() = cov_bias_acc_;
        // 输入给kf_,进行前向传播
        kf_state.predict(dt, Q_, in);

        /* save the poses at each IMU measurements */
        // 取推算后的状态
        imu_state = kf_state.get_x();
        // 更新角速度
        angvel_last_ = angvel_avr - imu_state.bg;
        // 更新加速度
        acc_s_last_ = imu_state.rot * (acc_avr - imu_state.ba);
        // 加速度去除重力影响
        for (int i = 0; i < 3; i++) {
            acc_s_last_[i] += imu_state.grav[i];
        }
        // 计算该状态时间戳相对于激光扫描起始的时间偏移
        double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
        // 取推算后的状态，保存到IMUpose_
        IMUpose_.emplace_back(common::set_pose6d(offs_t, acc_s_last_, angvel_last_, imu_state.vel, imu_state.pos,
                                                 imu_state.rot.toRotationMatrix()));
    }

    // 由于measures_中的IMU数据的时间范围是(xxx,激光扫描结束时刻之前)
    // 所以下面要用measures_中最后的imu数据再粗略往前推一点点（近似结果）
    /*** calculated the pos and attitude prediction at the frame-end ***/
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);
    // ROS_WARN("pcl_end_time > imu_end_time [%s]", (note == 1.0) ? "True" : "False");  // 貌似一直输出True
    kf_state.predict(dt, Q_, in);

    // 取推算后的状态(此时，状态的时间戳为激光扫描结束时刻)
    imu_state = kf_state.get_x();
    // 保存最后一个IMU，用于下一次去畸变的第一个IMU数据
    last_imu_ = meas.imu_.back();
    // 保存当前帧扫描结束时刻，即为下一帧扫描起始时刻
    last_lidar_end_time_ = pcl_end_time;

    /*** undistort each lidar point (backward propagation) ***/
    // 所谓的backward propagation就是用向前传播的状态将激光点
    if (pcl_out.points.empty()) {
        return;
    }
    // 取最后一个激光点（即激光扫描结束时刻）
    auto it_pcl = pcl_out.points.end() - 1;
    // 遍历刚刚前向传播的状态，从最后一个状态开始
    for (auto it_kp = IMUpose_.end() - 1; it_kp != IMUpose_.begin(); it_kp--) {
        auto head = it_kp - 1;  // 当前遍历状态的前一个状态
        auto tail = it_kp;  // 当前遍历状态
        R_imu = common::MatFromArray(head->rot);
        vel_imu = common::VecFromArray(head->vel);
        pos_imu = common::VecFromArray(head->pos);
        // 当前遍历状态的 acc 和 gyro
        acc_imu = common::VecFromArray(tail->acc);
        angvel_avr = common::VecFromArray(tail->gyr);

        // 遍历激光扫描 (如果 当前遍历状态的前一个状态时间戳 < 当前激光点的时间戳，就进入遍历)
        // offset_time: 该状态时间戳相对于激光扫描起始的时间偏移
        // curvature: 该激光点相对于激光扫描起始的时间偏移
        for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
            // 取该激光点相对于激光扫描起始的时间偏移
            dt = it_pcl->curvature / double(1000) - head->offset_time;

            /* Transform to the 'end' frame, using only the rotation
             * Note: Compensation direction is INVERSE of Frame's moving direction
             * So if we want to compensate a point at timestamp-i to the frame-e
             * p_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
            // 去畸变的过程是：
            // 1. 先利用前向转播的状态和IMU数据，推算得到该激光点对应时刻的载体Pose
            // 2. 利用(1)得到的Pose将该激光点投影到世界坐标系
            // 3. 利用前向传播得到激光扫描结束时刻的载体Pose(即imu_state)，将激光点从世界坐标系投影到激光扫描结束时刻的激光雷达坐标系

            // R_imu: 当前遍历状态的前一个状态的姿态
            // R_i: 从R_imu开始向前推算dt时刻后，得到的姿态(即激光点对应时刻的载体姿态)
            common::M3D R_i(R_imu * Exp(angvel_avr, dt));

            // P_i: 当前激光点在激光雷达坐标系的坐标
            common::V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
            // pos_imu: 当前遍历状态的前一个状态的位置
            // vel_imu: 当前遍历状态的前一个状态的速度
            // pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt: 当前激光点对应时刻的载体位置（世界坐标系）, 与 R_i 对应
            // imu_state.pos: 激光扫描结束时刻的位置（世界坐标系）
            // T_ei:
            common::V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);

            // imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I: 将当前激光点投影到IMU坐标系
            // p_compensate: 当前激光点投影到激光扫描结束时刻，在激光雷达坐标系的坐标
            common::V3D p_compensate =
                imu_state.offset_R_L_I.conjugate() *
                (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) -
                 imu_state.offset_T_L_I);  // not accurate!

            // save Undistorted points and their rotation
            it_pcl->x = p_compensate(0);
            it_pcl->y = p_compensate(1);
            it_pcl->z = p_compensate(2);

            // 如果是扫描起始点，表示点都已经遍历完了,这里的点可以按时序遍历是因为在预处理的时候根据点的时间进行了一次排序
            if (it_pcl == pcl_out.points.begin()) {
                break;
            }
        }
    }
}

void ImuProcess::Process(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                         PointCloudType::Ptr cur_pcl_un_) {
    if (meas.imu_.empty()) {
        return;
    }

    ROS_ASSERT(meas.lidar_ != nullptr);

    // 如果是第一帧测量，需要设置初始化状态
    if (imu_need_init_) {
        /// The very first lidar frame
        IMUInit(meas, kf_state, init_iter_num_);

        imu_need_init_ = true;

        // 保存初始化阶段measures_中的最后一个imu，用于后面ImuProcess::UndistortPcl 去畸变
        last_imu_ = meas.imu_.back();

        state_ikfom imu_state = kf_state.get_x();
        if (init_iter_num_ > MAX_INI_COUNT) {
            cov_acc_ *= pow(common::G_m_s2 / mean_acc_.norm(), 2);
            imu_need_init_ = false;

            cov_acc_ = cov_acc_scale_;
            cov_gyr_ = cov_gyr_scale_;
            LOG(INFO) << "IMU Initial Done";
            fout_imu_.open(common::DEBUG_FILE_DIR("imu_.txt"), std::ios::out);
        }

        return;
    }
    // 去畸变
    Timer::Evaluate([&, this]() { UndistortPcl(meas, kf_state, *cur_pcl_un_); }, "Undistort Pcl");
}
}  // namespace faster_lio

#endif
