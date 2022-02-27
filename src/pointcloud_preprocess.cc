#include "pointcloud_preprocess.h"

#include <glog/logging.h>
#include <execution>

namespace faster_lio {

void PointCloudPreprocess::Set(LidarType lid_type, double bld, int pfilt_num) {
    lidar_type_ = lid_type;
    blind_ = bld;
    point_filter_num_ = pfilt_num;
}

void PointCloudPreprocess::Process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudType::Ptr &pcl_out) {
    AviaHandler(msg);
    *pcl_out = cloud_out_;
}

void PointCloudPreprocess::Process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudType::Ptr &pcl_out) {
    switch (lidar_type_) {
        case LidarType::OUST64:
            Oust64Handler(msg);
            break;

        case LidarType::VELO32:
            // 检查点云的点是否有时间属性，没有则计算相对时间，然后进行简单点云滤除
            VelodyneHandler(msg);
            break;

        default:
            LOG(ERROR) << "Error LiDAR Type";
            break;
    }
    // 预处理后的点云输出
    *pcl_out = cloud_out_;
}

void PointCloudPreprocess::AviaHandler(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
    cloud_out_.clear();
    cloud_full_.clear();
    int plsize = msg->point_num;

    cloud_out_.reserve(plsize);
    cloud_full_.resize(plsize);

    std::vector<bool> is_valid_pt(plsize, false);
    std::vector<uint> index(plsize - 1);
    for (uint i = 0; i < plsize - 1; ++i) {
        index[i] = i + 1;  // 从1开始
    }

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const uint &i) {
        if ((msg->points[i].line < num_scans_) &&
            ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00)) {
            if (i % point_filter_num_ == 0) {
                cloud_full_[i].x = msg->points[i].x;
                cloud_full_[i].y = msg->points[i].y;
                cloud_full_[i].z = msg->points[i].z;
                cloud_full_[i].intensity = msg->points[i].reflectivity;
                cloud_full_[i].curvature =
                    msg->points[i].offset_time /
                    float(1000000);  // use curvature as time of each laser points, curvature unit: ms

                if ((abs(cloud_full_[i].x - cloud_full_[i - 1].x) > 1e-7) ||
                    (abs(cloud_full_[i].y - cloud_full_[i - 1].y) > 1e-7) ||
                    (abs(cloud_full_[i].z - cloud_full_[i - 1].z) > 1e-7) &&
                        (cloud_full_[i].x * cloud_full_[i].x + cloud_full_[i].y * cloud_full_[i].y +
                             cloud_full_[i].z * cloud_full_[i].z >
                         (blind_ * blind_))) {
                    is_valid_pt[i] = true;
                }
            }
        }
    });

    for (uint i = 1; i < plsize; i++) {
        if (is_valid_pt[i]) {
            cloud_out_.points.push_back(cloud_full_[i]);
        }
    }
}

void PointCloudPreprocess::Oust64Handler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    cloud_out_.clear();
    cloud_full_.clear();
    pcl::PointCloud<ouster_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.size();
    cloud_out_.reserve(plsize);

    for (int i = 0; i < pl_orig.points.size(); i++) {
        if (i % point_filter_num_ != 0) continue;

        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                       pl_orig.points[i].z * pl_orig.points[i].z;

        if (range < (blind_ * blind_)) continue;

        Eigen::Vector3d pt_vec;
        PointType added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.curvature = pl_orig.points[i].t / 1e6;  // curvature unit: ms

        cloud_out_.points.push_back(added_pt);
    }
}

/**
 * @brief 检查点云的点是否有时间属性，没有则计算相对时间，然后进行简单点云滤除
 * @param msg
 */
void PointCloudPreprocess::VelodyneHandler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    cloud_out_.clear();
    cloud_full_.clear();

    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    cloud_out_.reserve(plsize);

    /*** These variables only works when no point timestamps given ***/
    /**  下面的参数只有在点云中的每个点没有时间属性的时候才会用到             **/
    double omega_l = 3.61;  // scan angular velocity
    std::vector<bool> is_first(num_scans_, true);
    std::vector<double> yaw_fp(num_scans_, 0.0);    // yaw of first scan point
    std::vector<float> yaw_last(num_scans_, 0.0);   // yaw of last scan point
    std::vector<float> time_last(num_scans_, 0.0);  // last offset time
    /*****************************************************************/

    // 检查点云的点是否有时间属性
    if (pl_orig.points[plsize - 1].time > 0) {
        given_offset_time_ = true;
    } else {
        // 如果没有时间属性，那么这里会计算扫描一圈的起始、结束角度
        given_offset_time_ = false;
        // 计算第0个点的yaw角（度）
        double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
        double yaw_end = yaw_first;
        // 取第0个点的ring号
        int layer_first = pl_orig.points[0].ring;
        // 从最后一个点向前遍历
        for (uint i = plsize - 1; i > 0; i--) {
            // 如果与第0个点是同一线束，就计算yaw_end
            if (pl_orig.points[i].ring == layer_first) {
                yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
                break;
            }
        }
    }

    // 遍历每一个点
    for (int i = 0; i < plsize; i++) {
        PointType added_pt;

        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * time_scale_;  // curvature unit: ms   使用curvature储存该点时间

        // 如果点云的点没有时间属性，下面则计算时间
        if (!given_offset_time_) {
            // 取线束号为layer
            int layer = pl_orig.points[i].ring;
            // 计算该点角度
            double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

            // 如果是某条线束的第0个点
            if (is_first[layer]) {
                yaw_fp[layer] = yaw_angle;  // 设置起始角度
                is_first[layer] = false;    // 标记，第0个点
                added_pt.curvature = 0.0;   // 第0个点的时间偏移为0
                yaw_last[layer] = yaw_angle;
                time_last[layer] = added_pt.curvature;
                continue;
            }

            // compute offset time
            // curvature: 相对时间，以秒为单位
            if (yaw_angle <= yaw_fp[layer]) {   // 如果当前这个点的角度 > 该线束起始点的角度
                added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l; // 计算相对时间， 这里除以3.61, 是因为10Hz雷达，1s转过3600度，
            } else {
                added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
            }

            // 如果当前这个点的相对时间 < 上一个点的时间，表明已经越过一圈了
            if (added_pt.curvature < time_last[layer]) added_pt.curvature += 360.0 / omega_l;

            // 记录该线束最终扫描结束角度
            yaw_last[layer] = yaw_angle;
            // 保存时间，作为上一个点的时间
            time_last[layer] = added_pt.curvature;
        }

        // 简单粗暴按间隔滤除一些点，同时滤除距离较近的点
        if (i % point_filter_num_ == 0) {
            if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind_ * blind_)) {
                cloud_out_.points.push_back(added_pt);
            }
        }
    }
}

}  // namespace faster_lio
