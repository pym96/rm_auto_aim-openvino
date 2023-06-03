#include "PredictorEKF.hpp"

// 射击延时。单位 s
double shoot_delay_time = 0.1;
double pre_time = 1;//预测时延，单位s。


Modules::PredictorEKF::PredictorEKF()
{
    cv::FileStorage fin{PROJECT_DIR "/Configs/camera/camera.yaml", cv::FileStorage::READ};

    fin["cameraMatrix"] >> F_MAT;  //相机内参
    fin["distCoeffs"] >> C_MAT;    //相机畸变
    fin["R_C2G"] >> R_C2G_MAT;     //相机坐标系 到 云台坐标系 的旋转矩阵

    // p_g = R_C2G * p_c

    std::cout << "内参=" << F_MAT << ",\n 畸变=" << C_MAT << std::endl;

    cv::cv2eigen(R_C2G_MAT, R_C2G);
    cv::cv2eigen(F_MAT, F);
    cv::cv2eigen(C_MAT, C);
    // std::cout << C<< std::endl;
    // std::cout<< F << std::endl;

    double small_half_x, small_half_y;
    double big_half_x, big_half_y;

    fin["armour"]["small_half_x"] >> small_half_x;
    fin["armour"]["small_half_y"] >> small_half_y;
    fin["armour"]["big_half_x"] >> big_half_x;
    fin["armour"]["big_half_y"] >> big_half_y;

        // 空气阻力系数
    fin["k"] >> k;
    // 静态角度补偿
    fin["static_yaw"] >> static_yaw;
    fin["static_pitch"] >> static_pitch;
    /*
    - point 0: [-squareLength / 2, squareLength / 2, 0]
    - point 1: [ squareLength / 2, squareLength / 2, 0]
    - point 2: [ squareLength / 2, -squareLength / 2, 0]
    - point 3: [-squareLength / 2, -squareLength / 2, 0]
    */
    small_obj = std::vector<cv::Point3d>{
        {-small_half_x, -small_half_y, 0},  //left top
        {-small_half_x, small_half_y, 0},  //left bottom
        {small_half_x, small_half_y, 0},    //right bottom
        {small_half_x, -small_half_y, 0}};   //right top
    big_obj = std::vector<cv::Point3d>{
        {-big_half_x, -big_half_y, 0},  //left top
        {-big_half_x, big_half_y, 0},  //left bottom
        {big_half_x, big_half_y, 0},    //right bottom
        {big_half_x, -big_half_y, 0}};   //right top

    fin.release();

    cv::Mat Q_mat, R_mat, P_mat;

    // fin["EKF"]["Q"] >> Q_mat;
    fin["EKF"]["R"] >> R_mat;
    // fin["EKF"]["P"] >> P_mat;

    MatrixXX Q = MatrixXX::Identity();
    // Q(0, 0) = 0.01;
    // Q(1, 1) = 0.01;
    // Q(2, 2) = 0.01;
    for(int i = 0; i < 6; i++)
    {
        Q(i, i) = 100;
    }

    MatrixZZ R = MatrixZZ::Identity();

    // MatrixXX P;

    // cv::cv2eigen(Q_mat, Q);
    // cv::cv2eigen(R_mat, R);
    // cv::cv2eigen(P_mat, P);

    ekf = AdaptiveEKF{Q, R};
}

bool Modules::PredictorEKF::predict(
    Modules::Detection_pack & detection_pack, const Devices::ReceiveData & receive_data,
    Devices::SendData & send_data, cv::Mat & showimg, Robot::Color color)
{
    double timestamp = detection_pack.timestamp;
    auto & img       = detection_pack.img;
    auto & armours   = detection_pack.armors;

    if (armours.empty()) {  //.size() == 0, 会报错
        send_data.send_pitch = 0;
        send_data.send_yaw = 0;
        send_data.goal = 0;

        loss_frame++;
        return false;
    }



    // 选择策略
    // auto & armours = detection_pack.armours;
    //对装甲板进行排序, 优先大装甲板, 其次选最大的
    std::sort(
        armours.begin(), armours.end(), [color](const ArmorObject & a1, const ArmorObject & a2) {
            // 不是识别的颜色
            if(a1.color != (int)color)
            {
                return false;
            }

            if(a2.color != (int)color)
            {
                return true;
            }

            std::vector<cv::Point2f> tmp1(a1.apex, a1.apex + APEX_NUM);
            double area1 = cv::boundingRect(tmp1).area();

            std::vector<cv::Point2f> tmp2(a2.apex, a2.apex + APEX_NUM);
            double area2 = cv::boundingRect(tmp2).area();


            return area1 > area2;
        });

    auto select_armour = armours.front();
    if(select_armour.color != (int)color )
    {
        return false;
    }

    // 根据 传来的pitch角度构造 旋转矩阵
    double pitch_radian = receive_data.pitch / 180. * M_PI;
    // fmt::print("[read] pitch_angle={}, yaw_angle={},shoot_speed={}\n", receive_data.pitch, receive_data.yaw,receive_data.shoot_speed );

    Eigen::Matrix3d R_G2W = PredictorEKF::get_R_G2W(receive_data);

    // 得到3个坐标系下的坐标,
    Eigen::Vector3d camera_points = get_camera_points(select_armour);
    Eigen::Vector3d gimbal_points = R_C2G * camera_points;
    Eigen::Vector3d world_points  = R_G2W * gimbal_points;


    if(loss_frame > 10)
    {
        ekf.reset(world_points);    
    }
    loss_frame = 0;
    // 相机坐标系，
    // x轴向右，y轴向下，z轴延相机向前方
    fmt::print(
        "[{:<6}]: {:.3f},{:.3f},{:.3f}\n", camera_fmt, camera_points(0, 0), camera_points(1, 0),
        camera_points(2, 0));
    //x轴延枪管向前，y轴向左，z轴向上
    fmt::print(
        "[{:<6}]: {:.3f},{:.3f},{:.3f}\n", gimbal_fmt, gimbal_points(0, 0), gimbal_points(1, 0),
        gimbal_points(2, 0));
    //xy平面平行于地面，z轴垂直地面
    //x轴延枪管向前，y轴向左，z轴向上
    fmt::print(
        "[{:<6}]: {:.3f},{:.3f},{:.3f}\n", world_fmt, world_points(0, 0), world_points(1, 0),
        world_points(2, 0));


    // show posistion, 在图片上画 三个坐标系的坐标
    std::string camera_position_fmt = fmt::format("[camera]: x={:.3f},y={:.3f},z={:.3f}", camera_points(0, 0), camera_points(1, 0),
        camera_points(2, 0));
    std::string gimbal_positino_fmt = fmt::format("[gimbal]: x={:.3f}, y={:.3f}, z={:.3f}", gimbal_points(0, 0),
        gimbal_points(1, 0), gimbal_points(2, 0));
    std::string world_position_fmt = fmt::format("[world]: x={:.3f},y={:.3f},z={:.3f}", world_points(0, 0), world_points(1, 0),
        world_points(2, 0));

    cv::putText(showimg,world_position_fmt, select_armour.apex[1] - cv::Point2f(0, -30), 1, cv::FONT_HERSHEY_PLAIN, cv::Scalar(0, 0, 255));
    cv::putText(showimg,gimbal_positino_fmt, select_armour.apex[1] - cv::Point2f(0, -60), 1, cv::FONT_HERSHEY_PLAIN, cv::Scalar(0, 0, 255));
    cv::putText(showimg,camera_position_fmt, select_armour.apex[1] - cv::Point2f(0, -90), 1, cv::FONT_HERSHEY_PLAIN, cv::Scalar(0, 255, 255));
    // ----------------- 卡尔曼滤波的使用 ------------------------

    // 更新时间
    predictfunc.delta_t = timestamp - last_time;
    last_time           = timestamp;
    // fmt::print("delay_time = {}s\n", predictfunc.delta_t);

    // ekf滤波出来，三维坐标和速度

    ekf.predict(predictfunc);
    // x,y,z ,v_x, v_y, v_z
    VectorX smooth_status = ekf.update(measure, world_points);
    
    Eigen::Vector3d predict_world_points = smooth_status.topRows<3>();
    predict_world_points(0, 0) += smooth_status(3, 0) * pre_time;
    predict_world_points(1, 0) += smooth_status(4, 0) * pre_time;
    predict_world_points(2, 0) += smooth_status(5, 0) * pre_time;
    // ----------------------------------------------------------
    
    std::string world__EKF_position_fmt = fmt::format("[world_EKF]: x={:.3f},y={:.3f},z={:.3f}", smooth_status(0, 0), smooth_status(1, 0),
        smooth_status(2, 0));
    std::string world__predict_position_fmt = fmt::format("[world_predict]: x={:.3f},y={:.3f},z={:.3f}", predict_world_points(0, 0), predict_world_points(1, 0),
        predict_world_points(2, 0));
    cv::putText(showimg,world__EKF_position_fmt, select_armour.apex[1] - cv::Point2f(0, -120), 1, cv::FONT_HERSHEY_PLAIN, cv::Scalar(0, 255, 255));
    cv::putText(showimg,world__predict_position_fmt, select_armour.apex[1] - cv::Point2f(0, -150), 1, cv::FONT_HERSHEY_PLAIN, cv::Scalar(0, 255, 255));
    float yaw_predict_slove = std::atan2(predict_world_points(1,0),predict_world_points(0, 0));
            // -std::atan2(gimbal_points(1, 0), gimbal_points(0, 0));  // 向右为正
    //float yaw_world = std::atan2(smooth_status(1,0),smooth_status(0, 0));
    float pitch_predict_slove = std::atan2(predict_world_points(2,0),predict_world_points(0, 0));
    //float pitch_solve = std::atan2(gimbal_points(2, 0), gimbal_points(0, 0));

    // 解算弹道模型
    if (solve_ballistic_model(world_points, receive_data, pitch_predict_slove)) {
        fmt::print(
            "最终迭代picth_k={:.3f}度, shoot={}\n", pitch_predict_slove / M_PI * 180.,
            receive_data.shoot_speed);  // 弧度 转 度

        fmt::print("[send]: yaw={:.3f}, pitch={:.3f}\n", yaw_predict_slove / M_PI * 180., pitch_predict_slove / M_PI * 180.);

        send_data.send_pitch = pitch_predict_slove / M_PI * 180. + static_pitch;
        send_data.send_yaw   = yaw_predict_slove/ M_PI * 180. + static_yaw;
        send_data.goal       = 1;

    } else {
        fmt::print(
            fg(fmt::color::red) | fmt::emphasis::bold, "最终迭代picth_k={:.3f}度, shoot={}\n",
            pitch_predict_slove / M_PI * 180., receive_data.shoot_speed);

        send_data.goal       = 0;

        return false;
    }

    return true;
}

Eigen::Vector3d Modules::PredictorEKF::get_camera_points(const ArmorObject& armor)
{
    cv::Mat tvec, rvec;
    std::vector<cv::Point2f> armor_points(armor.apex, armor.apex + APEX_NUM);


    if ( BIG_ARMOR_CLASSES.find(armor.cls) != BIG_ARMOR_CLASSES.end() ) {
        cv::solvePnP(big_obj, armor_points, F_MAT, C_MAT, rvec, tvec);
        fmt::print("big armour\n");
    } else {
        cv::solvePnP(small_obj, armor_points, F_MAT, C_MAT, rvec, tvec);
        fmt::print("small armour\n");
    }

    Eigen::Vector3d camera_points;

    cv::cv2eigen(tvec, camera_points);

    // camera_points(1, 0) =   camera_points(1, 0) + 0.07;

    return camera_points;
}

// 解算弹道模型
bool Modules::PredictorEKF::solve_ballistic_model(
    const Eigen::Vector3d & world_points, const Devices::ReceiveData & receive_data,
    float & pitch_res)
{
    // 根据弹道模型求出 pitch角度 和 飞行时间
    // double T_k;  // 飞行时间
    double f_tk, f_tk_;
    double h_k, h_r;
    double e_k;

    // 竖直- Z轴
    double dist_vertical = world_points(2, 0);
    double vertical_tmp  = dist_vertical;

    // 水平- X轴+Y轴
    double dist_horizonal = std::sqrt(
        world_points(0, 0) * world_points(0, 0) + world_points(1, 0) * world_points(1, 0));

    double pitch_0     = std::atan(dist_vertical / dist_horizonal);
    double pitch_solve = pitch_0;

    for (int i = 0; i < max_iter; i++) {
        double x       = 0.0;
        double y       = 0.0;
        double p       = std::tan(pitch_solve);
        double v       = receive_data.shoot_speed;
        double u       = v / std::sqrt(1 + pow(p, 2));
        double delta_x = dist_horizonal / R_K_iter;

        for (int j = 0; j < R_K_iter; j++) {
            double k1_u     = -k * u * sqrt(1 + pow(p, 2));
            double k1_p     = -g / pow(u, 2);
            double k1_u_sum = u + k1_u * (delta_x / 2);
            double k1_p_sum = p + k1_p * (delta_x / 2);

            double k2_u     = -k * k1_u_sum * sqrt(1 + pow(k1_p_sum, 2));
            double k2_p     = -g / pow(k1_u_sum, 2);
            double k2_u_sum = u + k2_u * (delta_x / 2);
            double k2_p_sum = p + k2_p * (delta_x / 2);

            double k3_u     = -k * k2_u_sum * sqrt(1 + pow(k2_p_sum, 2));
            double k3_p     = -g / pow(k2_u_sum, 2);
            double k3_u_sum = u + k2_u * (delta_x / 2);
            double k3_p_sum = p + k2_p * (delta_x / 2);

            double k4_u = -k * k3_u_sum * sqrt(1 + pow(k3_p_sum, 2));
            double k4_p = -g / pow(k3_u_sum, 2);

            u += (delta_x / 6) * (k1_u + k2_u + k3_u + k4_u);
            p += (delta_x / 6) * (k1_p + k2_p + k3_p + k4_p);

            x += delta_x;
            y += p * delta_x;
        }
        double error = dist_vertical - y;
        if (fabs(error) < stop_error) {
            break;
        } else {
            vertical_tmp += error;
            pitch_solve = atan(vertical_tmp / dist_horizonal);  // 弧度
        }
    }

    if (std::fabs(pitch_solve) > M_PI) {
        return false;
    }

    pitch_res = pitch_solve;

    return true;
}