#ifndef _ROBOT_HPP_
#define _ROBOT_HPP_

#include <Eigen/Dense>
// #include <opencv2/core/types.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <fmt/color.h>
#include <sys/types.h>


#include "../devices/serial/serial.hpp"
#include "../modules/detect_armour/detect.hpp"

namespace Robot
{
enum class RobotType {
    Hero,        //英雄
    Engineer,    //工程
    Infantry_3,  //步兵3号
    Infantry_4,  //步兵4号
    Infantry_5,  //步兵5号
    Sentry,      //哨兵
};




enum class ArmorType { Small, Big };

enum class Color  //敌方颜色
{ BLUE = 0,
  RED  = 1,
  NONE = 2,
  };

enum class ShootLevel { Level1, Level2, Level3 };

inline bool drawArmour(const ArmorObject & armor, cv::Mat & drawImg, const Color & color)
{
    cv::Scalar paint;
    if (color == Color::RED) {
        paint = cv::Scalar(0, 0, 255);
    } else {
        paint = cv::Scalar(255, 0, 0);
    }
    auto fmt_str = fmt::format("Class: {}, Color: {}, Conf: {:.3f}", ARMOR_CLASSES.at(armor.cls), ARMOR_COLORS.at(armor.color), armor.prob);

    cv::putText(drawImg, fmt_str, armor.apex[0] - cv::Point2f(0, 10),cv::FONT_HERSHEY_PLAIN , 1.0, cv::Scalar(0, 255, 0));
    cv::line(drawImg, armor.apex[0], armor.apex[1], paint, 2);
    cv::line(drawImg, armor.apex[1], armor.apex[2], paint, 2);
    cv::line(drawImg, armor.apex[2], armor.apex[3], paint, 2);
    cv::line(drawImg, armor.apex[3], armor.apex[0], paint, 2);

    return true;
}

inline bool drawArmours(const std::vector<ArmorObject> & armours, cv::Mat & drawImg, const Color & color)
{
    for (auto armour : armours) {
        drawArmour(armour, drawImg, color);
    }
    return true;
}


inline bool drawFPS(
    cv::Mat & image, double fps, const std::string & model_name = "",
    const cv::Point & point = cv::Point2f(0, 40), int fontsize = 1,
    const cv::Scalar & color = cv::Scalar(255, 255, 255))
{
    auto str = fmt::format("{:<8}FPS={:.1f}", model_name, fps);
    cv::putText(image, str, point, cv::FONT_HERSHEY_SIMPLEX, fontsize, color);
    return true;
}

inline bool drawSerial(
    cv::Mat & image, Devices::ReceiveData & data, const cv::Point2f & point = cv::Point2f(0, 40),
    int fontsize = 1, const cv::Scalar & color = cv::Scalar(255, 255, 255))
{
    std::string s_str = fmt::format("shoot={:2.2f}m/s", data.shoot_speed);
    std::string ea_str = fmt::format("yaw={:3.2f}C, pitch={:3.2f}C", data.yaw, data.pitch);
    cv::putText(image, ea_str, point, fontsize, cv::FONT_HERSHEY_PLAIN, color);
    cv::putText(image, s_str, point + cv::Point2f(0, 15), fontsize, cv::FONT_HERSHEY_PLAIN, color);
    return true;
}

inline bool drawSend(
    cv::Mat & image, Devices::SendData & data, const cv::Point2f & point = cv::Point2f(0, 40),
    int fontsize = 1, const cv::Scalar & color = cv::Scalar(255, 255, 255))
{
    std::string ea_str = fmt::format("yaw={:3.2f}C, pitch={:3.2f}C", data.send_yaw, data.send_pitch);
    cv::putText(image, ea_str, point, fontsize, cv::FONT_HERSHEY_PLAIN, color);
    return true;
}



}  // namespace Robot

#endif /*_ROBOT_HPP_*/