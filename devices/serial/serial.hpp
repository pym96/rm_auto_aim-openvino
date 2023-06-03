#ifndef _SERIAL_HPP_
#define _SERIAL_HPP_

#include <fmt/core.h>
#include <opencv2/core/hal/interface.h>

#include <string>

#include <mutex>
#include <thread>

namespace Devices
{
const int BufferMaxSize = 50;

class DetectColor{

private:
   bool detect_red;

public:
    inline bool detectRed(){
        if(detect_red) return true;
        else return false;
    }

    void setColor(int color){
        if(color == 0){
            detect_red = false;
        }else{
            detect_red = true;
        }
    }
    
};

struct SendData
{
    float send_yaw;
    float send_pitch;
    std::uint8_t goal;
    std::uint8_t color;

    SendData()
    {
        send_pitch = send_pitch = 0;
        goal                    = 0;
        color                   = 0;
    };

    SendData(double yaw, double pitch)
    {
        send_yaw   = yaw;
        send_pitch = pitch;
        goal       = 1;
    }
};


struct ReceiveData
{
    float yaw;
    float pitch;
    float shoot_speed;
    int detect_color;  // 1: red , 0 : blue

    ReceiveData() = default;

    ReceiveData(float yaw, float pitch, float shoot_speed,int detect_color)
    : yaw(yaw), pitch(pitch), shoot_speed(shoot_speed),detect_color(detect_color)
    {
    }
};

union float_uchar {
    float f;
    uchar uchars[4];
};

class Serial
{
public:
    explicit Serial(const std::string &, std::mutex &);

    bool noArmour();

    bool openSerial();

    bool reOpen();

    bool closeSerial();

    bool isOpen();

    bool readSerial();

    bool sendData(SendData &);

    bool isRed();

    ReceiveData getData();

    Serial(Serial const &) = delete;
    Serial & operator=(Serial const &) = delete;

private:
    uchar send_buffer_[BufferMaxSize];
    uchar read_buffer_[BufferMaxSize];

    uchar frame_header, frame_tail;
    uchar loss;
    uchar color;
    int detect_color;

    float_uchar send_yaw, send_pitch;
    float_uchar shoot_speed, read_yaw, read_pitch;

    std::string name;
    std::mutex & serial_mutex;
    std::mutex data_mutex;

    float last_shoot_speed = 15;
    float updateShootSpeed(float);

    int fd;             //串口句柄
    fd_set fds;         //句柄集合
    struct timeval tv;  //时间
    DetectColor dk;
};

}  // namespace Devices

#endif /*_SERIAL_HPP_*/