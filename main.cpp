//自
#include "devices/camera/mv_camera.hpp"
#include "devices/serial/serial.hpp"

#include "modules/detect_armour/detect.hpp"
#include "modules/kalman_filter/PredictorEKF.hpp"

#include "utils/robot.hpp"
#include "utils/timer/timer.hpp"

#include "myThreads.h"

#include <string>

int main(int argc, char ** argv)
{
    // 注册信号量，处理 Ctrl + C 中断
    signal(SIGINT, signalHandler);

    // 控制量声明
    bool camera_start = false;
    std::mutex camera_mutex;
    std::mutex serial_mutex;

    // 确定敌方颜色
    Robot::Color color = Robot::Color::BLUE;
    // if (argc == 2) {
    //     if (argv[1][0] - '0' == 1) {  //1 红
    //         color = Robot::Color::RED;
    //     } else if (argv[1][0] - '0' == 0) {  //0 蓝
    //         color = Robot::Color::BLUE;
    //     }
    // }

    fmt::print("Init Modules\n");
    // 模块初始化
    utils::timer timer{"main", 10};
    Modules::Detector detector{PROJECT_DIR "/Configs/detect/armor-nano-poly-fp32-best-4-30.onnx"};
    Modules::PredictorEKF predictor{};
    Devices::Serial serial{"/dev/ttyACM0", serial_mutex};

    int frame = 1;  //主线程的帧数

    cv::Mat img;
    cv::Mat copyImg;
    double timestamp_ms;  //曝光时间, 单位us

    // 相机进程
    std::thread readSerialThread{readSerial_thread, std::ref(serial)};
    std::thread cameraThread{
        camera_thread, std::ref(main_loop_condition), std::ref(img), std::ref(camera_mutex),
        std::ref(timestamp_ms)};

    readSerialThread.detach();

    cameraThread.detach();

    //计时
    struct timespec tv_start;        //开始的时间戳
    struct timespec tv_end;          //结束的时间戳
    double a                = 0.99;  //线程运行时间的动量更新超参数
    double temp_time        = 0;     //temp
    double main_thread_time = 0;     //主线程的时间

    auto startTime = std::chrono::system_clock::now();
    serial.openSerial();

    
    for (; main_loop_condition; frame++) {
        //计时
        // timer.start(1);
        {  //上锁
            if (img.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            std::lock_guard<std::mutex> l(camera_mutex);
            copyImg = img.clone();
        }

        // timer.start(1);
        auto endTime = std::chrono::system_clock::now();
        auto wasteTime =
            std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() /
            1e6;  //单位 s  

        cv::flip(copyImg,copyImg,-1);

        cv::Mat showimg = copyImg;
        auto receive_data = serial.getData();
        // Devices::ReceiveData receive_data{0, 0, 14};

        if(receive_data.detect_color){
            color = Robot::Color::RED;
        
            fmt::print(bg(fmt::color::red),"Current detected color is {}\n",(serial.getData().detect_color));
        }else{
            color = Robot::Color::BLUE;
            fmt::print(bg(fmt::color::blue),"Current detected color is blue\n");
        }

        Modules::Detection_pack detection_pack{copyImg, wasteTime};
        // timer.end(1, "init detection_pack");

        timer.start(0);
        detector.detect(detection_pack);
        timer.end(0, "detect");

        Devices::SendData send_data{};
        predictor.predict(detection_pack, receive_data, send_data, showimg, color);

         std::thread sendSerialThread{sendSerial_thread, std::ref(serial), std::ref(send_data)};
         sendSerialThread.join();
        // // draw

        Robot::drawArmours(detection_pack.armors, showimg, color);
        // Robot::drawFPS(showimg, 1000. / main_thread_time, "Main", cv::Point2f(5, 80));
        Robot::drawSerial(showimg, receive_data, cv::Point2f(1000, 20));
        Robot::drawSend(showimg, send_data, cv::Point2f(1000, 80));

        // timer.start(0);
        cv::imshow("after_draw", showimg);
        cv::waitKey(1);  //waitKey(1) == waitKey(20)

        // using namespace std::chrono_literals;
        // std::this_thread::sleep_for(10ms);

        // timer.end(0, "imshow");
    }

    // fmt::print(fg(fmt::color::red), "end! wait for 5s\n");
    // std::this_thread::sleep_for(std::chrono::milliseconds(5000));

    return 0;
}