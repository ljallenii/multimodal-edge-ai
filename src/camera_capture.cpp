#include <iostream>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>

struct FramePacket
{
    cv::Mat frame;
    std::chrono::steady_clock::time_point capture_time;
    std::uint64_t frame_id;
};

class FrameBuffer
{
    public:
        FrameBuffer(std::size_t capacity)
            : max_capacity(capacity) {}

        void push(const FramePacket& packet)
        {
            std::lock_guard<std::mutex> lock(mutex);

            if (buffer.size() >= max_capacity)
            {
                buffer.clear();
            }

            buffer.push_back(packet);
        }
        
        bool pop(FramePacket& packet)
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (buffer.empty())
            {
                return false;
            }

            packet = buffer.front();
            buffer.pop_front();

            return true;
        }

    private:
        std::deque<FramePacket> buffer;
        std::size_t max_capacity;
        std::mutex mutex;
};

int main()
{
    std::cout << "Multimodal Edge AI camera module starting." << std::endl;
    
    cv::VideoCapture camera(0);

    if (!camera.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return 1;
    }

    // Shared shutdown flag used for all worker threads.
    std::atomic<bool> running{true};
    FrameBuffer frame_buffer(4);

    // Camera producer thread.
    // Captures frames and sends timestamped FramePackets to the shared buffer.
    std::thread camera_thread([&] ()
    {
        cv::Mat frame;

        std::uint64_t frame_id = 0;

        while(running)
        {
            camera >> frame;
            auto capture_time = std::chrono::steady_clock::now();
        

            if (frame.empty())
            {
                std::cerr << "Error: Failed to capture frame." << std::endl;
                break;
            }

            frame_id++;

            FramePacket packet{
                frame.clone(),
                capture_time,
                frame_id
            };

            frame_buffer.push(packet); 
    }
    });

    while (running)
    {
        FramePacket packet;

        if (frame_buffer.pop(packet))
        {
            cv::putText(
                packet.frame,
                "Frame ID: " + std::to_string(packet.frame_id),
                cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX, 
                1.0,
                cv::Scalar(0, 255, 0),
                2 
            );

            cv::imshow("Camera", packet.frame);
        }

        int key = cv::waitKey(1);

        if (key == 'q')
        {
            // Stops all worker threads
            running = false;
        }
    }

    camera_thread.join();

    return 0;
}