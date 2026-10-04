#include <iostream>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
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
            {
                std::lock_guard<std::mutex> lock(mutex);

                if (buffer.size() >= max_capacity)
                {
                    // Records every qeueued frame that is discarded by 
                    // the freshness-first policy
                    dropped_frames += buffer.size();

                    buffer.clear();
                }

                buffer.push_back(packet);
            }
            // Wake one waiting consumer because new frame data is available.
            data_available.notify_one();
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

        std::uint64_t get_dropped_frames()
        {
            // Protect the metric while it is being read because push()
            // may update it concurrently from the producer thread.
            std::lock_guard<std::mutex> lock(mutex);

            return dropped_frames;
        }

        bool wait_and_pop(FramePacket& packet, const std::atomic<bool>& running)
        {
             std::unique_lock<std::mutex> lock(mutex);

            // Sleep while there is no work. wait() temporarily releases the 
            // mutex so the producer can push a frame and wake this consumer.
            data_available.wait(lock, [&]()
            {
                return !buffer.empty() || !running;
            });
            
            // The consu,er may have been awakened only because the application
            // is shutting down, so there may not actually be a frame available.
            if (buffer.empty())
            {
                return false;
            }

            packet = buffer.front();
            buffer.pop_front();

            return true;
        }

        void notify_shutdown()
        {
            // Wake any consumer that may be sleeping so it can observe
            // the application's update shutdown state.
            data_available.notify_one();
        }

    private:
        std::deque<FramePacket> buffer;
        std::size_t max_capacity;
        std::mutex mutex;

        // Wakes consumers when new frame data becomes available.
        std::condition_variable data_available;

        // Total number of frames discarded when the buffer reaches capacity.
        // Used to observe whether the consumer is falling behind the producer.
        std::uint64_t dropped_frames{0};
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

    // Counts successfully captured camera frames.
    // Atomic allows the camera producer to update the counter while
    // the main thread reads it for performance reporting.
    std::atomic<std::uint64_t> captured_frames{0};

    std::atomic<std::uint64_t> processed_frames{0};

    // Stores the cumylative capture count from the previous report.
    // The differenve between this and the current count gives the
    // number of frames captured during the report.
    std::uint64_t prev_captured{0};
    std::uint64_t prev_processed{0};

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
                running = false;
                frame_buffer.notify_shutdown();
                break;
            }

            frame_id++;

            captured_frames++;

            FramePacket packet{
                frame.clone(),
                capture_time,
                frame_id
            };

            frame_buffer.push(packet); 
    }
    });

    auto last_report_time = std::chrono::steady_clock::now();

    std::thread vision_thread([&]()
    {
        while (running)
        {
            FramePacket packet;

            if (!frame_buffer.wait_and_pop(packet, running))
            {
                // No frame was returned because the system is shutting down.
                break;
            }

            // Future vision preprocessing and inference will run here.
            processed_frames++;
        }
    });

    // Defines a fixed runtime for the current headless pipeline test.
    auto test_start_time = std::chrono::steady_clock::now();

    while (running)
    {
        auto current_time = std::chrono::steady_clock::now();

        auto report_interval = std::chrono::duration_cast<std::chrono::seconds>(
            current_time - last_report_time
        );

        if (report_interval.count() >= 1)
        {
            std::uint64_t current_captured = captured_frames.load();
            std::uint64_t current_processed = processed_frames.load();

            // Measure the precise elapsed time between reports.
            // Using a floating-point duration preserves sub-second precision.
            std::chrono::duration<double> elapsed = current_time - last_report_time;

            double capture_fps = static_cast<double>(current_captured - prev_captured) / elapsed.count();
            double consumer_fps = static_cast<double>(current_processed - prev_processed) / elapsed.count();

            std::cout << "Report:\n" 
                << "Capture FPS: " << capture_fps
                << " | Consumer FPS: " << consumer_fps
                << " | Total captured: " << current_captured
                << " | Dropped frames: " << frame_buffer.get_dropped_frames()
                << std::endl;

            prev_captured = current_captured;
            prev_processed = current_processed;

            last_report_time = current_time;
        }

        auto test_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - test_start_time);

        if (test_elapsed.count() >= 10.0)
        {
            // Signal both workers to stop and wake any consumer that may
            // Currently be blocked waiting for another frame.
            running = false;
            frame_buffer.notify_shutdown();
        }
    }

    camera_thread.join();
    vision_thread.join();

    return 0;
}