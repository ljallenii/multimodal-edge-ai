#include <iostream>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <vector>
#include <cstring>
#include <array>
#include <algorithm>
#include <utility>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>

struct FramePacket
{
    cv::Mat frame;
    std::chrono::steady_clock::time_point capture_time;
    std::uint64_t frame_id;
};

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

struct VisionEvent
{
    std::uint64_t frame_id;
    std::chrono::steady_clock::time_point capture_time;
    std::vector<Detection> detections;
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

    // Owns the global ONNX Runtime environment used by the inference.
    Ort::Env ort_env(ORT_LOGGING_LEVEL_WARNING, "MultimodalEdgeAI");

    // Configure how ONNX Runtime will execute the vision model.
    Ort::SessionOptions session_options;

    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    const wchar_t* model_path = L"models/yolov8n.onnx";

    Ort::Session vision_session(
        ort_env,
        model_path,
        session_options
    );
    
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

    std::atomic<double> average_inference_ms{0.0};

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
        double inference_time_sum_ms = 0.0;
        std::uint64_t inference_count = 0;

        while (running)
        {
            FramePacket packet;

            if (!frame_buffer.wait_and_pop(packet, running))
            {
                // No frame was returned because the system is shutting down.
                break;
            }

            // Letterboxed the capture frame to the spatial dimensions expected by YOLO.
            const int model_width = 640;
            const int model_height = 640;

            const float scale = std::min(
                static_cast<float>(model_width) / packet.frame.cols,
                static_cast<float>(model_height) / packet.frame.rows
            );

            const int resized_width = static_cast<int>(packet.frame.cols * scale);
            const int resized_height = static_cast<int>(packet.frame.rows * scale);

            cv::Mat resized_frame;

            // Resize while preserving the camera frame's original aspect ratio.
            cv::resize(packet.frame, resized_frame, cv::Size(resized_width, resized_height));

            const int pad_width = model_width - resized_width;
            const int pad_height = model_height - resized_height;

            const int pad_left = pad_width / 2;
            const int pad_right = pad_width - pad_left;
            const int pad_top = pad_height / 2;
            const int pad_bottom = pad_height - pad_top;

            cv::Mat letterboxed_frame;

            // Pad the resized image to the model's required 640x640 input
            // without distorting the original aspect ratio.
            cv::copyMakeBorder(
                resized_frame,
                letterboxed_frame,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv::BORDER_CONSTANT,
                cv::Scalar(114, 114, 114)
            );

            // Convert OpenCV's BGR channel order ro the RGB order expected by YOLO.
            cv::Mat rgb_frame;
            cv::cvtColor(
                letterboxed_frame,
                rgb_frame,
                cv::COLOR_BGR2RGB
            );

            // Convert pixels to float32 and normalize from [0, 255] to [0, 1].
            cv::Mat float_frame;
            rgb_frame.convertTo(
                float_frame,
                CV_32F,
                1.0 / 255.0
            );

            // Rearrange interleaved RGB pixels into seperate channel planes.
            std::vector<cv::Mat> channels(3);
            cv::split(float_frame, channels);

            // Store the model input as one contiguous CHW float buffer.
            std::vector<float> input_tensor_values(
                3 * 640 * 640
            );

            const std::size_t channel_size = 640 * 640;

            for (std::size_t c = 0; c < 3; ++c)
            {
                std::memcpy(
                    input_tensor_values.data() + c * channel_size,
                    channels[c].ptr<float>(),
                    channel_size * sizeof(float)
                );
            }

            std::array<int64_t, 4> input_shape{1, 3, 640, 640};

            // Describe the CPU memory that backs the input tensor.
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault
            );

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                input_tensor_values.data(),
                input_tensor_values.size(),
                input_shape.data(),
                input_shape.size()
            );

            const char* input_names[] = {"images"};
            const char* output_names[] = {"output0"};

            // Measure model inference seperately from camera capture,
            // preprocessing, and postprocessing.
            auto inference_start = std::chrono::steady_clock::now();

            // Run YOLO inference on the preprocessed frame.
            auto output_tensors = vision_session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_names,
                1
            );

            auto inference_end = std::chrono::steady_clock::now();

            double inference_ms = std::chrono::duration<double, std::milli>(
                inference_end - inference_start
            ).count();

            inference_time_sum_ms += inference_ms;
            inference_count++;

            average_inference_ms.store(
                inference_time_sum_ms / static_cast<double>(inference_count)
            );

            float* output_data = output_tensors[0].GetTensorMutableData<float>();

            const int num_candidates = 8400;
            const int num_classes = 80;
            const float confidence_threshold = 0.25f;

            std::vector<cv::Rect> boxes;
            std::vector<float> scores;
            std::vector<int> class_ids;

            for (int i = 0; i < num_candidates; ++i)
            {
                float best_score = 0.0f;
                int best_class = -1;

                // Find the highest-scoring class for this candidate.
                for (int c = 0; c < num_classes; ++c)
                {
                    float score = output_data[(4 + c) * num_candidates + i];

                    if (score > best_score)
                    {
                        best_score = score;
                        best_class = c;
                    }
                }

                if (best_score < confidence_threshold)
                {
                    continue;
                }

                // Decode the bounding box for a candidate that passed confidence filtering.
                float x_center = output_data[0 * num_candidates + i];
                float y_center = output_data[1 * num_candidates + i];
                float width = output_data[2 * num_candidates + i];
                float height = output_data[3 * num_candidates + i];

                // Map the detection from the 640x640 letterboxed image
                // back to coordinates in the original camera frame.
                float x1 = x_center - (width / 2.0f);
                float y1 = y_center - (height / 2.0f);
                float x2 = x_center + (width / 2.0f);
                float y2 = y_center + (height / 2.0f);

                // Remove letterbox padding and undo the resize so the
                // detection coordinates match the original camera frame.
                x1 = (x1 - pad_left) / scale;
                y1 = (y1 - pad_top) / scale;
                x2 = (x2 - pad_left) / scale;
                y2 = (y2 - pad_top) / scale;

                // Keep the box inside the originalcamera frame.
                x1 = std::clamp(x1, 0.0f, static_cast<float>(packet.frame.cols - 1));
                y1 = std::clamp(y1, 0.0f, static_cast<float>(packet.frame.rows - 1));
                x2 = std::clamp(x2, 0.0f, static_cast<float>(packet.frame.cols - 1));
                y2 = std::clamp(y2, 0.0f, static_cast<float>(packet.frame.rows - 1));

                // Store confidence-filtered detection for later NMS.
                boxes.emplace_back(
                    static_cast<int>(x1),
                    static_cast<int>(y1),
                    static_cast<int>(x2-x1),
                    static_cast<int>(y2-y1)
                );

                scores.push_back(best_score);
                class_ids.push_back(best_class);
            }

            std::vector<int> nms_indices;

            // Suppress highly overlapping detections while keeping
            // the strongest confidence candidates.
            cv::dnn::NMSBoxesBatched(
                boxes,
                scores,
                class_ids,
                confidence_threshold,
                0.45f,
                nms_indices
            );

            std::vector<Detection> detections;

            // Inspect detections that survived confidence filtering and NMS.
            for (int index : nms_indices)
            {
                const cv::Rect& box = boxes[index];
                float score = scores[index];
                int class_id = class_ids[index];

                // Convert the surviving YOLO candidate into the structured
                // representation used by the rest of the vision pipeline.
                Detection detection{
                    class_id,
                    score,
                    box
                };

                detections.push_back(detection);
            }

            VisionEvent vision_event{
                packet.frame_id,
                packet.capture_time,
                std::move(detections)
            };

            processed_frames++;
        }
    });

    // Inspect the model input so proprocessing can be validated
    // against the interface declared by the ONNX model.
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name = vision_session.GetInputNameAllocated(0, allocator);

    Ort::TypeInfo input_type_info = vision_session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();

    std::vector<int64_t> model_input_shape = input_tensor_info.GetShape();

    std::cout << "Model input: " << input_name.get() << "\n";
    std::cout << "Input shape: ";

    for (int64_t dim : model_input_shape)
    {
        std::cout << dim << " ";
    }

    std::cout << "\n";

    // Inspect the model output so postprocessing is based on
    // the model's actual output format rather than an assumed shape.
    auto output_name = vision_session.GetOutputNameAllocated(0, allocator);

    Ort::TypeInfo output_type_info = vision_session.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();

    std::vector<int64_t> model_output_shape = output_tensor_info.GetShape();

    std::cout << "Model output: " << output_name.get() << "\n";
    std::cout << "Output shape: ";

    for (int64_t dim : model_output_shape)
    {
        std::cout << dim << " ";
    }

    std::cout << "\n";

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
                << " | Avg inference: " << average_inference_ms.load() << " ms"
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