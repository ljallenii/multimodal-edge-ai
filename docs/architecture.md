# System Architecture

## Project Goal

We are building a general-purpose Edge AI system that captures audio and video concurrently, processes each modality through a respective model, fuses the resulting information, and produces fused events that can be consumed by downstream applications.

## Design Objectives

These are the primary goals that we will work toward with the system:

- **Multimodal System** - Process video and audio while designing the architecture to support additional modalities in the future.

- **Concurrency** - Allow the vision and audio pipelines to make progress independently during overlapping periods rather than requiring one pipeline to finish before the other can proceed.

- **Resource Contention** - Measure how concurrent workloads compete for shared compute resources and minimize unnecessary interference between models.

- **Performance** - Maintain low inference and end-to-end latency while providing sufficient throughput and minimizing dropped or stale data.

- **Model Accuracy** - Optimize system and model performance without causing unacceptable degradation in model output quality.

- **Portability** - Design an architecture that is not dependent on a single device class and can be adapted to different edge-computing environments.

- **Buffer Management / Backpressure** - Manage situations where input data is produced faster than it can be processed without allowing queues or stale data to grow indefinitely.

- **Temporal Synchronization and Event Fusion** - Use timing information to associate related events from asynchronous audio and video streams and combine them into meaningful multimodal events.

- **Observability / Performance Instrumentation** - Collect measurements that establish performance baselines, identify bottlenecks, and determine whether system optimizations produce measurable improvements.

## System Data Flow

The camera and microphone independently capture sensor data, which is timestamped and placed into their respective bounded buffers. Vision frames and audio packets are then consumed by independent workers, preprocessed into the format required by their respective models, and passed through inference. Each model produces predictions with associated confidence scores, which the pipeline evaluates to determine whether they should be converted into standardized vision or audio events.

Standardized events are temporarily stored in a recent-event window, allowing the fusion system to compare events that may occur at slightly different times. The fusion pipeline evaluates both the temporal and semantic compatibility of vision and audio events to determine whether a meaningful relationship exists. Compatible events can then be combined into a fused event and passed to a downstream application. Performance is instrumented throughout the pipeline to measure metrics such as queue delay, inference latency, end-to-end latency, throughput, and dropped data.

## Design Decisions

### Favor Fresh Sensor Data

Our system should favor fresh sensor data over stale data because it is designed for real-time perception. When a buffer reaches capacity, older data may be discarded to prevent the system from falling behind and processing information that no longer represents the current environment.

### Preserve Sensor Capture Timestamps

Events should preserve the original sensor capture timestamps because they represent when observations occurred in the real world. Inference completion times may differ between vision and audio models, so using them for event fusion could create inaccurate temporal relationships.

### Standardize Model Outputs

Model-specific predictions should be converted into a standardized event format so that the fusion system does not depend on the output structure of a specific model. This allows vision or audio models to be replaced without requiring major changes to the fusion pipeline.

### Process Modalities Concurrently

Vision and audio should be processed concurrently through independent workers so that one modality does not block the other. This allows the system to continuously process incoming sensor data and maintain real-time performance even when the models have different inference times.