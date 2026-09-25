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