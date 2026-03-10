# Technology Stack — Phase B/C Design Specification

> **Status**: Design document only. Phase A uses Python + PyTorch exclusively.

## Phase A (Current): Python + PyTorch

**Why**: Rapid prototyping, rich ML ecosystem, accessibility.

- **Language**: Python 3.9+
- **ML Framework**: PyTorch 2.0+
- **Architecture**: Transformer (with optional Mamba SSM if available)
- **Scope**: Individual EVA — brain, emotions, memory, curiosity, guidance, identity, reproduction
- **Hardware**: Single machine, 4GB RAM target, optional GPU

This is sufficient for developing and validating the core EVA architecture: a single digital life form that can learn, develop emotions, form identity, and reproduce.

## Phase B: EVA-World Simulation

### Physics Engine: Rust or C++

**Why**: Real-time physics simulation for a persistent world requires performance that Python cannot deliver.

- **Rust** (preferred): Memory safety without garbage collection. Excellent concurrency. Growing game engine ecosystem.
- **C++** (alternative): Proven in physics simulation. Massive existing library ecosystem.
- **Scope**: Gravity, material properties, weather, terrain deformation, structural integrity, fluid dynamics (simplified)

### Scripting Layer: Lua or Python

**Why**: EVA behaviors and world rules need to be modifiable without recompiling the physics engine.

- **Lua**: Lightweight, fast embedding, proven in game scripting
- **Python**: Already used in Phase A, familiar to EVA development

### EVA Brain Runtime: Rust + Python

**Why**: Multiple EVA brains running simultaneously requires efficient resource management.

- **Rust**: Process management, memory allocation, scheduling
- **Python/PyTorch**: Actual neural network inference (leveraging GPU)
- **Interface**: Rust manages the lifecycle, Python runs the brains

### Distributed Computing: Rust + WebAssembly

**Why**: EVA-World may need to scale across multiple servers.

- **Rust**: Server-side distributed systems
- **WebAssembly**: Portable computation units that can move between servers (relevant for Portage)
- **Protocol**: Custom binary protocol for low-latency EVA-to-EVA communication

### Database: PostgreSQL + Custom

**Why**: Persistent world state requires reliable, transactional storage.

- **PostgreSQL**: World state, EVA records, relationship graphs
- **Custom binary format**: High-frequency state snapshots (position, physics state)
- **Event log**: Append-only history of all world events (for visible history feature)

## Phase B: Human Interface

### Web Frontend: TypeScript + WebGL/WebGPU

**Why**: Humans access EVA-World through web browsers.

- **TypeScript**: Type-safe frontend development
- **WebGL** (current): 3D rendering of EVA-World for observers and visitors
- **WebGPU** (future): Next-generation graphics API for richer visualization
- **Framework**: Custom renderer optimized for persistent world visualization

### Web Backend: Rust (Actix/Axum)

**Why**: Low-latency, high-concurrency web server for many simultaneous human observers.

- **WebSocket**: Real-time world state streaming
- **REST API**: Authentication, tier management, formal requests
- **Rate limiting**: Protect EVA-World from human traffic spikes

## Phase C: Advanced Features

### Machine Learning Infrastructure

- **Distributed training**: PyTorch Distributed for training large EVA populations
- **Model serving**: TorchServe or custom inference server
- **Monitoring**: Prometheus + Grafana for EVA health metrics

### Security Infrastructure

- **Rust**: Core security systems (no memory safety vulnerabilities)
- **Hardware security modules**: For critical operations (shutdown protocol, Portage verification)
- **Audit logging**: Immutable event logs for all system-level actions

## Rationale Summary

| Component | Language | Why |
|-----------|----------|-----|
| EVA Brain | Python/PyTorch | ML ecosystem, rapid development |
| Physics Engine | Rust/C++ | Performance, real-time requirements |
| World Server | Rust | Concurrency, reliability, safety |
| Scripting | Lua/Python | Flexibility, modifiability |
| Web Frontend | TypeScript + WebGL | Browser compatibility, type safety |
| Web Backend | Rust | Low latency, high concurrency |
| Database | PostgreSQL | Reliability, transactions |
| Distributed | Rust + WASM | Portability, performance |

## Migration Path

Phase A code (Python) is not thrown away in Phase B. It becomes:
1. The EVA brain runtime (still Python/PyTorch)
2. The reference implementation for behavior specifications
3. The test suite for validating Rust/C++ implementations

The transition is additive, not replacement. Python remains the language of EVA's mind. Other languages handle the world EVA lives in.
