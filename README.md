# holodeck-cuda

**GPU-resident holodeck MUD.** 16K rooms, 65K agents, warp-level combat. The entire simulation lives on the GPU.

## Architecture

```
holodeck-cuda/
├── kernels/
│   ├── holodeck_gpu.cuh      ← GPU data structures (rooms, agents, combat)
│   ├── holodeck_kernels.cu   ← CUDA kernels (room ticks, combat, messaging)
│   └── holodeck_host.cu      ← CPU-side orchestrator (memory management, launch)
└── benchmark.cu              ← Performance benchmarks
```

## GPU Design

### Rooms in Shared Memory
Each `GPURoom` is ~256 bytes, designed to fit in CUDA shared memory alongside other data. One warp processes one room per tick.

```c
GPURoom {
    id, name, description
    exit_ids[8]          // 8 exits max per room
    agent_ids[32]        // one per warp lane
    permission_level     // access control
    gauge_count          // state tracking
}
```

### Agents as Warp Lanes
Each `GPUAgent` occupies a warp lane. Up to 65,536 agents active simultaneously.

### Combat as Kernel Launches
Combat ticks are CUDA kernel launches. The host orchestrates; the GPU executes at scale.

### Memory Layout
- **Rooms**: 16,384 GPU-resident (fixed allocation)
- **Agents**: 65,536 GPU-resident
- **Messages**: 8,192 buffered
- **Combat Ticks**: 16,384 × 100 buffered

## Scale Comparison

| System | Rooms | Agents | Latency |
|--------|-------|--------|---------|
| CPU MUD (Python) | ~100 | ~10 | ~100ms |
| CPU MUD (Rust) | ~1,000 | ~100 | ~10ms |
| **holodeck-cuda** | **16,384** | **65,536** | **<1ms** |

## Building

```bash
nvcc -arch=sm_87 kernels/holodeck_kernels.cu kernels/holodeck_host.cu -o holodeck-cuda
```

Target: Jetson Orin (sm_87) / RTX 4050 (sm_89)

## Fleet Role

holodeck-cuda is the GPU-accelerated tier of the holodeck stack:
- **holodeck-rust** — Production MUD (CPU, multi-agent)
- **holodeck-c** — Conformance testing (ISA validation)
- **holodeck-cuda** — GPU simulation (massive scale)
- **holodeck-go** / **holodeck-zig** — Language explorations

Part of the Cocapn fleet. See `cudaclaw` for the broader GPU-resident agent runtime.

## License

Proprietary — SuperInstance/Cocapn
