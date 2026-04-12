/**
 * benchmark.cu — Tick latency benchmarks at various room/agent scales
 * Build: nvcc -o benchmark benchmark.cu kernels/holodeck_kernels.cu -arch=sm_87 -I.
 */
#include "kernels/holodeck_gpu.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Scale { const char *name; int rooms, agents, ticks; };
static const Scale scales[] = {
    {"100 rooms, 500 agents",     100,   500,   1000},
    {"1000 rooms, 5000 agents",   1000,  5000,  1000},
    {"5000 rooms, 25000 agents",  5000,  25000, 500},
    {"16384 rooms, 65536 agents", 16384, 65536, 200},
};

int main() {
    printf("═══════════════════════════════════════════════════\n");
    printf("  CUDAClaw Holodeck — Tick Latency Benchmarks\n");
    printf("  Jetson Super Orin Nano — 1024 CUDA cores\n");
    printf("═══════════════════════════════════════════════════\n\n");

    for (int s = 0; s < 4; s++) {
        const Scale &cfg = scales[s];
        GPUHolodeckState h; memset(&h, 0, sizeof(h));
        GPUHolodeckState *d;
        cudaMalloc(&d, sizeof(GPUHolodeckState));
        
        size_t rb = sizeof(GPURoom) * cfg.rooms;
        size_t ab = sizeof(GPUAgent) * cfg.agents;
        size_t tb = sizeof(GPUCombatTick) * cfg.rooms * 100;
        
        cudaMalloc(&h.rooms, rb);
        cudaMalloc(&h.agents, ab);
        cudaMalloc(&h.messages, sizeof(GPUMessage) * 8192);
        cudaMalloc(&h.ticks, tb);
        
        cudaMemset(h.rooms, 0, rb);
        cudaMemset(h.agents, 0, ab);
        cudaMemset(h.messages, 0, sizeof(GPUMessage) * 8192);
        cudaMemset(h.ticks, 0, tb);
        
        h.room_count = cfg.rooms;
        h.agent_count = cfg.agents;
        cudaMemcpy(d, &h, sizeof(h), cudaMemcpyHostToDevice);
        
        // Warmup
        int blk = (cfg.rooms + GPU_WARP_SIZE - 1) / GPU_WARP_SIZE;
        holodeck_combat_tick<<<blk, GPU_WARP_SIZE>>>(d, 1);
        cudaDeviceSynchronize();
        
        // Benchmark
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int t = 0; t < cfg.ticks; t++)
            holodeck_combat_tick<<<blk, GPU_WARP_SIZE>>>(d, t+1);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        float us = (ms / cfg.ticks) * 1000.0f;
        
        printf("  %s\n", cfg.name);
        printf("    %d ticks in %.2f ms\n", cfg.ticks, ms);
        printf("    %.1f us/tick (%.0f ticks/sec)\n", us, 1000000.0f / us);
        printf("    GPU alloc: %.1f MB\n\n", (rb+ab+tb)/1048576.0);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(h.rooms);
        cudaFree(h.agents);
        cudaFree(h.messages);
        cudaFree(h.ticks);
        cudaFree(d);
    }
    
    printf("═══════════════════════════════════════════════════\n");
    return 0;
}
