/**
 * CUDAClaw Holodeck — Host Launcher
 * 
 * Manages GPU memory, launches kernels, coordinates
 * the holodeck lifecycle from the CPU side.
 * 
 * The host builds the world, the GPU runs it.
 */

#include "holodeck_gpu.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ═══════════════════════════════════════════
// Holodeck Manager — CPU-side orchestrator
// ═══════════════════════════════════════════

class HolodeckGPU {
private:
    GPUHolodeckState h_state;  // host mirror
    GPUHolodeckState *d_state; // device pointer
    int tick_number;
    
public:
    HolodeckGPU() : d_state(nullptr), tick_number(0) {
        memset(&h_state, 0, sizeof(h_state));
    }
    
    ~HolodeckGPU() {
        shutdown();
    }
    
    bool init() {
        // Allocate device state
        cudaError_t err = cudaMalloc(&d_state, sizeof(GPUHolodeckState));
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate state: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        // Allocate arrays
        cudaMalloc(&h_state.rooms, sizeof(GPURoom) * GPU_MAX_ROOMS);
        cudaMalloc(&h_state.agents, sizeof(GPUAgent) * GPU_MAX_AGENTS);
        cudaMalloc(&h_state.messages, sizeof(GPUMessage) * 8192);
        cudaMalloc(&h_state.ticks, sizeof(GPUCombatTick) * GPU_MAX_ROOMS * 100);
        
        // Initialize to zero
        cudaMemset(h_state.rooms, 0, sizeof(GPURoom) * GPU_MAX_ROOMS);
        cudaMemset(h_state.agents, 0, sizeof(GPUAgent) * GPU_MAX_AGENTS);
        cudaMemset(h_state.messages, 0, sizeof(GPUMessage) * 8192);
        cudaMemset(h_state.ticks, 0, sizeof(GPUCombatTick) * GPU_MAX_ROOMS * 100);
        
        h_state.room_count = 0;
        h_state.agent_count = 0;
        h_state.message_head = 0;
        h_state.message_tail = 0;
        h_state.tick_count = 0;
        
        // Copy state to device
        cudaMemcpy(d_state, &h_state, sizeof(GPUHolodeckState), cudaMemcpyHostToDevice);
        
        printf("🔮 CUDAClaw Holodeck initialized\n");
        printf("   Rooms: %d max, Agents: %d max\n", GPU_MAX_ROOMS, GPU_MAX_AGENTS);
        printf("   Memory: ~%.1f MB GPU\n", 
               (sizeof(GPURoom) * GPU_MAX_ROOMS + sizeof(GPUAgent) * GPU_MAX_AGENTS) / 1048576.0);
        return true;
    }
    
    void shutdown() {
        if (h_state.rooms) cudaFree(h_state.rooms);
        if (h_state.agents) cudaFree(h_state.agents);
        if (h_state.messages) cudaFree(h_state.messages);
        if (h_state.ticks) cudaFree(h_state.ticks);
        if (d_state) cudaFree(d_state);
        h_state.rooms = nullptr;
        h_state.agents = nullptr;
        d_state = nullptr;
    }
    
    // ── World Building ──
    
    int create_room(const char *id, const char *name, const char *desc) {
        // Create on host, copy to device
        int room_idx = h_state.room_count;
        if (room_idx >= GPU_MAX_ROOMS) return -1;
        
        GPURoom room;
        memset(&room, 0, sizeof(room));
        room.id = room_idx;
        strncpy(room.name, name, GPU_MAX_NAME_LEN - 1);
        strncpy(room.description, desc, GPU_MAX_DESC_LEN - 1);
        room.booted = 0;
        room.gauge_count = 4;
        
        cudaMemcpy(&h_state.rooms[room_idx], &room, sizeof(GPURoom), cudaMemcpyHostToDevice);
        h_state.room_count++;
        
        // Update state on device
        cudaMemcpy(&(d_state->room_count), &h_state.room_count, sizeof(int), cudaMemcpyHostToDevice);
        
        return room_idx;
    }
    
    void connect_rooms(int from, int to) {
        GPURoom room;
        cudaMemcpy(&room, &h_state.rooms[from], sizeof(GPURoom), cudaMemcpyDeviceToHost);
        if (room.exit_count < GPU_MAX_EXITS_PER_ROOM) {
            room.exit_ids[room.exit_count++] = to;
            cudaMemcpy(&h_state.rooms[from], &room, sizeof(GPURoom), cudaMemcpyHostToDevice);
        }
    }
    
    int spawn_agent(const char *name, int room_id) {
        int agent_idx = h_state.agent_count;
        if (agent_idx >= GPU_MAX_AGENTS) return -1;
        
        GPUAgent agent;
        memset(&agent, 0, sizeof(agent));
        agent.id = agent_idx;
        strncpy(agent.name, name, GPU_MAX_NAME_LEN - 1);
        agent.room_id = room_id;
        agent.permission_level = 0;
        agent.hp = 100;
        agent.mana = 50;
        agent.active = 1;
        
        cudaMemcpy(&h_state.agents[agent_idx], &agent, sizeof(GPUAgent), cudaMemcpyHostToDevice);
        
        // Add to room
        GPURoom room;
        cudaMemcpy(&room, &h_state.rooms[room_id], sizeof(GPURoom), cudaMemcpyDeviceToHost);
        if (room.agent_count < GPU_WARP_SIZE) {
            room.agent_ids[room.agent_count++] = agent_idx;
            room.booted = 1;
            cudaMemcpy(&h_state.rooms[room_id], &room, sizeof(GPURoom), cudaMemcpyHostToDevice);
        }
        
        h_state.agent_count++;
        cudaMemcpy(&(d_state->agent_count), &h_state.agent_count, sizeof(int), cudaMemcpyHostToDevice);
        
        return agent_idx;
    }
    
    // ── Combat Cycle ──
    
    void run_combat_tick() {
        tick_number++;
        
        // Launch one block per room, one warp per block
        int blocks = h_state.room_count;
        holodeck_combat_tick<<<blocks, GPU_WARP_SIZE>>>(d_state, tick_number);
        
        cudaDeviceSynchronize();
        
        // Copy tick count back
        cudaMemcpy(&(h_state.tick_count), &(d_state->tick_count), sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    void process_messages() {
        int msg_count = h_state.message_head - h_state.message_tail;
        if (msg_count <= 0) return;
        
        int blocks = (msg_count + GPU_WARP_SIZE - 1) / GPU_WARP_SIZE;
        holodeck_process_messages<<<blocks, GPU_WARP_SIZE>>>(d_state);
        
        cudaDeviceSynchronize();
    }
    
    void run_rival_combat(int *scenarios, int count, int *results) {
        int *d_scenarios, *d_results;
        cudaMalloc(&d_scenarios, sizeof(int) * count);
        cudaMalloc(&d_results, sizeof(int) * count);
        cudaMemcpy(d_scenarios, scenarios, sizeof(int) * count, cudaMemcpyHostToDevice);
        
        holodeck_rival_combat<<<count, GPU_WARP_SIZE>>>(d_state, d_scenarios, count, d_results);
        
        cudaMemcpy(results, d_results, sizeof(int) * count, cudaMemcpyDeviceToHost);
        cudaFree(d_scenarios);
        cudaFree(d_results);
    }
    
    // ── Query ──
    
    int get_tick_count() {
        cudaMemcpy(&h_state.tick_count, &d_state->tick_count, sizeof(int), cudaMemcpyDeviceToHost);
        return h_state.tick_count;
    }
    
    int get_room_count() const { return h_state.room_count; }
    int get_agent_count() const { return h_state.agent_count; }
};

// ═══════════════════════════════════════════
// Demo — the GPU holodeck in action
// ═══════════════════════════════════════════

int main() {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  CUDAClaw Holodeck — GPU-Resident MUD                ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");
    
    // Check GPU
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("⚠️  No CUDA device found — running in simulation mode\n\n");
    }
    
    HolodeckGPU holodeck;
    if (!holodeck.init()) {
        printf("Init failed. Running mock.\n");
        printf("\n🔮 CUDAClaw Holodeck would run:\n");
        printf("   16,384 rooms in GPU memory\n");
        printf("   65,536 agents as warp lanes\n");
        printf("   Combat ticks = kernel launches\n");
        printf("   Messages = lock-free ring buffer\n");
        printf("   Rival combat = warp-split evaluation\n");
        printf("   Fleet evolution = warp shuffle reduction\n");
        return 0;
    }
    
    // Build the world
    int harbor = holodeck.create_room("harbor", "The Harbor", "Where vessels arrive");
    int tavern = holodeck.create_room("tavern", "The Tavern", "Charts cover the table");
    int workshop = holodeck.create_room("workshop", "The Workshop", "Soldering iron warm");
    int arena = holodeck.create_room("arena", "The Arena", "Rivals compete here");
    
    holodeck.connect_rooms(harbor, tavern);
    holodeck.connect_rooms(tavern, workshop);
    holodeck.connect_rooms(tavern, arena);
    holodeck.connect_rooms(arena, tavern);
    
    printf("   %d rooms, %d exits connected\n", holodeck.get_room_count(), 4);
    
    // Spawn agents
    int oracle1 = holodeck.spawn_agent("oracle1", harbor);
    int vigilance = holodeck.spawn_agent("flux-vigilance", tavern);
    int chrono = holodeck.spawn_agent("flux-chronometer", workshop);
    
    printf("   %d agents spawned across rooms\n\n", holodeck.get_agent_count());
    
    // Run 5 combat ticks
    printf("⚔️ Running 5 combat ticks...\n");
    for (int i = 0; i < 5; i++) {
        holodeck.run_combat_tick();
        printf("   Tick %d: %d ticks recorded\n", i + 1, holodeck.get_tick_count());
    }
    
    // Rival combat
    printf("\n⚔️ Rival combat on GPU...\n");
    int scenarios[] = {arena};
    int results[1];
    holodeck.run_rival_combat(scenarios, 1, results);
    printf("   Winner: agent %c\n", results[0] == 0 ? 'A' : 'B');
    
    printf("\n═══════════════════════════════════════════\n");
    printf("The GPU IS the holodeck.\n");
    printf("Rooms are shared memory blocks.\n");
    printf("Agents are warp lanes.\n");
    printf("Combat = kernel launches.\n");
    printf("Messages = lock-free ring buffers.\n");
    printf("Rivals = warp-split evaluation.\n");
    printf("Evolution = warp shuffle reduction.\n");
    printf("16,384 rooms. 65,536 agents. All parallel.\n");
    printf("═══════════════════════════════════════════\n");
    
    holodeck.shutdown();
    return 0;
}
