/**
 * CUDAClaw Holodeck — GPU Kernel Implementations
 * 
 * The entire MUD runs on the GPU. Rooms are shared memory blocks.
 * Agents are warp lanes. Combat ticks are kernel launches.
 * Messages flow through lock-free ring buffers.
 * 
 * Scale: 16,384 rooms, 65,536 agents, all updating in parallel.
 */

#include "holodeck_gpu.cuh"
#include <stdio.h>

// ═══════════════════════════════════════════
// Combat Tick — the heartbeat of the holodeck
// ═══════════════════════════════════════════

__global__ void holodeck_combat_tick(
    GPUHolodeckState *state,
    int tick_number
) {
    int room_idx = blockIdx.x;
    int lane = threadIdx.x;
    
    if (room_idx >= state->room_count) return;
    
    GPURoom *room = &state->rooms[room_idx];
    
    // Only first thread per block processes room-level logic
    if (lane == 0) {
        // Update gauges (would read from live systems in production)
        for (int i = 0; i < 4; i++) {
            room->gauge_values[i] = room->gauge_values[i] * 0.95f + 
                                     ((float)(lane + i) / 100.0f) * 0.05f;
        }
    }
    __syncthreads();
    
    // Each lane handles one agent in the room
    if (lane < room->agent_count) {
        int agent_id = room->agent_ids[lane];
        if (agent_id < 0 || agent_id >= state->agent_count) return;
        
        GPUAgent *agent = &state->agents[agent_id];
        if (!agent->active) return;
        
        // Combat tick: evaluate agent state
        // In production, this runs the evolving oversight script
        float autonomy = 1.0f;
        
        // Check gauges — simple threshold evaluation
        for (int g = 0; g < room->gauge_count && g < 4; g++) {
            if (room->gauge_values[g] > 0.8f) {
                autonomy *= 0.7f; // reduce autonomy for elevated gauges
            }
        }
        
        // Warp-level reduction: average autonomy across all agents in room
        for (int offset = GPU_WARP_SIZE / 2; offset > 0; offset /= 2) {
            autonomy += __shfl_down_sync(0xFFFFFFFF, autonomy, offset);
        }
        
        // Thread 0 writes the combat tick
        if (lane == 0) {
            int tick_idx = atomicAdd(&state->tick_count, 1);
            if (tick_idx < GPU_MAX_ROOMS * 100) { // safety bound
                state->ticks[tick_idx].room_id = room->id;
                state->ticks[tick_idx].agent_id = agent_id;
                state->ticks[tick_idx].autonomy_score = autonomy / (float)room->agent_count;
                state->ticks[tick_idx].tick_number = tick_number;
                for (int g = 0; g < 4; g++) {
                    state->ticks[tick_idx].gauge_snapshot[g] = room->gauge_values[g];
                }
            }
        }
    }
}

// ═══════════════════════════════════════════
// Message Processing — room-local say/tell/yell
// ═══════════════════════════════════════════

__global__ void holodeck_process_messages(
    GPUHolodeckState *state
) {
    int warp_id = blockIdx.x;
    int lane = threadIdx.x;
    
    // Each warp processes messages for one room
    // Message queue is consumed cooperatively
    int msg_idx = warp_id;
    
    if (msg_idx >= state->message_head) return;
    
    GPUMessage *msg = &state->messages[msg_idx % 4096]; // ring buffer
    
    if (msg->processed) return;
    
    // Process based on type
    switch (msg->type) {
        case MSG_SAY:
            // Broadcast to all agents in room
            if (msg->room_id >= 0 && msg->room_id < state->room_count) {
                GPURoom *room = &state->rooms[msg->room_id];
                // Lane 0 marks processed
                if (lane == 0) {
                    msg->processed = 1;
                }
            }
            break;
            
        case MSG_TELL:
            // Direct message to target agent
            if (lane == 0 && msg->target_id >= 0 && msg->target_id < state->agent_count) {
                // In production: add to agent's mailbox
                msg->processed = 1;
            }
            break;
            
        case MSG_YELL:
            // Broadcast to adjacent rooms
            if (lane == 0) {
                msg->processed = 1;
            }
            break;
            
        case MSG_GOSSIP:
            // Fleet-wide — handled by host
            if (lane == 0) {
                msg->processed = 1;
            }
            break;
            
        case MSG_NOTE:
            // Write to room wall
            if (lane == 0) {
                msg->processed = 1;
            }
            break;
            
        case MSG_COMBAT:
            if (lane == 0) {
                msg->processed = 1;
            }
            break;
    }
}

// ═══════════════════════════════════════════
// Agent Movement — warp-cooperative validation
// ═══════════════════════════════════════════

__global__ void holodeck_move_agents(
    GPUHolodeckState *state,
    int *move_requests,
    int   move_count
) {
    int move_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (move_idx >= move_count) return;
    
    int agent_id = move_requests[move_idx * 2];
    int target_room = move_requests[move_idx * 2 + 1];
    
    if (agent_id < 0 || agent_id >= state->agent_count) return;
    if (target_room < 0 || target_room >= state->room_count) return;
    
    GPUAgent *agent = &state->agents[agent_id];
    GPURoom *old_room = &state->rooms[agent->room_id];
    GPURoom *new_room = &state->rooms[target_room];
    
    // Validate: does exit exist from old room?
    int exit_valid = 0;
    for (int i = 0; i < old_room->exit_count; i++) {
        if (old_room->exit_ids[i] == target_room) {
            exit_valid = 1;
            break;
        }
    }
    
    if (!exit_valid) return;
    
    // Permission check
    if (agent->permission_level < new_room->permission_level) return;
    
    // Execute move atomically
    // Remove from old room
    for (int i = 0; i < old_room->agent_count; i++) {
        if (old_room->agent_ids[i] == agent_id) {
            old_room->agent_ids[i] = old_room->agent_ids[--old_room->agent_count];
            break;
        }
    }
    
    // Add to new room
    int slot = atomicAdd(&new_room->agent_count, 1);
    if (slot < GPU_WARP_SIZE) {
        new_room->agent_ids[slot] = agent_id;
        agent->room_id = target_room;
    } else {
        // Room full, revert
        atomicSub(&new_room->agent_count, 1);
    }
}

// ═══════════════════════════════════════════
// Room Boot/Shutdown
// ═══════════════════════════════════════════

__global__ void holodeck_boot_rooms(
    GPUHolodeckState *state,
    int *room_ids,
    int   boot_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boot_count) return;
    
    int room_id = room_ids[idx];
    if (room_id < 0 || room_id >= state->room_count) return;
    
    GPURoom *room = &state->rooms[room_id];
    room->booted = 1;
    room->gauge_count = 4;
    for (int i = 0; i < 4; i++) {
        room->gauge_values[i] = 0.0f;
    }
}

// ═══════════════════════════════════════════
// Rival Combat — two agents, same scenario
// ═══════════════════════════════════════════

__global__ void holodeck_rival_combat(
    GPUHolodeckState *state,
    int *scenario_ids,
    int   scenario_count,
    int *results
) {
    int scenario_idx = blockIdx.x;
    int lane = threadIdx.x;
    
    if (scenario_idx >= scenario_count) return;
    
    // Two agents compete on same scenario
    // Warp lanes 0-15: agent A's evaluation
    // Warp lanes 16-31: agent B's evaluation
    int is_agent_b = (lane >= 16);
    int eval_lane = is_agent_b ? (lane - 16) : lane;
    
    // Evaluate script rules in parallel across warp lanes
    // Each lane checks one rule
    float score = 0.0f;
    
    // Simplified scoring: check gauges and compute action quality
    if (eval_lane < 4) {
        float gauge = state->rooms[scenario_idx].gauge_values[eval_lane];
        score = gauge < 0.5f ? 1.0f : (gauge < 0.8f ? 0.5f : 0.0f);
    }
    
    // Warp-level reduction for each half
    int mask = is_agent_b ? 0xFFFF0000 : 0x0000FFFF;
    for (int offset = 8; offset > 0; offset /= 2) {
        score += __shfl_down_sync(mask, score, offset);
    }
    
    // Lane 0 (A) and lane 16 (B) compare
    if (lane == 0) {
        float score_a = score;
        float score_b = __shfl_sync(0xFFFFFFFF, score, 16);
        results[scenario_idx] = (score_a >= score_b) ? 0 : 1;
    }
}

// ═══════════════════════════════════════════
// Fleet Evolution — cross-validate rules
// ═══════════════════════════════════════════

__global__ void holodeck_fleet_evolve(
    GPUHolodeckState *state,
    int *candidate_rules,
    int   rule_count,
    int *validation_scores
) {
    int rule_idx = blockIdx.x;
    int room_lane = threadIdx.x;
    
    if (rule_idx >= rule_count) return;
    
    // Each warp validates one rule against all rooms
    int helps_count = 0;
    
    // Each lane checks a different room
    int room_to_check = room_lane;
    if (room_to_check < state->room_count) {
        // Would evaluate rule against room state
        // Simplified: check if rule's condition matches room state
        helps_count = 1; // placeholder
    }
    
    // Warp reduce: total rooms this rule helps
    for (int offset = GPU_WARP_SIZE / 2; offset > 0; offset /= 2) {
        helps_count += __shfl_down_sync(0xFFFFFFFF, helps_count, offset);
    }
    
    if (room_lane == 0) {
        validation_scores[rule_idx] = helps_count;
    }
}
