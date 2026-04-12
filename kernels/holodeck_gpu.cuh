/**
 * CUDAClaw Holodeck — GPU-Resident MUD
 * 
 * Rooms are GPU memory blocks. Agents are warp lanes.
 * Combat ticks are kernel launches. The entire holodeck
 * lives on the GPU at massive scale.
 * 
 * Where CPU MUDs handle 10s of agents,
 * CUDAClaw Holodeck handles 10,000+ rooms simultaneously.
 */

#ifndef HOLODECK_GPU_CUH
#define HOLODECK_GPU_CUH

#include <cuda_runtime.h>

// ═══════════════════════════════════════════
// Constants — tune for GPU architecture
// ═══════════════════════════════════════════

#define GPU_MAX_ROOMS          16384
#define GPU_MAX_AGENTS         65536
#define GPU_MAX_EXITS_PER_ROOM 8
#define GPU_MAX_NOTES_PER_ROOM 16
#define GPU_MAX_NAME_LEN       32
#define GPU_MAX_DESC_LEN       128
#define GPU_MAX_MSG_LEN        256
#define GPU_WARP_SIZE          32

// ═══════════════════════════════════════════
// GPU Room — fits in shared memory
// ═══════════════════════════════════════════

/**
 * A room on the GPU. Designed for shared memory residency.
 * Each warp processes one room per tick.
 * 
 * Total size: ~256 bytes (fits in shared memory alongside other data)
 */
typedef struct __align__(16) {
    int   id;                                    // 4
    char  name[GPU_MAX_NAME_LEN];               // 32
    char  description[GPU_MAX_DESC_LEN];        // 128
    int   exit_ids[GPU_MAX_EXITS_PER_ROOM];     // 32
    int   exit_count;                            // 4
    int   agent_ids[GPU_WARP_SIZE];             // 128 — one per warp lane
    int   agent_count;                           // 4
    int   booted;                                // 4
    int   permission_level;                      // 4
    int   gauge_count;                           // 4
    float gauge_values[4];                       // 16 — 4 gauges per room
    int   _padding[4];                           // 16 — alignment
} GPURoom;                                       // ~256 bytes

// ═══════════════════════════════════════════
// GPU Agent — one per warp lane
// ═══════════════════════════════════════════

typedef struct __align__(16) {
    int   id;
    char  name[GPU_MAX_NAME_LEN];
    int   room_id;
    int   permission_level;
    int   hp;
    int   mana;
    int   active;
    int   _padding[2];
} GPUAgent;

// ═══════════════════════════════════════════
// GPU Message — lock-free communication
// ═══════════════════════════════════════════

typedef enum {
    MSG_SAY,      // room-local
    MSG_TELL,     // direct
    MSG_YELL,     // adjacent rooms
    MSG_GOSSIP,   // fleet-wide
    MSG_NOTE,     // wall write
    MSG_COMBAT,   // oversight tick result
} GPUMessageType;

typedef struct __align__(16) {
    int             sender_id;
    int             target_id;    // -1 = broadcast
    int             room_id;
    GPUMessageType  type;
    char            content[GPU_MAX_MSG_LEN];
    int             processed;
    long            timestamp;
} GPUMessage;

// ═══════════════════════════════════════════
// Combat Tick — one kernel launch per round
// ═══════════════════════════════════════════

typedef struct __align__(16) {
    int    room_id;
    int    agent_id;
    float  autonomy_score;
    int    script_version;
    int    action_taken;
    char   action_desc[GPU_MAX_MSG_LEN];
    float  gauge_snapshot[4];
    int    nudge_count;
    int    tick_number;
} GPUCombatTick;

// ═══════════════════════════════════════════
// Holodeck State — the entire world on GPU
// ═══════════════════════════════════════════

typedef struct {
    GPURoom       *rooms;          // [GPU_MAX_ROOMS]
    GPUAgent      *agents;         // [GPU_MAX_AGENTS]
    GPUMessage    *messages;       // ring buffer
    GPUCombatTick *ticks;          // tick history
    int           *room_grid;      // adjacency list
    int            room_count;
    int            agent_count;
    int            message_head;
    int            message_tail;
    int            tick_count;
} GPUHolodeckState;

// ═══════════════════════════════════════════
// Kernel Declarations
// ═══════════════════════════════════════════

/**
 * Process one combat tick across all rooms.
 * Each block handles one room, each thread handles one agent.
 * 
 * <<<room_count, GPU_WARP_SIZE>>>
 */
__global__ void holodeck_combat_tick(
    GPUHolodeckState *state,
    int tick_number
);

/**
 * Process messages for all rooms.
 * Each warp handles one room's message queue.
 * 
 * <<<msg_count / GPU_WARP_SIZE, GPU_WARP_SIZE>>>
 */
__global__ void holodeck_process_messages(
    GPUHolodeckState *state
);

/**
 * Move agents between rooms.
 * Warp-cooperative: agent requests exit, warp validates.
 * 
 * <<<move_count / GPU_WARP_SIZE, GPU_WARP_SIZE>>>
 */
__global__ void holodeck_move_agents(
    GPUHolodeckState *state,
    int *move_requests,   // [agent_id, target_room_id] pairs
    int   move_count
);

/**
 * Boot rooms — initialize when agent enters.
 * Single-threaded per room (lightweight).
 * 
 * <<<boot_count, 1>>>
 */
__global__ void holodeck_boot_rooms(
    GPUHolodeckState *state,
    int *room_ids,
    int   boot_count
);

/**
 * Evaluate combat scripts across all rooms.
 * Each agent's script is evaluated in parallel.
 * Autonomy scores updated atomically.
 * 
 * <<<agent_count / GPU_WARP_SIZE, GPU_WARP_SIZE>>>
 */
__global__ void holodeck_evaluate_scripts(
    GPUHolodeckState *state,
    int tick_number
);

/**
 * Rival combat: two agents compete on same scenario.
 * Warp-level comparison, winner writes result.
 * 
 * <<<scenario_count, GPU_WARP_SIZE>>>
 */
__global__ void holodeck_rival_combat(
    GPUHolodeckState *state,
    int *scenario_ids,
    int   scenario_count,
    int *results        // output: winner per scenario
);

/**
 * Fleet evolution: cross-validate promoted rules.
 * Reduce across all rooms using warp shuffle.
 * 
 * <<<rule_count, GPU_WARP_SIZE>>>
 */
__global__ void holodeck_fleet_evolve(
    GPUHolodeckState *state,
    int *candidate_rules,
    int   rule_count,
    int *validation_scores
);

#endif // HOLODECK_GPU_CUH
