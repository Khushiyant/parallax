#include "prlx_runtime.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <ctime>
#include <sys/stat.h>
#include <errno.h>

#ifdef PRLX_HAS_ZSTD
#include <zstd.h>
#endif

// Global state
static void* d_trace_buffer = nullptr;
static size_t trace_buffer_size = 0;
static void* d_history_buffer = nullptr;
static size_t history_buffer_size = 0;
static uint32_t history_depth = 0;
static void* d_snapshot_buffer = nullptr;
static size_t snapshot_buffer_size = 0;
static uint32_t snapshot_depth = 0;
static char output_path[256] = "trace.prlx";
static bool initialized = false;
static bool enabled = true;

// Configuration from environment variables
// events_per_warp is compile-time only (PRLX_EVENTS_PER_WARP) — device code
// uses this constant for buffer stride and overflow checks. A runtime override
// would cause host/device stride mismatch and GPU memory corruption.
static const uint32_t events_per_warp = PRLX_EVENTS_PER_WARP;
static uint32_t sample_rate = 1;
static bool compress_enabled = false;

// Session state
static bool session_active = false;
static char session_dir[256] = "";
static uint32_t session_launch_count = 0;

#define PRLX_MAX_SESSION_ENTRIES 256
struct SessionEntry {
    char kernel_name[64];
    char filename[256];
    uint32_t launch_idx;
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
};
static SessionEntry session_entries[PRLX_MAX_SESSION_ENTRIES];
static uint32_t session_entry_count = 0;

// FNV-1a hash for kernel name (64-bit)
static uint64_t fnv1a_hash(const char* str) {
    uint64_t hash = 14695981039346656037ULL;
    while (*str) {
        hash ^= (uint64_t)(*str++);
        hash *= 1099511628211ULL;
    }
    return hash;
}

// Initialize tracing system (called once at startup)
extern "C" void prlx_init(void) {
    if (initialized) return;

    const char* trace_path = getenv("PRLX_TRACE");
    if (trace_path) {
        strncpy(output_path, trace_path, sizeof(output_path) - 1);
        output_path[sizeof(output_path) - 1] = '\0';
    }

    const char* enabled_str = getenv("PRLX_ENABLED");
    enabled = !enabled_str || atoi(enabled_str) != 0;

    // Note: PRLX_BUFFER_SIZE env var was removed — events_per_warp is compile-time
    // only (PRLX_EVENTS_PER_WARP). Changing it at runtime would cause device/host
    // stride mismatch. To change buffer size, recompile with -DPRLX_EVENTS_PER_WARP=N.

    const char* history_depth_str = getenv("PRLX_HISTORY_DEPTH");
    if (history_depth_str) {
        history_depth = atoi(history_depth_str);
    } else {
        history_depth = PRLX_HISTORY_DEPTH_DEFAULT;
    }

    const char* sample_rate_str = getenv("PRLX_SAMPLE_RATE");
    if (sample_rate_str) {
        sample_rate = atoi(sample_rate_str);
        if (sample_rate < 1) sample_rate = 1;
    }

    const char* snapshot_depth_str = getenv("PRLX_SNAPSHOT_DEPTH");
    if (snapshot_depth_str) {
        snapshot_depth = atoi(snapshot_depth_str);
    } else {
        snapshot_depth = PRLX_SNAPSHOT_DEPTH_DEFAULT;
    }

#ifdef PRLX_HAS_ZSTD
    const char* compress_str = getenv("PRLX_COMPRESS");
    compress_enabled = compress_str && atoi(compress_str) != 0;
#endif

    initialized = true;

    if (enabled) {
        fprintf(stderr, "[prlx] Tracing enabled, output: %s\n", output_path);
    }
}

// Pre-launch hook: allocate and set up trace buffer
extern "C" void prlx_pre_launch(const char* kernel_name, dim3 gridDim, dim3 blockDim) {
    if (!initialized) prlx_init();
    if (!enabled) return;

    uint32_t threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    uint32_t warps_per_block = (threads_per_block + 31) / 32;
    uint32_t total_blocks = gridDim.x * gridDim.y * gridDim.z;
    uint32_t total_warps = total_blocks * warps_per_block;

    size_t warp_buffer_size = sizeof(WarpBufferHeader) + events_per_warp * sizeof(TraceEvent);
    trace_buffer_size = sizeof(TraceFileHeader) + total_warps * warp_buffer_size;

    fprintf(stderr, "[prlx] Allocating trace buffer: %zu MB for %u warps\n",
            trace_buffer_size / (1024*1024), total_warps);

    cudaError_t err = cudaMalloc(&d_trace_buffer, trace_buffer_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[prlx] ERROR: cudaMalloc failed: %s\n", cudaGetErrorString(err));
        enabled = false;
        return;
    }

    cudaMemset(d_trace_buffer, 0, trace_buffer_size);

    TraceFileHeader header = {};
    header.magic = PRLX_MAGIC;
    header.version = PRLX_VERSION;
    header.flags = (history_depth > 0) ? PRLX_FLAG_HISTORY : 0;
    if (sample_rate > 1) header.flags |= PRLX_FLAG_SAMPLED;
    header.history_depth = history_depth;
    header.history_section_offset = 0;  // 0 = immediately after event buffers
    header.sample_rate = sample_rate;
    if (snapshot_depth > 0) {
        header.flags |= PRLX_FLAG_SNAPSHOT;
        header.snapshot_depth = snapshot_depth;
        header.snapshot_section_offset = 0;  // 0 = auto (after event + history)
    }
    header.kernel_name_hash = fnv1a_hash(kernel_name);
    strncpy(header.kernel_name, kernel_name, sizeof(header.kernel_name) - 1);
    header.grid_dim[0] = gridDim.x;
    header.grid_dim[1] = gridDim.y;
    header.grid_dim[2] = gridDim.z;
    header.block_dim[0] = blockDim.x;
    header.block_dim[1] = blockDim.y;
    header.block_dim[2] = blockDim.z;
    header.num_warps_per_block = warps_per_block;
    header.total_warp_slots = total_warps;
    header.events_per_warp = events_per_warp;
    header.timestamp = time(nullptr);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    header.cuda_arch = prop.major * 10 + prop.minor;

    cudaMemcpy(d_trace_buffer, &header, sizeof(header), cudaMemcpyHostToDevice);

    // Set the global device variable g_prlx_buffer to point to this buffer
    // This is the key technique to avoid modifying kernel signatures (Death Valley 2)
    // g_prlx_buffer is declared in prlx_runtime.h (included at top)
    err = cudaMemcpyToSymbol(g_prlx_buffer, &d_trace_buffer, sizeof(void*));
    if (err != cudaSuccess) {
        fprintf(stderr, "[prlx] ERROR: cudaMemcpyToSymbol failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_trace_buffer);
        d_trace_buffer = nullptr;
        enabled = false;
        return;
    }

    cudaMemcpyToSymbol(__prlx_sample_rate, &sample_rate, sizeof(uint32_t));
    if (sample_rate > 1) {
        fprintf(stderr, "[prlx] Sampling rate: 1/%u (recording ~%.1f%% of events)\n",
                sample_rate, 100.0f / sample_rate);
    }

    if (history_depth > 0) {
        size_t ring_size = sizeof(HistoryRingHeader) + history_depth * sizeof(HistoryEntry);
        history_buffer_size = total_warps * ring_size;

        err = cudaMalloc(&d_history_buffer, history_buffer_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "[prlx] WARNING: history buffer allocation failed: %s (disabling history)\n",
                    cudaGetErrorString(err));
            d_history_buffer = nullptr;
            history_buffer_size = 0;
            header.flags &= ~PRLX_FLAG_HISTORY;
            header.history_depth = 0;
            cudaMemcpy(d_trace_buffer, &header, sizeof(header), cudaMemcpyHostToDevice);
        } else {
            cudaMemset(d_history_buffer, 0, history_buffer_size);

            // Initialize each ring header with the depth
            void* h_ring_init = malloc(ring_size);
            if (!h_ring_init) {
                fprintf(stderr, "[prlx] ERROR: malloc for history ring init failed\n");
            } else {
                memset(h_ring_init, 0, ring_size);
                ((HistoryRingHeader*)h_ring_init)->depth = history_depth;
                for (uint32_t w = 0; w < total_warps; w++) {
                    cudaMemcpy((char*)d_history_buffer + w * ring_size, h_ring_init,
                               sizeof(HistoryRingHeader), cudaMemcpyHostToDevice);
                }
                free(h_ring_init);
            }

            cudaMemcpyToSymbol(g_prlx_history_buffer, &d_history_buffer, sizeof(void*));
            cudaMemcpyToSymbol(g_prlx_history_depth, &history_depth, sizeof(uint32_t));

            fprintf(stderr, "[prlx] History buffer: %zu KB (%u entries/warp, %u warps)\n",
                    history_buffer_size / 1024, history_depth, total_warps);
        }
    }

    if (snapshot_depth > 0) {
        size_t snap_ring_size = sizeof(SnapshotRingHeader) + snapshot_depth * sizeof(SnapshotEntry);
        snapshot_buffer_size = total_warps * snap_ring_size;

        err = cudaMalloc(&d_snapshot_buffer, snapshot_buffer_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "[prlx] WARNING: snapshot buffer allocation failed: %s (disabling snapshots)\n",
                    cudaGetErrorString(err));
            d_snapshot_buffer = nullptr;
            snapshot_buffer_size = 0;
            header.flags &= ~PRLX_FLAG_SNAPSHOT;
            header.snapshot_depth = 0;
            cudaMemcpy(d_trace_buffer, &header, sizeof(header), cudaMemcpyHostToDevice);
        } else {
            cudaMemset(d_snapshot_buffer, 0, snapshot_buffer_size);

            // Initialize each ring header with the depth
            void* h_snap_init = malloc(snap_ring_size);
            if (!h_snap_init) {
                fprintf(stderr, "[prlx] ERROR: malloc for snapshot ring init failed\n");
            } else {
                memset(h_snap_init, 0, snap_ring_size);
                ((SnapshotRingHeader*)h_snap_init)->depth = snapshot_depth;
                for (uint32_t w = 0; w < total_warps; w++) {
                    cudaMemcpy((char*)d_snapshot_buffer + w * snap_ring_size, h_snap_init,
                               sizeof(SnapshotRingHeader), cudaMemcpyHostToDevice);
                }
                free(h_snap_init);
            }

            cudaMemcpyToSymbol(g_prlx_snapshot_buffer, &d_snapshot_buffer, sizeof(void*));
            cudaMemcpyToSymbol(g_prlx_snapshot_depth, &snapshot_depth, sizeof(uint32_t));

            fprintf(stderr, "[prlx] Snapshot buffer: %zu KB (%u entries/warp, %u warps)\n",
                    snapshot_buffer_size / 1024, snapshot_depth, total_warps);
        }
    }

    fprintf(stderr, "[prlx] Trace buffer ready for kernel: %s\n", kernel_name);
}

// Post-launch hook: copy buffer to host and write to file
extern "C" void prlx_post_launch(void) {
    if (!enabled || !d_trace_buffer) return;

    fprintf(stderr, "[prlx] Copying trace buffer from device...\n");

    void* h_buffer = malloc(trace_buffer_size);
    if (!h_buffer) {
        fprintf(stderr, "[prlx] ERROR: malloc failed\n");
        return;
    }

    cudaError_t err = cudaMemcpy(h_buffer, d_trace_buffer, trace_buffer_size,
                                  cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[prlx] ERROR: cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(h_buffer);
        return;
    }

    // Update warp buffer headers with actual event counts
    TraceFileHeader* header = (TraceFileHeader*)h_buffer;
    WarpBuffer* warps = (WarpBuffer*)((char*)h_buffer + sizeof(TraceFileHeader));

    uint64_t total_events = 0;
    uint64_t total_overflows = 0;

    for (uint32_t i = 0; i < header->total_warp_slots; i++) {
        uint32_t write_idx = warps[i].header.write_idx;
        warps[i].header.num_events = (write_idx < events_per_warp) ? write_idx : events_per_warp;
        total_events += warps[i].header.num_events;
        total_overflows += warps[i].header.overflow_count;
    }

    fprintf(stderr, "[prlx] Recorded %lu events across %u warps (%lu overflows)\n",
            total_events, header->total_warp_slots, total_overflows);

    void* h_history = nullptr;
    if (d_history_buffer && history_buffer_size > 0) {
        h_history = malloc(history_buffer_size);
        if (h_history) {
            err = cudaMemcpy(h_history, d_history_buffer, history_buffer_size,
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                fprintf(stderr, "[prlx] WARNING: history copy failed: %s\n",
                        cudaGetErrorString(err));
                free(h_history);
                h_history = nullptr;
            } else {
                uint64_t total_hist = 0;
                size_t ring_size = sizeof(HistoryRingHeader)
                                 + history_depth * sizeof(HistoryEntry);
                for (uint32_t i = 0; i < header->total_warp_slots; i++) {
                    HistoryRingHeader* ring = (HistoryRingHeader*)((char*)h_history + i * ring_size);
                    total_hist += (ring->total_writes < history_depth)
                                ? ring->total_writes : history_depth;
                }
                fprintf(stderr, "[prlx] History: %lu entries across %u warps\n",
                        total_hist, header->total_warp_slots);
            }
        }
    }

    void* h_snapshot = nullptr;
    if (d_snapshot_buffer && snapshot_buffer_size > 0) {
        h_snapshot = malloc(snapshot_buffer_size);
        if (h_snapshot) {
            err = cudaMemcpy(h_snapshot, d_snapshot_buffer, snapshot_buffer_size,
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                fprintf(stderr, "[prlx] WARNING: snapshot copy failed: %s\n",
                        cudaGetErrorString(err));
                free(h_snapshot);
                h_snapshot = nullptr;
            } else {
                uint64_t total_snaps = 0;
                size_t snap_ring_size = sizeof(SnapshotRingHeader)
                                      + snapshot_depth * sizeof(SnapshotEntry);
                for (uint32_t i = 0; i < header->total_warp_slots; i++) {
                    SnapshotRingHeader* ring = (SnapshotRingHeader*)((char*)h_snapshot + i * snap_ring_size);
                    total_snaps += (ring->total_writes < snapshot_depth)
                                 ? ring->total_writes : snapshot_depth;
                }
                fprintf(stderr, "[prlx] Snapshots: %lu entries across %u warps\n",
                        total_snaps, header->total_warp_slots);
            }
        }
    }

    // Determine output path (session mode or single-file mode)
    char effective_path[512];
    if (session_active && session_entry_count >= PRLX_MAX_SESSION_ENTRIES) {
        fprintf(stderr, "[prlx] WARNING: Session entry limit reached (%u). "
                "Trace will be written to '%s' instead of session directory.\n",
                PRLX_MAX_SESSION_ENTRIES, output_path);
    }

    if (session_active && session_entry_count < PRLX_MAX_SESSION_ENTRIES) {
        // Session mode: write to session_dir/kernel_N.prlx
        snprintf(effective_path, sizeof(effective_path), "%s/%s_%u.prlx",
                 session_dir, header->kernel_name, session_launch_count);

        SessionEntry& entry = session_entries[session_entry_count];
        strncpy(entry.kernel_name, header->kernel_name, sizeof(entry.kernel_name) - 1);
        strncpy(entry.filename, effective_path, sizeof(entry.filename) - 1);
        entry.launch_idx = session_launch_count;
        memcpy(entry.grid_dim, header->grid_dim, sizeof(entry.grid_dim));
        memcpy(entry.block_dim, header->block_dim, sizeof(entry.block_dim));
        session_entry_count++;
        session_launch_count++;
    } else {
        // Re-read PRLX_TRACE here so it can change between launches in the same
        // process (e.g. capturing run A then run B from one Python session).
        const char* env_path = getenv("PRLX_TRACE");
        const char* path = (env_path && env_path[0]) ? env_path : output_path;
        strncpy(effective_path, path, sizeof(effective_path) - 1);
        effective_path[sizeof(effective_path) - 1] = '\0';
    }

    // Build contiguous payload (everything after header) for potential compression
    size_t payload_size = trace_buffer_size - sizeof(TraceFileHeader);
    if (h_history) payload_size += history_buffer_size;
    if (h_snapshot) payload_size += snapshot_buffer_size;

    FILE* f = fopen(effective_path, "wb");
    if (!f) {
        fprintf(stderr, "[prlx] ERROR: cannot open output file: %s\n", effective_path);
        free(h_buffer);
        if (h_history) free(h_history);
        if (h_snapshot) free(h_snapshot);
        return;
    }

    size_t total_written = 0;

#ifdef PRLX_HAS_ZSTD
    if (compress_enabled && payload_size > 0) {
        header->flags |= PRLX_FLAG_COMPRESS;

        // Write header (uncompressed, 160 bytes)
        size_t hdr_w = fwrite(header, 1, sizeof(TraceFileHeader), f);
        if (hdr_w != sizeof(TraceFileHeader)) {
            fprintf(stderr, "[prlx] ERROR: incomplete header write (%zu of %zu bytes)\n",
                    hdr_w, sizeof(TraceFileHeader));
        }
        total_written += hdr_w;

        void* payload = malloc(payload_size);
        if (payload) {
            size_t off = 0;
            size_t warp_data_size = trace_buffer_size - sizeof(TraceFileHeader);
            memcpy((char*)payload + off, (char*)h_buffer + sizeof(TraceFileHeader), warp_data_size);
            off += warp_data_size;
            if (h_history) {
                memcpy((char*)payload + off, h_history, history_buffer_size);
                off += history_buffer_size;
            }
            if (h_snapshot) {
                memcpy((char*)payload + off, h_snapshot, snapshot_buffer_size);
            }

            // Compress
            size_t comp_bound = ZSTD_compressBound(payload_size);
            void* comp_buf = malloc(comp_bound);
            if (comp_buf) {
                size_t comp_size = ZSTD_compress(comp_buf, comp_bound, payload, payload_size, 3);
                if (!ZSTD_isError(comp_size)) {
                    size_t comp_w = fwrite(comp_buf, 1, comp_size, f);
                    if (comp_w != comp_size) {
                        fprintf(stderr, "[prlx] ERROR: incomplete compressed write (%zu of %zu bytes)\n",
                                comp_w, comp_size);
                    }
                    total_written += comp_w;
                    fprintf(stderr, "[prlx] Compressed: %zu -> %zu bytes (%.1fx)\n",
                            payload_size, comp_size, (double)payload_size / comp_size);
                } else {
                    fprintf(stderr, "[prlx] WARNING: zstd compression failed: %s, writing uncompressed\n",
                            ZSTD_getErrorName(comp_size));
                    header->flags &= ~PRLX_FLAG_COMPRESS;
                    // Rewrite header without compress flag, then payload
                    fseek(f, 0, SEEK_SET);
                    size_t hdr_fb = fwrite(header, 1, sizeof(TraceFileHeader), f);
                    size_t pay_fb = fwrite(payload, 1, payload_size, f);
                    if (hdr_fb != sizeof(TraceFileHeader) || pay_fb != payload_size) {
                        fprintf(stderr, "[prlx] ERROR: incomplete fallback write\n");
                    }
                    total_written = hdr_fb + pay_fb;
                }
                free(comp_buf);
            }
            free(payload);
        }
    } else
#endif
    {
        // Uncompressed path
        size_t written = fwrite(h_buffer, 1, trace_buffer_size, f);
        total_written = written;

        if (h_history && written == trace_buffer_size) {
            size_t hist_written = fwrite(h_history, 1, history_buffer_size, f);
            total_written += hist_written;
            if (hist_written != history_buffer_size) {
                fprintf(stderr, "[prlx] WARNING: incomplete history write (%zu of %zu bytes)\n",
                        hist_written, history_buffer_size);
            }
        }

        if (h_snapshot) {
            size_t snap_written = fwrite(h_snapshot, 1, snapshot_buffer_size, f);
            total_written += snap_written;
            if (snap_written != snapshot_buffer_size) {
                fprintf(stderr, "[prlx] WARNING: incomplete snapshot write (%zu of %zu bytes)\n",
                        snap_written, snapshot_buffer_size);
            }
        }

        if (written != trace_buffer_size) {
            fprintf(stderr, "[prlx] ERROR: incomplete write (%zu of %zu bytes)\n",
                    written, trace_buffer_size);
        }
    }

    fclose(f);
    fprintf(stderr, "[prlx] Trace written to: %s (%zu KB)\n",
            effective_path, total_written / 1024);

    free(h_buffer);
    if (h_history) free(h_history);
    if (h_snapshot) free(h_snapshot);

    cudaFree(d_trace_buffer);
    d_trace_buffer = nullptr;

    if (d_history_buffer) {
        cudaFree(d_history_buffer);
        d_history_buffer = nullptr;
        history_buffer_size = 0;

        void* null_ptr = nullptr;
        uint32_t zero = 0;
        cudaMemcpyToSymbol(g_prlx_history_buffer, &null_ptr, sizeof(void*));
        cudaMemcpyToSymbol(g_prlx_history_depth, &zero, sizeof(uint32_t));
    }

    if (d_snapshot_buffer) {
        cudaFree(d_snapshot_buffer);
        d_snapshot_buffer = nullptr;
        snapshot_buffer_size = 0;
    }

    {
        void* null_ptr = nullptr;
        uint32_t zero = 0;
        cudaMemcpyToSymbol(g_prlx_snapshot_buffer, &null_ptr, sizeof(void*));
        cudaMemcpyToSymbol(g_prlx_snapshot_depth, &zero, sizeof(uint32_t));
        cudaMemcpyToSymbol(g_prlx_buffer, &null_ptr, sizeof(void*));
    }
}

// Session API: begin capturing multiple kernel launches
extern "C" void prlx_session_begin(const char* name) {
    if (!initialized) prlx_init();

    snprintf(session_dir, sizeof(session_dir), "%s", name);

    if (mkdir(session_dir, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "[prlx] Warning: failed to create session directory: %s\n", session_dir);
    }

    session_active = true;
    session_launch_count = 0;
    session_entry_count = 0;
    fprintf(stderr, "[prlx] Session started: %s\n", session_dir);
}

// Escape a string for JSON output (handles backslash, quote, control chars)
static void fprint_json_string(FILE* f, const char* str) {
    fputc('"', f);
    for (const char* p = str; *p; p++) {
        switch (*p) {
            case '\\': fputs("\\\\", f); break;
            case '"':  fputs("\\\"", f); break;
            case '\n': fputs("\\n", f);  break;
            case '\r': fputs("\\r", f);  break;
            case '\t': fputs("\\t", f);  break;
            default:   fputc(*p, f);     break;
        }
    }
    fputc('"', f);
}

// Session API: end session and write manifest
extern "C" void prlx_session_end(void) {
    if (!session_active) return;

    char manifest_path[512];
    snprintf(manifest_path, sizeof(manifest_path), "%s/session.json", session_dir);

    FILE* f = fopen(manifest_path, "w");
    if (f) {
        fprintf(f, "[\n");
        for (uint32_t i = 0; i < session_entry_count; i++) {
            const SessionEntry& e = session_entries[i];
            fprintf(f, "  {\n");
            fprintf(f, "    \"launch\": %u,\n", e.launch_idx);
            fprintf(f, "    \"kernel\": "); fprint_json_string(f, e.kernel_name); fprintf(f, ",\n");
            fprintf(f, "    \"file\": "); fprint_json_string(f, e.filename); fprintf(f, ",\n");
            fprintf(f, "    \"grid\": [%u, %u, %u],\n", e.grid_dim[0], e.grid_dim[1], e.grid_dim[2]);
            fprintf(f, "    \"block\": [%u, %u, %u]\n", e.block_dim[0], e.block_dim[1], e.block_dim[2]);
            fprintf(f, "  }%s\n", (i < session_entry_count - 1) ? "," : "");
        }
        fprintf(f, "]\n");
        fclose(f);
        fprintf(stderr, "[prlx] Session manifest written: %s (%u launches)\n",
                manifest_path, session_entry_count);
    } else {
        fprintf(stderr, "[prlx] ERROR: cannot write session manifest: %s\n", manifest_path);
    }

    session_active = false;
    session_launch_count = 0;
    session_entry_count = 0;
}

// Getters for Python FFI — expose device pointers so the module binder
// can write them into Triton JIT-compiled CUmodules via CUDA Driver API.
extern "C" void* prlx_get_trace_buffer(void)      { return d_trace_buffer; }
extern "C" void* prlx_get_history_buffer(void)     { return d_history_buffer; }
extern "C" uint32_t prlx_get_history_depth(void)   { return history_depth; }
extern "C" uint32_t prlx_get_sample_rate(void)     { return sample_rate; }
extern "C" void* prlx_get_snapshot_buffer(void)    { return d_snapshot_buffer; }
extern "C" uint32_t prlx_get_snapshot_depth(void)  { return snapshot_depth; }

// Cleanup
extern "C" void prlx_shutdown(void) {
    if (d_trace_buffer) {
        cudaFree(d_trace_buffer);
        d_trace_buffer = nullptr;
    }
    if (d_history_buffer) {
        cudaFree(d_history_buffer);
        d_history_buffer = nullptr;
        history_buffer_size = 0;
    }
    if (d_snapshot_buffer) {
        cudaFree(d_snapshot_buffer);
        d_snapshot_buffer = nullptr;
        snapshot_buffer_size = 0;
    }
}

// Constructor to auto-initialize
__attribute__((constructor))
static void auto_init() {
    prlx_init();
}

// Destructor to auto-cleanup
__attribute__((destructor))
static void auto_shutdown() {
    prlx_shutdown();
}
