// ipc_core.c — compiled shared memory backend for bbos IPC
// Build: gcc -shared -fPIC -O2 -o ipc_core.so ipc_core.c -lrt -lm

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <signal.h>
#include <errno.h>
#include <sched.h>
#include <math.h>

#define META_SIZE 4096

// ── helpers ──

static inline int64_t floor_div(int64_t a, int64_t b) {
    return a / b - (a % b != 0 && (a ^ b) < 0);
}

static inline int64_t floor_mod(int64_t a, int64_t b) {
    int64_t r = a % b;
    return r < 0 ? r + b : r;
}

// Reader index: ((ridx // 2) - 1) % N  (Python floor semantics)
static inline int reader_idx(int64_t ridx, int64_t N) {
    return (int)floor_mod(floor_div(ridx, 2) - 1, N);
}

// Writer index: (seq / 2) % N  (seq always >= 0)
static inline int writer_idx(uint32_t seq, uint32_t N) {
    return (seq / 2) % N;
}

// ── atomic seq ──

static inline uint32_t seq_load(void *map) {
    return __atomic_load_n((volatile uint32_t *)map, __ATOMIC_ACQUIRE);
}

static inline void seq_store(void *map, uint32_t v) {
    __atomic_store_n((volatile uint32_t *)map, v, __ATOMIC_RELEASE);
}

// ── writer ──

// Create or reclaim shared memory for a writer.
// Returns mmap pointer on success, NULL on failure.
// On failure with a live writer, *err is set to the existing writer's pid (>0).
// On other failures, *err is set to -1.
void *ipc_writer_create(const char *name, const char *meta, int meta_len,
                         int total_size, int N, int pid, int *err) {
    int fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0644);
    if (fd < 0 && errno == EEXIST) {
        // Existing shm — check if writer is alive
        int old_fd = shm_open(name, O_RDWR, 0);
        if (old_fd < 0) { *err = -1; return NULL; }
        struct stat st;
        fstat(old_fd, &st);
        void *old_map = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, old_fd, 0);
        close(old_fd);
        if (old_map == MAP_FAILED) { *err = -1; return NULL; }
        int old_pid = *(int32_t *)((char *)old_map + 8);
        munmap(old_map, st.st_size);
        if (old_pid > 0 && kill(old_pid, 0) == 0) {
            *err = old_pid;  // writer alive
            return NULL;
        }
        shm_unlink(name);
        fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0644);
    }
    if (fd < 0) { *err = -1; return NULL; }

    ftruncate(fd, total_size);
    void *map = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { *err = -1; return NULL; }

    // Zero and write header: seq(4) N(4) pid(4) meta_len(4) meta(...)
    memset(map, 0, total_size);
    *(uint32_t *)map = 0;
    *(uint32_t *)((char *)map + 4) = (uint32_t)N;
    *(int32_t *)((char *)map + 8) = pid;
    *(uint32_t *)((char *)map + 12) = (uint32_t)meta_len;
    memcpy((char *)map + 16, meta, meta_len);
    msync(map, total_size, MS_SYNC);

    *err = 0;
    return map;
}

void ipc_writer_unlink(const char *name) {
    shm_unlink(name);
}

void ipc_munmap(void *map, int size) {
    if (map) munmap(map, size);
}

// Increment seq by 1 with release semantics
void ipc_seq_inc(void *map) {
    uint32_t v = *(volatile uint32_t *)map;
    seq_store(map, v + 1);
}

uint32_t ipc_seq_read(void *map) {
    return seq_load(map);
}

int ipc_writer_idx(void *map) {
    uint32_t s = *(volatile uint32_t *)map;
    uint32_t N = *(uint32_t *)((char *)map + 4);
    return writer_idx(s, N);
}

// ── reader ──

// Open existing shared memory for reading.
// Returns mmap pointer on success, NULL on failure.
// out_meta must be at least META_SIZE bytes.
// *err: 0=ok, -1=shm not found or mmap fail, -2=no metadata yet
void *ipc_reader_open(const char *name, int *out_size, int *out_pid,
                       char *out_meta, int *out_meta_len, int *err) {
    int fd = shm_open(name, O_RDONLY, 0);
    if (fd < 0) { *err = -1; return NULL; }

    struct stat st;
    fstat(fd, &st);
    void *map = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { *err = -1; return NULL; }

    *out_size = (int)st.st_size;
    *out_pid = *(int32_t *)((char *)map + 8);

    int ml = (int)*(uint32_t *)((char *)map + 12);
    if (ml == 0) {
        munmap(map, st.st_size);
        *err = -2;
        return NULL;
    }
    memcpy(out_meta, (char *)map + 16, ml);
    *out_meta_len = ml;
    *err = 0;
    return map;
}

int ipc_check_pid(int pid) {
    if (pid <= 0) return 1;
    return kill(pid, 0) == 0 ? 1 : 0;
}

// Current inode of a topic's shm segment, or 0 if it doesn't exist. Lets a reader detect that the
// writer closed + recreated the segment (new inode) even when the writer PID is unchanged/alive —
// the kill()-based liveness check alone can't see that, so the reader would stay stuck on the
// deleted segment until the writer PID actually dies.
unsigned long ipc_inode(const char *name) {
    int fd = shm_open(name, O_RDONLY, 0);
    if (fd < 0) return 0;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return 0; }
    close(fd);
    return (unsigned long)st.st_ino;
}

// ── reader spin-read ──

typedef struct {
    int64_t ridx;
    int64_t last_seq;
    int32_t dropped;
} read_result_t;

// Perform one lock-free read from the ring buffer.
// last_seq: pass -1 on first call (will be initialized to current seq).
// align_ts: nanosecond timestamp for alignment mode, or -1 to disable.
// Copies one item_size record into out_data.
// Returns 0 on success, -1 if the writer died mid-publish (seq stuck odd).
int ipc_reader_read(void *map, int item_size, int ts_offset, int sync, int decimate,
                     int64_t last_seq, int64_t align_ts,
                     void *out_data, read_result_t *result) {
    char *data_base = (char *)map + META_SIZE;
    int odd_spins = 0;

    for (;;) {
        uint32_t s0 = seq_load(map);
        if (s0 & 1) {
            // Writer is mid-publish (normally microseconds). If the writer is killed
            // between its two seq increments the counter stays odd forever and a reader
            // would spin here for eternity. After ~1k consecutive odd sightings (~1ms),
            // check the writer pid; if it's gone, bail so the caller can disconnect.
            if (++odd_spins >= 1024) {
                if (!ipc_check_pid(*(int32_t *)((char *)map + 8))) return -1;
                odd_spins = 0;  // writer alive, just a long write — keep waiting
            }
            sched_yield();
            continue;
        }
        odd_spins = 0;
        if (seq_load(map) != s0) continue;

        int64_t N = (int64_t)*(uint32_t *)((char *)map + 4);
        int64_t s = (int64_t)s0;

        if (last_seq < 0) last_seq = s;

        int64_t oldest = s - 2 * (N - 1);
        int64_t ridx;

        if (align_ts >= 0) {
            int li = reader_idx(s, N);
            int oi = reader_idx(oldest, N);
            int64_t lts = *(int64_t *)(data_base + li * item_size + ts_offset);
            int64_t ots = *(int64_t *)(data_base + oi * item_size + ts_offset);
            if (align_ts < ots) {
                ridx = oldest;
            } else if (align_ts > lts) {
                ridx = s;
            } else {
                double interp = (lts == ots) ? 0.0
                    : (double)(align_ts - ots) / (double)(lts - ots);
                if (interp < 0.0) interp = 0.0;
                ridx = llround(interp * (double)(s - oldest) + (double)oldest);
            }
        } else if (sync) {
            int64_t target = last_seq + 2 * (int64_t)decimate;
            ridx = target < oldest ? oldest : (target > s ? s : target);
        } else {
            ridx = s;
        }

        int di = reader_idx(ridx, N);
        memcpy(out_data, data_base + di * item_size, item_size);

        if (seq_load(map) == s0) {
            int64_t backlog = (s - last_seq) / 2;
            int64_t dropped = backlog - N;
            result->ridx = ridx;
            result->last_seq = ridx;
            result->dropped = (int32_t)(dropped > 0 ? dropped : 0);
            return 0;
        }
    }
}
