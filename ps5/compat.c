/* libc functions the title needs that no system module gives a title.
 * ps5/check-imports.py names any that are missing: each would otherwise be an
 * import that resolves to NULL on the console. */
#include <errno.h>

/* RADV asks whether a path is a symbolic link (ac_gpu_info.c); the console's
 * file systems give a title none. */
#include <stddef.h>
#include <sys/types.h>

ssize_t readlink(const char *restrict path, char *restrict buf, size_t size) {
    (void)path;
    (void)buf;
    (void)size;
    errno = EINVAL;
    return -1;
}

/* ACO's disassembly path makes a temporary file. Six X's replaced from a
 * counter and the clock until the name is new. */
#include <fcntl.h>
#include <string.h>
#include <time.h>

int mkstemp(char *path) {
    static const char alphabet[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    static unsigned long counter;
    const size_t n = strlen(path);
    if (n < 6 || strcmp(path + n - 6, "XXXXXX") != 0) {
        errno = EINVAL;
        return -1;
    }
    for (int attempt = 0; attempt < 1000; attempt++) {
        unsigned long v = (unsigned long)time(NULL) ^ (++counter * 2654435761ul);
        for (int i = 0; i < 6; i++, v /= 36) path[n - 6 + i] = alphabet[v % 36];
        const int fd = open(path, O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd >= 0 || errno != EEXIST) return fd;
    }
    errno = EEXIST;
    return -1;
}

/* whisper.cpp's ggml, on an error it can't go on from, forks a debugger to
 * print where (ggml_print_backtrace). A title can't start a process: fork is
 * only in libkernel_sys and no module has execlp. Both fail here, and ggml
 * skips the backtrace. */
#include <unistd.h>

pid_t fork(void) {
    errno = ENOSYS;
    return -1;
}

int execlp(const char *file, const char *arg, ...) {
    (void)file;
    (void)arg;
    errno = ENOSYS;
    return -1;
}

/* ggml finds its backends with std::filesystem, and libc++'s file system
 * operations come as one piece, links and all; only libkernel_sys has these
 * three. ggml looks in folders only when asked to load backends from files,
 * which the title never does: its CPU backend is linked in. A path is as long
 * as the system's own limit; links can't be made. */
#include <limits.h>

long pathconf(const char *path, int name) {
    (void)path;
    if (name == _PC_PATH_MAX) return PATH_MAX;
    if (name == _PC_NAME_MAX) return NAME_MAX;
    errno = EINVAL;
    return -1;
}

int link(const char *from, const char *to) {
    (void)from;
    (void)to;
    errno = EPERM;
    return -1;
}

int symlink(const char *target, const char *path) {
    (void)target;
    (void)path;
    errno = EPERM;
    return -1;
}

/* kokoro.cpp's model loader advises the kernel on how it will read a mapping
 * (third_party/kokoro.cpp's gguf_loader.cpp). The title reads its models
 * with stdio (KOKORO_NO_MMAP), and advice may always be ignored. */
#include <sys/mman.h>

int posix_madvise(void *addr, size_t len, int advice) {
    (void)addr;
    (void)len;
    (void)advice;
    return 0;
}
