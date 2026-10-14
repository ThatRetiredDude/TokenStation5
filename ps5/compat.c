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
