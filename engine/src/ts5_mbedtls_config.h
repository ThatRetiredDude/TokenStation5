// ts5's changes to Mbed TLS's default configuration (MBEDTLS_USER_CONFIG_FILE):
// a TLS client on sockets ts5 opens itself, on Linux and in the PS5 title.

// Randomness comes from mbedtls_hardware_poll in net.cpp (the kernel's
// generator on Linux, sceRandomGetRandomNumber on the PS5), not from Mbed
// TLS's own platform probing.
#define MBEDTLS_NO_PLATFORM_ENTROPY
#define MBEDTLS_ENTROPY_HARDWARE_ALT
// UTC dates for certificate validity, without libc's gmtime_r (the PS5's
// system libc has no export of it): mbedtls_platform_gmtime_r in net.cpp.
#define MBEDTLS_PLATFORM_GMTIME_R_ALT
// net.cpp does its own sockets; nothing is stored on disk.
#undef MBEDTLS_NET_C
#undef MBEDTLS_TIMING_C
#undef MBEDTLS_PSA_CRYPTO_STORAGE_C
#undef MBEDTLS_PSA_ITS_FILE_C
