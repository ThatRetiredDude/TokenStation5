; copy: dst = src, 16 bytes per thread, 256 threads per workgroup. Puts
; data the CPU wrote (system memory) into VRAM, which the CPU can't map.
;
; Arguments: s[0:1] src, s[2:3] dst, s4 count of 16-byte units.  s5 = workgroup id x
.kernel copy
.wave32
.user_sgprs 5
.tgid_x
    s_lshl_b32 s6, s5, 8
    v_add_nc_u32 v1, s6, v0
    v_cmp_gt_u32 vcc_lo, s4, v1
    s_and_saveexec_b32 s7, vcc_lo
    s_cbranch_execz done
    v_mov_b32 v2, 0
    v_lshlrev_b64 v[1:2], 4, v[1:2]
    v_add_co_u32 v3, vcc_lo, s0, v1
    v_add_co_ci_u32 v4, vcc_lo, s1, v2, vcc_lo
    global_load_dwordx4 v[8:11], v[3:4], off
    v_add_co_u32 v5, vcc_lo, s2, v1
    v_add_co_ci_u32 v6, vcc_lo, s3, v2, vcc_lo
    s_waitcnt vmcnt(0)
    global_store_dwordx4 v[5:6], v[8:11], off
done:
    s_endpgm
