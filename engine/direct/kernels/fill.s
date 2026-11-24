; fill: writes a value to every dword of a buffer. The direct path's first
; smoke test: if the buffer reads back filled, dispatch, arguments and
; memory all work. 256 threads per workgroup, one dword each.
;
; Arguments: s[0:1] buffer, s2 value, s3 dword count.  s4 = workgroup id x
.kernel fill
.wave32
.user_sgprs 4
.tgid_x
    s_lshl_b32 s5, s4, 8
    v_add_nc_u32 v1, s5, v0
    v_cmp_gt_u32 vcc_lo, s3, v1          ; only threads inside the buffer
    s_and_saveexec_b32 s6, vcc_lo
    s_cbranch_execz done
    v_lshlrev_b32 v1, 2, v1
    v_mov_b32 v2, s2
    global_store_dword v1, v2, s[0:1]
done:
    s_endpgm
