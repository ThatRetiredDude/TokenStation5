; read_bw: streams a buffer through the GPU as fast as it can and sums it,
; to measure read bandwidth on the direct path (ts5 peak's Vulkan read gets
; ~429 GB/s on the PS5). 256 threads (8 waves of 32) per workgroup; every
; thread reads 16 bytes per load, four loads per loop.
;
; Arguments (user SGPRs):
;   s[0:1]  source address            s2  loops (4 loads each)
;   s3      bytes between a thread's loads (= all threads x 16)
;   s[4:5]  output: one dword per thread (the sum, so nothing is skipped)
; s6 = workgroup id x
.kernel read_bw
.wave32
.user_sgprs 6
.tgid_x
    s_lshl_b32 s7, s6, 8                 ; first thread of this workgroup
    v_add_nc_u32 v1, s7, v0              ; global thread id
    v_lshlrev_b32 v2, 4, v1              ; byte offset of its first load
    v_mov_b32 v20, 0
    v_mov_b32 v21, 0
    v_mov_b32 v22, 0
    v_mov_b32 v23, 0
    s_mov_b32 s8, s0
    s_mov_b32 s9, s1
loop:
    global_load_dwordx4 v[4:7], v2, s[8:9]
    s_add_u32 s8, s8, s3
    s_addc_u32 s9, s9, 0
    global_load_dwordx4 v[8:11], v2, s[8:9]
    s_add_u32 s8, s8, s3
    s_addc_u32 s9, s9, 0
    global_load_dwordx4 v[12:15], v2, s[8:9]
    s_add_u32 s8, s8, s3
    s_addc_u32 s9, s9, 0
    global_load_dwordx4 v[16:19], v2, s[8:9]
    s_add_u32 s8, s8, s3
    s_addc_u32 s9, s9, 0
    s_sub_u32 s2, s2, 1
    s_waitcnt vmcnt(3)
    v_add_nc_u32 v20, v20, v4
    v_add_nc_u32 v21, v21, v5
    v_add_nc_u32 v22, v22, v6
    v_add_nc_u32 v23, v23, v7
    s_waitcnt vmcnt(2)
    v_add_nc_u32 v20, v20, v8
    v_add_nc_u32 v21, v21, v9
    v_add_nc_u32 v22, v22, v10
    v_add_nc_u32 v23, v23, v11
    s_waitcnt vmcnt(1)
    v_add_nc_u32 v20, v20, v12
    v_add_nc_u32 v21, v21, v13
    v_add_nc_u32 v22, v22, v14
    v_add_nc_u32 v23, v23, v15
    s_waitcnt vmcnt(0)
    v_add_nc_u32 v20, v20, v16
    v_add_nc_u32 v21, v21, v17
    v_add_nc_u32 v22, v22, v18
    v_add_nc_u32 v23, v23, v19
    s_cmp_lg_u32 s2, 0
    s_cbranch_scc1 loop
    v_add3_u32 v20, v20, v21, v22
    v_add_nc_u32 v20, v20, v23
    v_lshlrev_b32 v1, 2, v1
    global_store_dword v1, v20, s[4:5]
    s_endpgm
