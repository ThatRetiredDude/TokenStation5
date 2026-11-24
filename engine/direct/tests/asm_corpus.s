; Every instruction form the direct engine's assembler knows, checked byte for
; byte against llvm-mc by asm_check.py. One instruction per line.
s_add_u32 s0, s1, s2
s_sub_u32 s0, s1, s2
s_add_i32 s0, s1, s2
s_sub_i32 s0, s1, s2
s_addc_u32 s0, s1, s2
s_subb_u32 s0, s1, s2
s_min_i32 s0, s1, s2
s_min_u32 s0, s1, s2
s_max_i32 s0, s1, s2
s_max_u32 s0, s1, s2
s_cselect_b32 s0, s1, s2
s_cselect_b64 s[0:1], s[2:3], s[4:5]
s_and_b32 s0, s1, s2
s_and_b64 s[0:1], s[2:3], s[4:5]
s_or_b32 s0, s1, s2
s_or_b64 s[0:1], s[2:3], s[4:5]
s_xor_b32 s0, s1, s2
s_xor_b64 s[0:1], s[2:3], s[4:5]
s_andn2_b32 s0, s1, s2
s_andn2_b64 s[0:1], s[2:3], s[4:5]
s_orn2_b32 s0, s1, s2
s_nand_b32 s0, s1, s2
s_nor_b32 s0, s1, s2
s_xnor_b32 s0, s1, s2
s_lshl_b32 s0, s1, s2
s_lshl_b64 s[0:1], s[2:3], s2
s_lshr_b32 s0, s1, s2
s_lshr_b64 s[0:1], s[2:3], s2
s_ashr_i32 s0, s1, s2
s_ashr_i64 s[0:1], s[2:3], s2
s_bfm_b32 s0, s1, s2
s_mul_i32 s0, s1, s2
s_bfe_u32 s0, s1, s2
s_bfe_i32 s0, s1, s2
s_bfe_u64 s[0:1], s[2:3], s4
s_absdiff_i32 s0, s1, s2
s_lshl1_add_u32 s0, s1, s2
s_lshl2_add_u32 s0, s1, s2
s_lshl3_add_u32 s0, s1, s2
s_lshl4_add_u32 s0, s1, s2
s_pack_ll_b32_b16 s0, s1, s2
s_pack_lh_b32_b16 s0, s1, s2
s_pack_hh_b32_b16 s0, s1, s2
s_mul_hi_u32 s0, s1, s2
s_mul_hi_i32 s0, s1, s2
s_mov_b32 s0, s1
s_mov_b64 s[0:1], s[2:3]
s_cmov_b32 s0, s1
s_cmov_b64 s[0:1], s[2:3]
s_not_b32 s0, s1
s_not_b64 s[0:1], s[2:3]
s_brev_b32 s0, s1
s_bcnt1_i32_b32 s0, s1
s_bcnt1_i32_b64 s0, s[2:3]
s_ff1_i32_b32 s0, s1
s_ff1_i32_b64 s0, s[2:3]
s_flbit_i32_b32 s0, s1
s_sext_i32_i8 s0, s1
s_sext_i32_i16 s0, s1
s_bitset0_b32 s0, s1
s_bitset1_b32 s0, s1
s_getpc_b64 s[0:1]
s_setpc_b64 s[0:1]
s_swappc_b64 s[0:1], s[2:3]
s_and_saveexec_b32 s0, s1
s_or_saveexec_b32 s0, s1
s_xor_saveexec_b32 s0, s1
s_andn2_saveexec_b32 s0, s1
s_and_saveexec_b64 s[0:1], s[2:3]
s_or_saveexec_b64 s[0:1], s[2:3]
s_abs_i32 s0, s1
s_movk_i32 s0, 0x1234
s_cmovk_i32 s0, 0x1234
s_addk_i32 s0, 0x1234
s_mulk_i32 s0, 0x1234
s_cmpk_eq_u32 s0, 0x1234
s_cmpk_lg_u32 s0, 0x1234
s_cmpk_gt_u32 s0, 0x1234
s_cmpk_lt_u32 s0, 0x1234
s_getreg_b32 s0, hwreg(HW_REG_MODE)
s_setreg_b32 hwreg(HW_REG_MODE), s0
s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 1), 1
s_waitcnt_vscnt null, 0x0
s_waitcnt_vmcnt null, 0x0
s_waitcnt_lgkmcnt null, 0x0
s_cmp_eq_i32 s0, s1
s_cmp_lg_i32 s0, s1
s_cmp_gt_i32 s0, s1
s_cmp_ge_i32 s0, s1
s_cmp_lt_i32 s0, s1
s_cmp_le_i32 s0, s1
s_cmp_eq_u32 s0, s1
s_cmp_lg_u32 s0, s1
s_cmp_gt_u32 s0, s1
s_cmp_ge_u32 s0, s1
s_cmp_lt_u32 s0, s1
s_cmp_le_u32 s0, s1
s_bitcmp0_b32 s0, s1
s_bitcmp1_b32 s0, s1
s_cmp_eq_u64 s[0:1], s[2:3]
s_cmp_lg_u64 s[0:1], s[2:3]
s_nop 0
s_endpgm
s_branch 0
s_cbranch_scc0 0
s_cbranch_scc1 0
s_cbranch_vccz 0
s_cbranch_vccnz 0
s_cbranch_execz 0
s_cbranch_execnz 0
s_barrier
s_waitcnt 0
s_sleep 0
s_setprio 0
s_sendmsg 0
s_icache_inv
s_code_end
s_inst_prefetch 0x0
s_round_mode 0x0
s_denorm_mode 0
s_ttracedata
s_load_dword s0, s[2:3], 0x0
s_load_dwordx2 s[0:1], s[2:3], 0x0
s_load_dwordx4 s[0:3], s[2:3], 0x0
s_load_dwordx8 s[0:7], s[2:3], 0x0
s_load_dwordx16 s[0:15], s[2:3], 0x0
s_buffer_load_dword s0, s[4:7], 0x0
s_buffer_load_dwordx2 s[0:1], s[4:7], 0x0
s_buffer_load_dwordx4 s[0:3], s[4:7], 0x0
s_buffer_load_dwordx8 s[0:7], s[4:7], 0x0
s_memtime s[0:1]
s_memrealtime s[0:1]
s_dcache_inv
s_gl1_inv
v_nop
v_mov_b32 v0, v1
v_readfirstlane_b32 s0, v1
v_cvt_f32_i32 v0, v1
v_cvt_f32_u32 v0, v1
v_cvt_u32_f32 v0, v1
v_cvt_i32_f32 v0, v1
v_cvt_f16_f32 v0, v1
v_cvt_f32_f16 v0, v1
v_cvt_rpi_i32_f32 v0, v1
v_cvt_flr_i32_f32 v0, v1
v_cvt_f32_ubyte0 v0, v1
v_cvt_f32_ubyte1 v0, v1
v_cvt_f32_ubyte2 v0, v1
v_cvt_f32_ubyte3 v0, v1
v_fract_f32 v0, v1
v_trunc_f32 v0, v1
v_ceil_f32 v0, v1
v_rndne_f32 v0, v1
v_floor_f32 v0, v1
v_exp_f32 v0, v1
v_log_f32 v0, v1
v_rcp_f32 v0, v1
v_rcp_iflag_f32 v0, v1
v_rsq_f32 v0, v1
v_sqrt_f32 v0, v1
v_sin_f32 v0, v1
v_cos_f32 v0, v1
v_not_b32 v0, v1
v_bfrev_b32 v0, v1
v_ffbh_u32 v0, v1
v_ffbl_b32 v0, v1
v_ffbh_i32 v0, v1
v_movreld_b32 v0, v1
v_movrels_b32 v0, v1
v_cvt_f16_u16 v0, v1
v_cvt_f16_i16 v0, v1
v_cvt_u16_f16 v0, v1
v_rcp_f16 v0, v1
v_sqrt_f16 v0, v1
v_rsq_f16 v0, v1
v_log_f16 v0, v1
v_exp_f16 v0, v1
v_swap_b32 v0, v1
v_cndmask_b32 v0, v1, v2, vcc_lo
v_add_f32 v0, v1, v2
v_sub_f32 v0, v1, v2
v_subrev_f32 v0, v1, v2
v_mul_f32 v0, v1, v2
v_mul_i32_i24 v0, v1, v2
v_mul_hi_i32_i24 v0, v1, v2
v_mul_u32_u24 v0, v1, v2
v_mul_hi_u32_u24 v0, v1, v2
v_min_f32 v0, v1, v2
v_max_f32 v0, v1, v2
v_min_i32 v0, v1, v2
v_max_i32 v0, v1, v2
v_min_u32 v0, v1, v2
v_max_u32 v0, v1, v2
v_lshrrev_b32 v0, v1, v2
v_ashrrev_i32 v0, v1, v2
v_lshlrev_b32 v0, v1, v2
v_and_b32 v0, v1, v2
v_or_b32 v0, v1, v2
v_xor_b32 v0, v1, v2
v_xnor_b32 v0, v1, v2
v_mac_f32 v0, v1, v2
v_madmk_f32 v0, v1, 0x3fc00000, v2
v_madak_f32 v0, v1, v2, 0x3fc00000
v_add_nc_u32 v0, v1, v2
v_sub_nc_u32 v0, v1, v2
v_subrev_nc_u32 v0, v1, v2
v_add_co_ci_u32 v0, vcc_lo, v1, v2, vcc_lo
v_sub_co_ci_u32 v0, vcc_lo, v1, v2, vcc_lo
v_subrev_co_ci_u32 v0, vcc_lo, v1, v2, vcc_lo
v_fmac_f32 v0, v1, v2
v_fmamk_f32 v0, v1, 0x3fc00000, v2
v_fmaak_f32 v0, v1, v2, 0x3fc00000
v_cvt_pkrtz_f16_f32 v0, v1, v2
v_add_f16 v0, v1, v2
v_sub_f16 v0, v1, v2
v_subrev_f16 v0, v1, v2
v_mul_f16 v0, v1, v2
v_fmac_f16 v0, v1, v2
v_max_f16 v0, v1, v2
v_min_f16 v0, v1, v2
v_ldexp_f16 v0, v1, v2
v_pk_fmac_f16 v0, v1, v2
v_cmp_f_f32 vcc_lo, v1, v2
v_cmp_lt_f32 vcc_lo, v1, v2
v_cmp_eq_f32 vcc_lo, v1, v2
v_cmp_le_f32 vcc_lo, v1, v2
v_cmp_gt_f32 vcc_lo, v1, v2
v_cmp_lg_f32 vcc_lo, v1, v2
v_cmp_ge_f32 vcc_lo, v1, v2
v_cmp_o_f32 vcc_lo, v1, v2
v_cmp_u_f32 vcc_lo, v1, v2
v_cmp_neq_f32 vcc_lo, v1, v2
v_cmp_nlt_f32 vcc_lo, v1, v2
v_cmp_lt_i32 vcc_lo, v1, v2
v_cmp_eq_i32 vcc_lo, v1, v2
v_cmp_le_i32 vcc_lo, v1, v2
v_cmp_gt_i32 vcc_lo, v1, v2
v_cmp_ne_i32 vcc_lo, v1, v2
v_cmp_ge_i32 vcc_lo, v1, v2
v_cmp_lt_u32 vcc_lo, v1, v2
v_cmp_eq_u32 vcc_lo, v1, v2
v_cmp_le_u32 vcc_lo, v1, v2
v_cmp_gt_u32 vcc_lo, v1, v2
v_cmp_ne_u32 vcc_lo, v1, v2
v_cmp_ge_u32 vcc_lo, v1, v2
v_cmp_class_f32 vcc_lo, v1, v2
v_cmp_lt_f16 vcc_lo, v1, v2
v_cmp_gt_f16 vcc_lo, v1, v2
v_cmpx_lt_f32 v1, v2
v_cmpx_gt_f32 v1, v2
v_cmpx_eq_u32 v1, v2
v_cmpx_ne_u32 v1, v2
v_cmpx_lt_u32 v1, v2
v_cmpx_gt_u32 v1, v2
v_cmpx_le_u32 v1, v2
v_cmpx_ge_u32 v1, v2
v_cmpx_lt_i32 v1, v2
v_cmpx_gt_i32 v1, v2
v_mad_f32 v0, v1, v2, v3
v_mad_i32_i24 v0, v1, v2, v3
v_mad_u32_u24 v0, v1, v2, v3
v_cubeid_f32 v0, v1, v2, v3
v_bfe_u32 v0, v1, v2, v3
v_bfe_i32 v0, v1, v2, v3
v_bfi_b32 v0, v1, v2, v3
v_fma_f32 v0, v1, v2, v3
v_alignbit_b32 v0, v1, v2, v3
v_alignbyte_b32 v0, v1, v2, v3
v_min3_f32 v0, v1, v2, v3
v_min3_i32 v0, v1, v2, v3
v_min3_u32 v0, v1, v2, v3
v_max3_f32 v0, v1, v2, v3
v_max3_i32 v0, v1, v2, v3
v_max3_u32 v0, v1, v2, v3
v_med3_f32 v0, v1, v2, v3
v_med3_i32 v0, v1, v2, v3
v_sad_u8 v0, v1, v2, v3
v_sad_u32 v0, v1, v2, v3
v_div_fixup_f32 v0, v1, v2, v3
v_div_fmas_f32 v0, v1, v2, v3
v_div_scale_f32 v0, vcc_lo, v1, v2, v3
v_msad_u8 v0, v1, v2, v3
v_mad_u64_u32 v[0:1], s0, v1, v2, v[3:4]
v_mad_i64_i32 v[0:1], s0, v1, v2, v[3:4]
v_mul_lo_u32 v0, v1, v2
v_mul_hi_u32 v0, v1, v2
v_mul_hi_i32 v0, v1, v2
v_ldexp_f32 v0, v1, v2
v_readlane_b32 s0, v1, s2
v_writelane_b32 v0, s1, s2
v_bcnt_u32_b32 v0, v1, v2
v_mbcnt_lo_u32_b32 v0, v1, v2
v_mbcnt_hi_u32_b32 v0, v1, v2
v_lshlrev_b64 v[0:1], v1, v[2:3]
v_lshrrev_b64 v[0:1], v1, v[2:3]
v_ashrrev_i64 v[0:1], v1, v[2:3]
v_bfm_b32 v0, v1, v2
v_cvt_pknorm_i16_f32 v0, v1, v2
v_cvt_pknorm_u16_f32 v0, v1, v2
v_cvt_pk_u16_u32 v0, v1, v2
v_cvt_pk_i16_i32 v0, v1, v2
v_pack_b32_f16 v0, v1, v2
v_add_co_u32 v0, vcc_lo, v1, v2
v_sub_co_u32 v0, vcc_lo, v1, v2
v_subrev_co_u32 v0, vcc_lo, v1, v2
v_lshl_add_u32 v0, v1, v2, v3
v_add_lshl_u32 v0, v1, v2, v3
v_add3_u32 v0, v1, v2, v3
v_lshl_or_b32 v0, v1, v2, v3
v_and_or_b32 v0, v1, v2, v3
v_or3_b32 v0, v1, v2, v3
v_xad_u32 v0, v1, v2, v3
v_perm_b32 v0, v1, v2, v3
v_xor3_b32 v0, v1, v2, v3
v_mad_u32_u16 v0, v1, v2, v3
v_fma_f16 v0, v1, v2, v3
v_permlane16_b32 v0, v1, s0, s1
v_permlanex16_b32 v0, v1, s0, s1
v_pk_mad_i16 v0, v1, v2, v3
v_pk_mul_lo_u16 v0, v1, v2
v_pk_add_i16 v0, v1, v2
v_pk_sub_i16 v0, v1, v2
v_pk_lshlrev_b16 v0, v1, v2
v_pk_lshrrev_b16 v0, v1, v2
v_pk_ashrrev_i16 v0, v1, v2
v_pk_max_i16 v0, v1, v2
v_pk_min_i16 v0, v1, v2
v_pk_mad_u16 v0, v1, v2, v3
v_pk_add_u16 v0, v1, v2
v_pk_sub_u16 v0, v1, v2
v_pk_max_u16 v0, v1, v2
v_pk_min_u16 v0, v1, v2
v_pk_fma_f16 v0, v1, v2, v3
v_pk_add_f16 v0, v1, v2
v_pk_mul_f16 v0, v1, v2
v_pk_min_f16 v0, v1, v2
v_pk_max_f16 v0, v1, v2
v_fma_mix_f32 v0, v1, v2, v3
v_fma_mixlo_f16 v0, v1, v2, v3
v_fma_mixhi_f16 v0, v1, v2, v3
ds_add_u32 v1, v0
ds_sub_u32 v1, v0
ds_min_u32 v1, v0
ds_max_u32 v1, v0
ds_and_b32 v1, v0
ds_or_b32 v1, v0
ds_write_b32 v1, v0
ds_write2_b32 v1, v2, v3
ds_write2st64_b32 v1, v2, v3
ds_min_f32 v1, v0
ds_max_f32 v1, v0
ds_add_f32 v1, v0
ds_write_b8 v1, v0
ds_write_b16 v1, v0
ds_add_rtn_u32 v0, v1, v2
ds_max_rtn_f32 v0, v1, v2
ds_add_rtn_f32 v0, v1, v2
ds_swizzle_b32 v0, v1
ds_permute_b32 v0, v1, v2
ds_bpermute_b32 v0, v1, v2
ds_read_b32 v0, v1
ds_read2_b32 v[0:1], v1
ds_read2st64_b32 v[0:1], v1
ds_read_i8 v0, v1
ds_read_u8 v0, v1
ds_read_i16 v0, v1
ds_read_u16 v0, v1
ds_read_u16_d16 v0, v1
ds_read_u16_d16_hi v0, v1
ds_write_b64 v1, v[2:3]
ds_write2_b64 v1, v[2:3], v[4:5]
ds_read_b64 v[0:1], v1
ds_read2_b64 v[0:3], v1
ds_write_b96 v1, v[2:4]
ds_write_b128 v1, v[2:5]
ds_read_b96 v[0:2], v1
ds_read_b128 v[0:3], v1
ds_write_b16_d16_hi v1, v0
global_load_ubyte v0, v[2:3], off
global_load_sbyte v0, v[2:3], off
global_load_ushort v0, v[2:3], off
global_load_sshort v0, v[2:3], off
global_load_dword v0, v[2:3], off
global_load_dwordx2 v[0:1], v[2:3], off
global_load_dwordx3 v[0:2], v[2:3], off
global_load_dwordx4 v[0:3], v[2:3], off
global_load_ubyte_d16 v0, v[2:3], off
global_load_short_d16 v0, v[2:3], off
global_load_short_d16_hi v0, v[2:3], off
global_store_byte v[2:3], v0, off
global_store_short v[2:3], v0, off
global_store_short_d16_hi v[2:3], v0, off
global_store_dword v[2:3], v0, off
global_store_dwordx2 v[2:3], v[0:1], off
global_store_dwordx3 v[2:3], v[0:2], off
global_store_dwordx4 v[2:3], v[0:3], off
global_atomic_swap v[2:3], v0, off
global_atomic_cmpswap v[2:3], v[4:5], off
global_atomic_add v[2:3], v0, off
global_atomic_sub v[2:3], v0, off
global_atomic_smin v[2:3], v0, off
global_atomic_umin v[2:3], v0, off
global_atomic_smax v[2:3], v0, off
global_atomic_umax v[2:3], v0, off
global_atomic_and v[2:3], v0, off
global_atomic_or v[2:3], v0, off
global_atomic_xor v[2:3], v0, off
global_atomic_inc v[2:3], v0, off
global_atomic_fmin v[2:3], v0, off
global_atomic_fmax v[2:3], v0, off
buffer_load_ubyte v0, v1, s[4:7], 0 offen
buffer_load_ushort v0, v1, s[4:7], 0 offen
buffer_load_dword v0, v1, s[4:7], 0 offen
buffer_load_dwordx2 v[0:1], v1, s[4:7], 0 offen
buffer_load_dwordx4 v[0:3], v1, s[4:7], 0 offen
buffer_store_byte v0, v1, s[4:7], 0 offen
buffer_store_short v0, v1, s[4:7], 0 offen
buffer_store_dword v0, v1, s[4:7], 0 offen
buffer_store_dwordx2 v[0:1], v1, s[4:7], 0 offen
buffer_store_dwordx4 v[0:3], v1, s[4:7], 0 offen
buffer_atomic_add v0, v1, s[4:7], 0 offen
buffer_gl0_inv
buffer_gl1_inv
s_inst_prefetch 0
s_waitcnt_vscnt null, 0
s_waitcnt_vmcnt null, 0
s_waitcnt_lgkmcnt null, 0
v_madmk_f32 v0, v1, 0x3f800000, v2
v_madak_f32 v0, v1, v2, 0x3f800000
v_fmamk_f32 v0, v1, 0x3f800000, v2
v_fmaak_f32 v0, v1, v2, 0x3f800000
global_load_dword v0, v2, s[4:5] offset:-16
global_load_dword v0, v2, s[4:5] offset:2047 glc slc dlc
global_store_dword v2, v0, s[4:5] offset:8
global_atomic_add v0, v[2:3], v1, off glc
ds_read_b32 v0, v1 offset:65535
ds_read_b64 v[0:1], v1 offset:8
ds_read2_b32 v[0:1], v1 offset0:1 offset1:2
ds_read2st64_b32 v[0:1], v1 offset1:1
ds_write_b64 v1, v[2:3] offset:16
ds_write2_b32 v1, v2, v3 offset0:1 offset1:2
ds_swizzle_b32 v0, v1 offset:swizzle(SWAP,16)
buffer_load_dwordx4 v[0:3], v1, s[4:7], s8 offen offset:16
v_mov_b32_dpp v0, v1 row_shr:1 row_mask:0xf bank_mask:0xf
v_add_f32_dpp v0, v1, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:0
v_add_f32_dpp v0, v1, v2 row_xmask:1 row_mask:0xf bank_mask:0xf
v_add_f32_dpp v0, v1, v2 row_ror:4 row_mask:0xf bank_mask:0xf
v_add_f32_dpp v0, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v0, v1, v2 row_share:3 row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v0, v1 dpp8:[1,0,3,2,5,4,7,6]
v_add_f32_dpp v0, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
v_fmac_f32_dpp v0, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_add_f32_e64 v0, -v1, |v2|
v_add_f32_e64 v0, v1, v2 clamp
v_mul_f32_e64 v0, v1, v2 mul:2
v_fma_f32 v0, -v1, v2, |v3|
v_add_f32 v0, 1.0, v2
v_add_f32 v0, 0x40490fdb, v2
v_mul_f32 v0, s4, v2
v_fma_f32 v0, s4, v2, 0.5
v_cndmask_b32_e64 v0, v1, v2, s4
v_cmp_gt_f32_e64 s4, v1, v2
v_add_co_ci_u32_e64 v0, s4, v1, v2, s5
v_mov_b32 v0, exec_lo
s_mov_b32 exec_lo, s4
s_mov_b32 s0, -1
s_mov_b32 s0, 0x12345678
s_add_u32 s0, s1, 64
s_mov_b32 m0, s1
v_cvt_f32_f16_sdwa v0, v1 src0_sel:WORD_1
v_fma_mix_f32 v0, v1, v2, v3 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v0, v1, v2, v3 op_sel:[1,0,0] op_sel_hi:[0,1,1]
v_pk_fma_f16 v0, v1, v2, v3 neg_lo:[1,0,0] neg_hi:[1,0,0]
v_cmpx_gt_u32_e64 v1, v2
v_cmpx_gt_u32_e64 s4, v2
v_cmp_gt_u32_e64 vcc_lo, v1, s2
v_add_f32 v0, 0x3f800000, v2
v_add_f32 v0, 0x3e22f983, v2
v_add_f32 v0, 0.15915494, v2
v_add_nc_u32 v0, 0x3f800000, v2
v_add_nc_u32 v0, 1.0, v2
v_add_nc_u32 v0, -16, v2
v_add_nc_u32 v0, 0xfffffff0, v2
v_mov_b32 v0, 0x80000000
s_mov_b32 s0, 0x3f800000
v_add_f32 v0, v2, s1
v_add_f32 v0, v2, 1.0
v_lshrrev_b32 v0, 16, v2
v_and_b32 v0, 0xf0f0f0f, v2
v_and_b32 v0, v2, 15
v_fma_f32 v0, v1, v2, 0x3fc00000
v_fma_f32 v0, 0x3fc00000, v2, 0x3fc00000
v_cndmask_b32 v0, 0, 1.0, vcc_lo
v_cndmask_b32 v0, v1, v2, s4
v_add_co_u32 v0, s4, v1, v2
v_add_co_ci_u32 v0, vcc_lo, 0, v2, vcc_lo
v_add_co_ci_u32 v0, vcc_lo, v2, 0, vcc_lo
v_add_f16 v0, 1.0, v2
v_add_f16 v0, 0x3c00, v2
v_mul_f16 v0, 0x4000, v2
v_pk_mul_f16 v0, 1.0, v2
v_pk_fma_f16 v0, v1, s2, v3
v_readlane_b32 s0, v1, 5
v_writelane_b32 v0, s1, 5
v_mov_b32_e64 v0, v1
s_lshl_b32 s0, s1, 4
s_and_b32 s0, s1, 0xff00
s_and_b32 vcc_lo, exec_lo, s0
s_mov_b32 s0, scc
s_cselect_b32 s0, -1, 0
s_load_dwordx4 s[0:3], s[2:3], 0x10
s_load_dwordx4 s[0:3], s[2:3], s5
s_load_dword s0, s[2:3], 0x10 glc dlc
s_buffer_load_dword s0, s[4:7], s8
s_waitcnt vmcnt(1) lgkmcnt(2)
s_waitcnt vmcnt(63)
s_waitcnt lgkmcnt(0)
s_waitcnt expcnt(0)
s_waitcnt_vscnt null, 0x3f
s_mov_b64 exec, -1
s_mov_b64 vcc, exec
v_mov_b32 v255, v0
v_mov_b32 v0, s105
s_mov_b32 s105, 0
v_add_f32_e64 v0, s1, s2
v_fma_f32 v0, s1, s1, v2
v_mul_f32_e64 v0, v1, v2 div:2
v_mul_f32_e64 v0, v1, v2 mul:4
v_add_f32_e64 v0, neg(v1), v2
v_add_f32_e64 v0, abs(v1), v2
v_max_f32_e64 v0, |v1|, |v1|
v_exp_f32_e64 v0, -v1
v_add_f32_e64 v0, v1, -v2
v_mov_b32_dpp v0, v1 row_shr:1 row_mask:0x1 bank_mask:0x3 fi:1
v_add_f32_dpp v0, -v1, |v2| row_shr:1 row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v0, v1 row_mirror row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v0, v1 row_half_mirror row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v0, v1 row_shl:15 row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v0, v1 dpp8:[1,0,3,2,5,4,7,6] fi:1
v_add_f32_sdwa v0, v1, v2 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:WORD_0
v_cvt_f32_f16_sdwa v0, v1 src0_sel:WORD_0
v_cvt_f32_ubyte0_sdwa v0, s1 src0_sel:DWORD
ds_swizzle_b32 v0, v1 offset:swizzle(QUAD_PERM,1,0,3,2)
ds_swizzle_b32 v0, v1 offset:swizzle(REVERSE,8)
ds_swizzle_b32 v0, v1 offset:swizzle(BROADCAST,8,0)
ds_swizzle_b32 v0, v1 offset:0x041f
global_load_dword v0, v1, s[4:5]
global_load_dword v0, v[1:2], off offset:-2048
global_atomic_add v0, v1, v2, s[4:5] glc
global_atomic_add v1, v2, s[4:5]
global_atomic_cmpswap v0, v[2:3], v[4:5], off glc
buffer_load_dword v0, off, s[4:7], 0 offset:4
buffer_load_dword v0, v1, s[4:7], 0 idxen
buffer_load_dword v0, v[1:2], s[4:7], 0 idxen offen
buffer_load_dword v0, v1, s[4:7], 0 offen glc slc dlc
s_denorm_mode 0x0
s_getreg_b32 s0, hwreg(HW_REG_HW_ID)
