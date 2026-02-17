export interface IOConfig {
    input: string;
    output_dir: string;
    output_alpha: string;
    frame_start: number;
    frame_end: number;
    shot_type: 'locked_off' | 'moving' | 'unknown';
    alpha_format: 'png16' | 'png8' | 'dwaa' | 'exr_dwaa' | 'exr_dwaa_hq' | 'exr_lossless' | 'exr_raw';
    alpha_dwaa_quality: number;
    force_overwrite: boolean;
}

export interface ProjectConfig {
    path: string;
    masks_dir: string;
    cache_dir: string;
    autosave: boolean;
}

export interface AssignmentConfig {
    mode: 'mask_first';
    default_keyframe: number;
    require_assignment: boolean;
    unknown_radius_px: number;
    fg_erosion_px: number;
    bg_dilation_px: number;
}

export interface MemoryConfig {
    backend: string;
    memory_frames: number;
    window: number;
    max_anchors: number;
    confidence_reanchor_threshold: number;
    query_long_side?: number;
    spatial_weight?: number;
    temperature?: number;
    auto_anchor_min_gap?: number;
    region_constraint_enabled?: boolean;
    region_constraint_source?: 'none' | 'propagated_bbox' | 'propagated_mask' | 'nearest_keyframe_bbox';
    region_constraint_anchor_frame?: number;
    region_constraint_backend?: string;
    region_constraint_fallback_to_flow?: boolean;
    region_constraint_flow_downscale?: number;
    region_constraint_flow_min_coverage?: number;
    region_constraint_flow_max_coverage?: number;
    region_constraint_flow_feather_px?: number;
    region_constraint_samurai_model_cfg?: string;
    region_constraint_samurai_checkpoint?: string;
    region_constraint_samurai_offload_video_to_cpu?: boolean;
    region_constraint_samurai_offload_state_to_cpu?: boolean;
    region_constraint_threshold?: number;
    region_constraint_bbox_margin_px?: number;
    region_constraint_bbox_expand_ratio?: number;
    region_constraint_dilate_px?: number;
    region_constraint_soften_px?: number;
    region_constraint_outside_confidence_cap?: number;
}

export interface BackgroundConfig {
    enabled: boolean;
    sample_count: number;
    variance_threshold: number;
    photometric_normalize: boolean;
    occlusion_threshold: number;
    occlusion_fallback: 'auto' | 'temporal_extremes' | 'patch_inpaint' | 'ai_inpaint';
    manual_plate_path: string;
}

export interface ROIConfig {
    detect_every: number;
    pad_ratio: number;
    context_px: number;
    smooth_alpha: number;
    multi_person: 'single' | 'union_k';
    k: number;
    use_motion_mask: boolean;
    qc: boolean;
}

export interface GlobalConfig {
    model: string;
    long_side: number;
    chunk_len: number;
    chunk_overlap: number;
    use_roi_crop: boolean;
}

export interface IntermediateConfig {
    enabled: boolean;
    long_side: number;
    model: string;
    guide_filter_radius: number;
    guide_filter_eps: number;
    temporal_smooth: 'none' | 'ema' | 'flow';
    smooth_strength: number;
    selective_enabled: boolean;
    selective_rgb_threshold: number;
    selective_a0_threshold: number;
    selective_recheck_every: number;
    selective_max_skip: number;
    selective_delta_decay: number;
}

export interface BandConfig {
    mode: 'adaptive' | 'fixed';
    alpha_grad_threshold: number;
    dilate_alpha_px: number;
    dilate_rgb_px: number;
    rgb_proximity_px: number;
    edge_alignment_threshold: number;
    rgb_alpha_range: [number, number];
    dilate_bg_px: number;
    bg_enabled: boolean;
    bg_confidence_gate: number;
    bg_edge_persist_frames: number;
    bg_edge_roi_margin_px: number;
    band_max_coverage: number;
    auto_tighten: boolean;
    feather_px: number;
    hair_aware: boolean;
    hair_dilation_multiplier: number;
    compute_downscale: number;
}

export interface TrimapConfig {
    method: 'distance_transform' | 'erosion';
    unknown_width: number;
    unknown_width_hair: number;
    unknown_width_body: number;
    adaptive_width: boolean;
    adaptive_thresholds: boolean;
    t_fg: number;
    t_bg: number;
}

export interface TileConfig {
    tile_size: number;
    tile_size_backoff: number[];
    vram_headroom: number;
    overlap: number;
    min_band_coverage: number;
    blend_space: string;
    priority: string;
    tile_batch_size: number;
}

export interface RefineConfig {
    model: string;
    use_bg_plate: boolean;
    bg_confidence_gate: number;
    enabled: boolean;
    backend?: string;
    mematte_repo_dir?: string;
    mematte_checkpoint?: string;
    mematte_max_number_token?: number;
    mematte_patch_decoder?: boolean;
    unknown_band_px: number;
    region_trimap_enabled?: boolean;
    region_trimap_threshold?: number;
    region_trimap_fg_erode_px?: number;
    region_trimap_bg_dilate_px?: number;
    region_trimap_cleanup_px?: number;
    region_trimap_keep_largest?: boolean;
    region_trimap_min_coverage?: number;
    region_trimap_max_coverage?: number;
    tile_size: number;
    overlap: number;
    alpha_bg_threshold?: number;
    alpha_fg_threshold?: number;
    min_confidence?: number;
    guided_radius?: number;
    guided_eps?: number;
    edge_boost?: number;
    confidence_gain?: number;
    tile_min_coverage?: number;
}

export interface TemporalConfig {
    method: string;
    structural_sigma: number;
    structural_threshold: number;
    structural_blend_strength: number;
    detail_blend_strength: number;
    flow_consistency_sigma: number;
    fallback_threshold: number;
}

export interface PostprocessConfig {
    despill: {
        enabled: boolean;
        method: string;
        spill_color: [number, number, number];
        luma_bias: number;
    };
    fg_output: {
        enabled: boolean;
        format: string;
        premultiplied: boolean;
    };
}

export interface ReferenceFrameConfig {
    enabled: boolean;
    count: number;
    selection_method: string;
    propagation_range_max: number;
    propagation_error_limit: number;
    propagation_motion_limit: number;
}

export interface PreviewConfig {
    enabled: boolean;
    scale: number;
    every: number;
    modes: string[];
}

export interface QCConfig {
    enabled: boolean;
    fail_on_regression: boolean;
    auto_stage_diagnosis_on_fail: boolean;
    output_subdir: string;
    metrics_filename: string;
    report_filename: string;
    sample_output_frames: number;
    max_output_roundtrip_mae: number;
    alpha_range_eps: number;
    max_p95_flicker: number;
    max_p95_edge_flicker: number;
    min_mean_edge_confidence: number;
    band_spike_ratio: number;
    max_band_spike_frames: number;
}

export interface RuntimeConfig {
    device: string;
    precision: string;
    workers_io: number;
    cache_dir: string;
    resume: boolean;
    verbose: boolean;
}

export interface DebugConfig {
    export_stage_samples: boolean;
    auto_stage_samples_on_qc_fail: boolean;
    sample_count: number;
    sample_frames: number[];
    auto_sample_frames: number[];
    stage_dir: string;
    save_rgb: boolean;
    save_overlay: boolean;
}

export interface TemporalCleanupConfig {
    enabled: boolean;
    outside_band_ema_enabled?: boolean;
    outside_band_ema: number;
    min_confidence: number;
    confidence_clamp_enabled?: boolean;
    reset_on_new_anchor: boolean;
    anchor_reset_frames?: number;
    edge_bg_threshold?: number;
    edge_fg_threshold?: number;
    edge_band_radius_px?: number;
    edge_band_ema_enabled?: boolean;
    edge_band_ema?: number;
    edge_band_min_confidence?: number;
    edge_snap_enabled?: boolean;
    edge_snap_radius?: number;
    edge_snap_eps?: number;
    edge_snap_min_confidence?: number;
    clamp_delta?: number;
}

export interface MatteTuningConfig {
    enabled: boolean;
    shrink_grow_px: number;
    feather_px: number;
    offset_x_px: number;
    offset_y_px: number;
}

export interface VideoMatteConfig {
    io: IOConfig;
    project: ProjectConfig;
    assignment: AssignmentConfig;
    memory: MemoryConfig;
    background: BackgroundConfig;
    roi: ROIConfig;
    global: GlobalConfig; // Note: 'global' in TS, mapped to 'global_' in Python via alias
    intermediate: IntermediateConfig;
    band: BandConfig;
    trimap: TrimapConfig;
    tiles: TileConfig;
    refine: RefineConfig;
    temporal: TemporalConfig;
    postprocess: PostprocessConfig;
    reference_frames: ReferenceFrameConfig;
    preview: PreviewConfig;
    qc: QCConfig;
    temporal_cleanup: TemporalCleanupConfig;
    matte_tuning: MatteTuningConfig;
    runtime: RuntimeConfig;
    debug: DebugConfig;
}
