export interface IOConfig {
    input: string;
    output_dir: string;
    output_alpha: string;
    frame_start: number;
    frame_end: number;
    shot_type: 'locked_off' | 'moving' | 'unknown';
    alpha_format: 'png16' | 'png8' | 'dwaa';
    alpha_dwaa_quality: number;
    force_overwrite: boolean;
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

export interface RuntimeConfig {
    device: string;
    precision: string;
    workers_io: number;
    cache_dir: string;
    resume: boolean;
    verbose: boolean;
}

export interface VideoMatteConfig {
    io: IOConfig;
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
    runtime: RuntimeConfig;
}
