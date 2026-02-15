import { useState, useEffect, useCallback, useRef } from 'react'
import { FaPlay, FaSpinner, FaExclamationCircle, FaFileVideo, FaUpload, FaSync } from 'react-icons/fa'
import { VideoMatteConfig } from '../types'
import { Section } from './ui/Section'
import { Input } from './ui/Input'
import { Select } from './ui/Select'
import { Switch } from './ui/Switch'

interface ProjectKeyframe {
    frame: number
    mask_asset: string
    source: string
    kind: 'initial' | 'correction'
    updated_at: string
}

interface ProjectSummary {
    project_path: string
    keyframe_count: number
    keyframes: ProjectKeyframe[]
    require_assignment: boolean
}

interface SuggestedReprocessRange {
    frame_start: number
    frame_end: number
    reason: string
}

interface BuilderPoint {
    x: number
    y: number
}

interface BuilderBox {
    x0: number
    y0: number
    x1: number
    y1: number
}

interface BuilderCandidate extends BuilderBox {
    score: number
    source: string
    label: string
}

type BuilderTool = 'box' | 'fg' | 'bg'
type BuilderBackend = 'grabcut' | 'sam'
type PropagationBackend = 'flow' | 'sam2_video_predictor' | 'cutie'

interface MatteTuningPreset {
    id: string
    label: string
    description: string
    unknown_band_px: number
    shrink_grow_px: number
    feather_px: number
    offset_x_px: number
    offset_y_px: number
}

const MATTE_TUNING_PRESETS: MatteTuningPreset[] = [
    {
        id: 'subtle',
        label: 'Subtle',
        description: 'Mild refinement with very light feather.',
        unknown_band_px: 56,
        shrink_grow_px: 0,
        feather_px: 1,
        offset_x_px: 0,
        offset_y_px: 0,
    },
    {
        id: 'balanced',
        label: 'Balanced',
        description: 'Moderate edge tightening and smoothing.',
        unknown_band_px: 64,
        shrink_grow_px: -1,
        feather_px: 2,
        offset_x_px: 0,
        offset_y_px: 0,
    },
    {
        id: 'aggressive',
        label: 'Aggressive',
        description: 'Stronger choke and feather for difficult halos.',
        unknown_band_px: 80,
        shrink_grow_px: -2,
        feather_px: 3,
        offset_x_px: 0,
        offset_y_px: 0,
    },
]

// Default Configuration
const DEFAULT_CONFIG: VideoMatteConfig = {
    io: {
        input: "",
        output_dir: "output",
        output_alpha: "alpha/frame_%05d.png",
        frame_start: 0,
        frame_end: -1,
        shot_type: "locked_off",
        alpha_format: "png16",
        alpha_dwaa_quality: 45.0,
        force_overwrite: false
    },
    project: {
        path: "",
        masks_dir: "masks",
        cache_dir: "cache",
        autosave: true
    },
    assignment: {
        mode: "mask_first",
        default_keyframe: 0,
        require_assignment: true,
        unknown_radius_px: 64,
        fg_erosion_px: 6,
        bg_dilation_px: 12
    },
    memory: {
        backend: "appearance_memory_bank",
        memory_frames: 12,
        window: 120,
        max_anchors: 20,
        confidence_reanchor_threshold: 0.35,
        query_long_side: 960,
        spatial_weight: 0.1,
        temperature: 1.0,
        auto_anchor_min_gap: 0
    },
    background: {
        enabled: true,
        sample_count: 15,
        variance_threshold: 0.05,
        photometric_normalize: true,
        occlusion_threshold: 0.3,
        occlusion_fallback: "auto",
        manual_plate_path: ""
    },
    roi: {
        detect_every: 15,
        pad_ratio: 0.25,
        context_px: 256,
        smooth_alpha: 0.3,
        multi_person: "union_k",
        k: 2,
        use_motion_mask: true,
        qc: true
    },
    global: {
        model: "rvm",
        long_side: 2048,
        chunk_len: 24,
        chunk_overlap: 6,
        use_roi_crop: true
    },
    intermediate: {
        enabled: true,
        long_side: 4096,
        model: "vitmatte",
        guide_filter_radius: 8,
        guide_filter_eps: 0.01,
        temporal_smooth: "flow",
        smooth_strength: 0.3,
        selective_enabled: true,
        selective_rgb_threshold: 0.010,
        selective_a0_threshold: 0.005,
        selective_recheck_every: 8,
        selective_max_skip: 6,
        selective_delta_decay: 0.98
    },
    band: {
        mode: "adaptive",
        alpha_grad_threshold: 0.01,
        dilate_alpha_px: 96,
        dilate_rgb_px: 64,
        rgb_proximity_px: 192,
        edge_alignment_threshold: 0.3,
        rgb_alpha_range: [0.05, 0.95],
        dilate_bg_px: 64,
        bg_enabled: true,
        bg_confidence_gate: 0.5,
        bg_edge_persist_frames: 3,
        bg_edge_roi_margin_px: 128,
        band_max_coverage: 0.35,
        auto_tighten: true,
        feather_px: 64,
        hair_aware: true,
        hair_dilation_multiplier: 2.0,
        compute_downscale: 0.25
    },
    trimap: {
        method: "distance_transform",
        unknown_width: 32,
        unknown_width_hair: 48,
        unknown_width_body: 24,
        adaptive_width: true,
        adaptive_thresholds: true,
        t_fg: 0.95,
        t_bg: 0.05
    },
    tiles: {
        tile_size: 2048,
        tile_size_backoff: [2048, 1536, 1024],
        vram_headroom: 0.85,
        overlap: 384,
        min_band_coverage: 0.005,
        blend_space: "logit",
        priority: "hair_first",
        tile_batch_size: 4
    },
    refine: {
        enabled: true,
        backend: "guided_band",
        unknown_band_px: 64,
        tile_size: 1536,
        overlap: 96,
        alpha_bg_threshold: 0.05,
        alpha_fg_threshold: 0.95,
        min_confidence: 0.5,
        guided_radius: 8,
        guided_eps: 0.01,
        edge_boost: 0.15,
        confidence_gain: 1.0,
        tile_min_coverage: 0.002,
        model: "vitmatte",
        use_bg_plate: true,
        bg_confidence_gate: 0.8
    },
    temporal: {
        method: "frequency_separation",
        structural_sigma: 1.0,
        structural_threshold: 0.015,
        structural_blend_strength: 0.7,
        detail_blend_strength: 0.95,
        flow_consistency_sigma: 1.0,
        fallback_threshold: 0.2
    },
    postprocess: {
        despill: {
            enabled: true,
            method: "advanced",
            spill_color: [0.0, 1.0, 0.0],
            luma_bias: 0.1
        },
        fg_output: {
            enabled: true,
            format: "png",
            premultiplied: false
        }
    },
    reference_frames: {
        enabled: false,
        count: 5,
        selection_method: "auto_quality",
        propagation_range_max: 30,
        propagation_error_limit: 15.0,
        propagation_motion_limit: 50.0
    },
    preview: {
        enabled: false,
        scale: 0.5,
        every: 10,
        modes: ["checker", "alpha", "white", "flicker"]
    },
    qc: {
        enabled: true,
        fail_on_regression: false,
        output_subdir: "qc",
        metrics_filename: "optionb_metrics.json",
        report_filename: "optionb_report.md",
        sample_output_frames: 3,
        max_output_roundtrip_mae: 0.01,
        alpha_range_eps: 0.001,
        max_p95_flicker: 0.08,
        max_p95_edge_flicker: 0.12,
        min_mean_edge_confidence: 0.15,
        band_spike_ratio: 2.5,
        max_band_spike_frames: 8,
    },
    temporal_cleanup: {
        enabled: true,
        outside_band_ema: 0.15,
        min_confidence: 0.5,
        reset_on_new_anchor: true,
        anchor_reset_frames: 6,
        edge_bg_threshold: 0.05,
        edge_fg_threshold: 0.95,
        edge_band_radius_px: 2,
        edge_snap_enabled: false,
        edge_snap_radius: 2,
        edge_snap_eps: 0.01,
        clamp_delta: 0.25
    },
    matte_tuning: {
        enabled: true,
        shrink_grow_px: 0,
        feather_px: 0,
        offset_x_px: 0,
        offset_y_px: 0,
    },
    runtime: {
        device: "cuda",
        precision: "fp16",
        workers_io: 4,
        cache_dir: ".cache",
        resume: true,
        verbose: false
    }
}

export default function RunTab({ onSuccess }: { onSuccess: () => void }) {
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [status, setStatus] = useState<string | null>(null)
    const [showAdvanced, setShowAdvanced] = useState(true)
    const [projectSummary, setProjectSummary] = useState<ProjectSummary | null>(null)
    const [assignmentMaskPath, setAssignmentMaskPath] = useState("")
    const [assignmentFrame, setAssignmentFrame] = useState(0)
    const [assignmentLoading, setAssignmentLoading] = useState(false)
    const [assignmentKind, setAssignmentKind] = useState<'initial' | 'correction'>('initial')
    const [autoApplySuggestedRange, setAutoApplySuggestedRange] = useState(true)
    const [suggestedRange, setSuggestedRange] = useState<SuggestedReprocessRange | null>(null)
    const [builderTool, setBuilderTool] = useState<BuilderTool>('box')
    const [builderFrameDataUrl, setBuilderFrameDataUrl] = useState<string | null>(null)
    const [builderMaskPreviewUrl, setBuilderMaskPreviewUrl] = useState<string | null>(null)
    const [builderFrameSize, setBuilderFrameSize] = useState<{ width: number; height: number } | null>(null)
    const [builderBox, setBuilderBox] = useState<BuilderBox | null>(null)
    const [builderDraftBox, setBuilderDraftBox] = useState<BuilderBox | null>(null)
    const [builderFgPoints, setBuilderFgPoints] = useState<BuilderPoint[]>([])
    const [builderBgPoints, setBuilderBgPoints] = useState<BuilderPoint[]>([])
    const [builderPointRadius, setBuilderPointRadius] = useState(8)
    const [builderIterCount, setBuilderIterCount] = useState(5)
    const [builderPrompt, setBuilderPrompt] = useState("person")
    const [builderSuggestingBoxes, setBuilderSuggestingBoxes] = useState(false)
    const [builderCandidates, setBuilderCandidates] = useState<BuilderCandidate[]>([])
    const [builderBackend, setBuilderBackend] = useState<BuilderBackend>('grabcut')
    const [builderSamModelId, setBuilderSamModelId] = useState("facebook/sam-vit-base")
    const [builderSamLocalOnly, setBuilderSamLocalOnly] = useState(true)
    const [propagateBackend, setPropagateBackend] = useState<PropagationBackend>('flow')
    const [propagateFrameStart, setPropagateFrameStart] = useState<number>(DEFAULT_CONFIG.io.frame_start)
    const [propagateFrameEnd, setPropagateFrameEnd] = useState<number>(DEFAULT_CONFIG.io.frame_end)
    const [propagateStride, setPropagateStride] = useState(8)
    const [propagateMaxNewKeyframes, setPropagateMaxNewKeyframes] = useState(24)
    const [propagateFallbackToFlow, setPropagateFallbackToFlow] = useState(true)
    const [propagateOverwriteExisting, setPropagateOverwriteExisting] = useState(false)
    const [propagateFlowDownscale, setPropagateFlowDownscale] = useState(0.5)
    const [propagateFlowMinCoverage, setPropagateFlowMinCoverage] = useState(0.002)
    const [propagateFlowMaxCoverage, setPropagateFlowMaxCoverage] = useState(0.98)
    const [propagateFlowFeatherPx, setPropagateFlowFeatherPx] = useState(1)
    const [propagateRunning, setPropagateRunning] = useState(false)
    const [builderLoadingFrame, setBuilderLoadingFrame] = useState(false)
    const [builderBuildingMask, setBuilderBuildingMask] = useState(false)
    const [builderDragStart, setBuilderDragStart] = useState<BuilderPoint | null>(null)
    const builderImgRef = useRef<HTMLImageElement | null>(null)

    // Initialize config with localStorage defaults if available
    const [config, setConfig] = useState<VideoMatteConfig>(() => {
        const defaultsStr = localStorage.getItem('videomatte_defaults')
        if (defaultsStr) {
            try {
                const defaults = JSON.parse(defaultsStr)
                return {
                    ...DEFAULT_CONFIG,
                    io: {
                        ...DEFAULT_CONFIG.io,
                        output_dir: defaults.outputDir || DEFAULT_CONFIG.io.output_dir
                    },
                    runtime: {
                        ...DEFAULT_CONFIG.runtime,
                        device: defaults.device || DEFAULT_CONFIG.runtime.device,
                        precision: defaults.precision || DEFAULT_CONFIG.runtime.precision
                    }
                }
            } catch (e) {
                console.error("Error parsing defaults", e)
            }
        }
        return DEFAULT_CONFIG
    })

    // Load UI prefs on mount
    useEffect(() => {
        const prefsStr = localStorage.getItem('videomatte_ui_prefs')
        if (prefsStr) {
            try {
                const prefs = JSON.parse(prefsStr)
                if (typeof prefs.showAdvanced === 'boolean') {
                    setShowAdvanced(prefs.showAdvanced)
                }
            } catch (e) {
                console.error("Error parsing prefs", e)
            }
        }
    }, [])

    const updateConfig = (section: keyof VideoMatteConfig, field: string, value: any) => {
        setConfig(prev => ({
            ...prev,
            [section]: {
                ...prev[section],
                [field]: value
            }
        }))
    }

    const updateNestedConfig = (section: keyof VideoMatteConfig, subsection: string, field: string, value: any) => {
        setConfig(prev => {
            const sec = prev[section] as any;
            return {
                ...prev,
                [section]: {
                    ...sec,
                    [subsection]: {
                        ...sec[subsection],
                        [field]: value
                    }
                }
            }
        })
    }

    const [dragOver, setDragOver] = useState(false)

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        setDragOver(false)
        // Try to get file path from dropped items
        const files = e.dataTransfer.files
        if (files.length > 0) {
            // Browser security prevents reading full paths, but we show the filename
            // User would need to type the full path — this handles the UX hint
            const name = files[0].name
            updateConfig('io', 'input', name)
        }
        // Try text data (e.g., dragged from file manager on some platforms)
        const text = e.dataTransfer.getData('text/plain')
        if (text) {
            updateConfig('io', 'input', text.trim())
        }
    }, [])

    async function parseApiError(res: Response): Promise<string> {
        try {
            const data = await res.json()
            if (typeof data?.detail === 'string') return data.detail
            return JSON.stringify(data)
        } catch {
            return await res.text()
        }
    }

    async function refreshProjectSummary(configOverride?: VideoMatteConfig): Promise<ProjectSummary> {
        const cfg = configOverride ?? config
        const res = await fetch('/api/project/state', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config: cfg })
        })
        if (!res.ok) {
            throw new Error(await parseApiError(res))
        }
        const data = await res.json() as ProjectSummary
        setProjectSummary(data)
        return data
    }

    function applySuggestedRange(range: SuggestedReprocessRange) {
        setConfig(prev => ({
            ...prev,
            io: {
                ...prev.io,
                frame_start: range.frame_start,
                frame_end: range.frame_end,
            }
        }))
    }

    function clampBuilderPoint(x: number, y: number, width: number, height: number): BuilderPoint {
        return {
            x: Math.max(0, Math.min(width - 1, x)),
            y: Math.max(0, Math.min(height - 1, y)),
        }
    }

    function normalizeBuilderBox(box: BuilderBox): BuilderBox {
        return {
            x0: Math.min(box.x0, box.x1),
            y0: Math.min(box.y0, box.y1),
            x1: Math.max(box.x0, box.x1),
            y1: Math.max(box.y0, box.y1),
        }
    }

    function getBuilderPointFromMouse(e: React.MouseEvent<HTMLDivElement>): BuilderPoint | null {
        const img = builderImgRef.current
        const size = builderFrameSize
        if (!img || !size) return null
        const rect = img.getBoundingClientRect()
        if (rect.width <= 0 || rect.height <= 0) return null
        const px = ((e.clientX - rect.left) * size.width) / rect.width
        const py = ((e.clientY - rect.top) * size.height) / rect.height
        return clampBuilderPoint(px, py, size.width, size.height)
    }

    function clearBuilderPrompts() {
        setBuilderBox(null)
        setBuilderDraftBox(null)
        setBuilderFgPoints([])
        setBuilderBgPoints([])
        setBuilderMaskPreviewUrl(null)
        setBuilderCandidates([])
    }

    async function handleLoadBuilderFrame() {
        setError(null)
        setStatus(null)
        setBuilderLoadingFrame(true)
        try {
            const res = await fetch('/api/assignments/frame-preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    config,
                    frame: assignmentFrame,
                }),
            })
            if (!res.ok) throw new Error(await parseApiError(res))
            const data = await res.json() as {
                data_url: string
                width: number
                height: number
            }
            setBuilderFrameDataUrl(data.data_url)
            setBuilderFrameSize({ width: data.width, height: data.height })
            clearBuilderPrompts()
            setStatus(`Loaded frame ${assignmentFrame} for prompt-based mask building.`)
        } catch (err: any) {
            setError(err.message)
        } finally {
            setBuilderLoadingFrame(false)
        }
    }

    async function handleSuggestBuilderBoxes() {
        setError(null)
        setStatus(null)
        if (!builderFrameDataUrl || !builderFrameSize) {
            setError("Load a frame before requesting prompt box suggestions.")
            return
        }
        const prompt = builderPrompt.trim()
        if (!prompt) {
            setError("Enter a prompt first (for example: person, person left, person center).")
            return
        }
        setBuilderSuggestingBoxes(true)
        try {
            const res = await fetch('/api/assignments/suggest-boxes', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    config,
                    frame: assignmentFrame,
                    prompt,
                    max_candidates: 5,
                }),
            })
            if (!res.ok) throw new Error(await parseApiError(res))
            const data = await res.json() as { candidates: BuilderCandidate[] }
            const cands = data.candidates || []
            setBuilderCandidates(cands)
            if (cands.length > 0) {
                setBuilderBox({
                    x0: cands[0].x0,
                    y0: cands[0].y0,
                    x1: cands[0].x1,
                    y1: cands[0].y1,
                })
                setBuilderDraftBox(null)
                setBuilderMaskPreviewUrl(null)
                setStatus(`Detected ${cands.length} candidate box(es). Applied top candidate.`)
            } else {
                setStatus("No prompt candidates found; draw a manual box.")
            }
        } catch (err: any) {
            setError(err.message)
        } finally {
            setBuilderSuggestingBoxes(false)
        }
    }

    function applyBuilderCandidate(c: BuilderCandidate, index: number) {
        setBuilderBox({ x0: c.x0, y0: c.y0, x1: c.x1, y1: c.y1 })
        setBuilderDraftBox(null)
        setBuilderMaskPreviewUrl(null)
        setStatus(`Applied candidate #${index + 1} (${c.label}, ${Math.round(c.score * 100)}%).`)
    }

    async function handleBuildMaskFromPrompts() {
        setError(null)
        setStatus(null)
        if (!builderFrameDataUrl || !builderFrameSize) {
            setError("Load a frame in the mask builder first.")
            return
        }
        if (!builderBox) {
            setError("Draw a box around the subject first.")
            return
        }

        setBuilderBuildingMask(true)
        try {
            const res = await fetch('/api/assignments/build-mask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    config,
                    frame: assignmentFrame,
                    kind: assignmentKind,
                    source: "ui_builder",
                    backend: builderBackend,
                    box: builderBox,
                    fg_points: builderFgPoints,
                    bg_points: builderBgPoints,
                    point_radius: builderPointRadius,
                    iter_count: builderIterCount,
                    sam_model_id: builderSamModelId,
                    sam_local_files_only: builderSamLocalOnly,
                    sam_fallback_to_grabcut: true,
                }),
            })
            if (!res.ok) throw new Error(await parseApiError(res))
            const data = await res.json() as {
                project_path: string
                keyframe_count: number
                keyframes: ProjectKeyframe[]
                require_assignment: boolean
                suggested_reprocess_range?: SuggestedReprocessRange
                mask_preview_data_url?: string
                coverage?: number
                backend_used?: string
                builder_note?: string | null
            }
            const range = data.suggested_reprocess_range
            setProjectSummary({
                project_path: data.project_path,
                keyframe_count: data.keyframe_count,
                keyframes: data.keyframes || [],
                require_assignment: data.require_assignment ?? true,
            })
            if (data.mask_preview_data_url) {
                setBuilderMaskPreviewUrl(data.mask_preview_data_url)
            }
            const backendUsedLabel = data.backend_used ? ` (${data.backend_used})` : ""
            if (range) {
                setSuggestedRange(range)
                if (assignmentKind === 'correction' && autoApplySuggestedRange) {
                    applySuggestedRange(range)
                    setStatus(
                        `Built and imported correction mask${backendUsedLabel} at frame ${assignmentFrame}. Applied reprocess range ${range.frame_start}..${range.frame_end}.`
                    )
                } else {
                    const covPct = Math.round((data.coverage ?? 0) * 1000) / 10
                    setStatus(
                        `Built and imported mask${backendUsedLabel} at frame ${assignmentFrame} (coverage ${covPct}%). Suggested range: ${range.frame_start}..${range.frame_end}.`
                    )
                }
            } else {
                setStatus(`Built and imported mask${backendUsedLabel} at frame ${assignmentFrame}.`)
            }
            if (data.builder_note) {
                setStatus(prev => prev ? `${prev} ${data.builder_note}` : data.builder_note || null)
            }
        } catch (err: any) {
            setError(err.message)
        } finally {
            setBuilderBuildingMask(false)
        }
    }

    async function handlePropagateAssignments() {
        setError(null)
        setStatus(null)
        setPropagateRunning(true)
        try {
            const res = await fetch('/api/assignments/propagate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    config,
                    anchor_frame: assignmentFrame,
                    frame_start: propagateFrameStart,
                    frame_end: propagateFrameEnd,
                    backend: propagateBackend,
                    fallback_to_flow: propagateFallbackToFlow,
                    stride: propagateStride,
                    max_new_keyframes: propagateMaxNewKeyframes,
                    flow_downscale: propagateFlowDownscale,
                    flow_min_coverage: propagateFlowMinCoverage,
                    flow_max_coverage: propagateFlowMaxCoverage,
                    flow_feather_px: propagateFlowFeatherPx,
                    kind: 'correction',
                    source: 'ui_propagate',
                    overwrite_existing: propagateOverwriteExisting,
                }),
            })
            if (!res.ok) throw new Error(await parseApiError(res))
            const data = await res.json() as {
                project_path: string
                keyframe_count: number
                keyframes: ProjectKeyframe[]
                require_assignment: boolean
                inserted_count?: number
                inserted_frames?: number[]
                skipped_existing_frames?: number[]
                backend_used?: string
                builder_note?: string | null
                suggested_reprocess_range?: SuggestedReprocessRange
                frame_start?: number
                frame_end?: number
            }

            setProjectSummary({
                project_path: data.project_path,
                keyframe_count: data.keyframe_count,
                keyframes: data.keyframes || [],
                require_assignment: data.require_assignment ?? true,
            })

            const inserted = data.inserted_count ?? (data.inserted_frames?.length ?? 0)
            const backendUsedLabel = data.backend_used ? ` (${data.backend_used})` : ""
            const rangeMsg = `range ${data.frame_start ?? propagateFrameStart}..${data.frame_end ?? propagateFrameEnd}`
            let msg = `Propagation assist${backendUsedLabel} inserted ${inserted} keyframe(s) from anchor ${assignmentFrame} over ${rangeMsg}.`
            if ((data.skipped_existing_frames?.length ?? 0) > 0) {
                msg += ` Skipped existing: ${data.skipped_existing_frames!.length}.`
            }
            if (data.builder_note) {
                msg += ` ${data.builder_note}`
            }
            setStatus(msg)

            if (data.suggested_reprocess_range) {
                setSuggestedRange(data.suggested_reprocess_range)
                if (autoApplySuggestedRange) {
                    applySuggestedRange(data.suggested_reprocess_range)
                }
            }
        } catch (err: any) {
            setError(err.message)
        } finally {
            setPropagateRunning(false)
        }
    }

    function handleBuilderMouseDown(e: React.MouseEvent<HTMLDivElement>) {
        if (builderTool !== 'box' || !builderFrameSize) return
        const p = getBuilderPointFromMouse(e)
        if (!p) return
        setBuilderDragStart(p)
        setBuilderDraftBox({ x0: p.x, y0: p.y, x1: p.x, y1: p.y })
    }

    function handleBuilderMouseMove(e: React.MouseEvent<HTMLDivElement>) {
        if (builderTool !== 'box' || !builderFrameSize || !builderDragStart) return
        const p = getBuilderPointFromMouse(e)
        if (!p) return
        setBuilderDraftBox({
            x0: builderDragStart.x,
            y0: builderDragStart.y,
            x1: p.x,
            y1: p.y,
        })
    }

    function finalizeBuilderBox(p: BuilderPoint | null) {
        if (!builderDragStart || !builderFrameSize) return
        const end = p ?? builderDragStart
        const normalized = normalizeBuilderBox({
            x0: builderDragStart.x,
            y0: builderDragStart.y,
            x1: end.x,
            y1: end.y,
        })
        setBuilderDragStart(null)
        setBuilderDraftBox(null)
        if ((normalized.x1 - normalized.x0) < 3 || (normalized.y1 - normalized.y0) < 3) {
            return
        }
        setBuilderBox(normalized)
        setBuilderMaskPreviewUrl(null)
        setBuilderCandidates([])
    }

    function handleBuilderMouseUp(e: React.MouseEvent<HTMLDivElement>) {
        if (builderTool !== 'box') return
        const p = getBuilderPointFromMouse(e)
        finalizeBuilderBox(p)
    }

    function handleBuilderMouseLeave() {
        if (builderTool !== 'box') return
        finalizeBuilderBox(null)
    }

    function handleBuilderClick(e: React.MouseEvent<HTMLDivElement>) {
        if (!builderFrameSize || builderTool === 'box') return
        const p = getBuilderPointFromMouse(e)
        if (!p) return
        if (builderTool === 'fg') {
            setBuilderFgPoints(prev => [...prev, p])
        } else if (builderTool === 'bg') {
            setBuilderBgPoints(prev => [...prev, p])
        }
        setBuilderMaskPreviewUrl(null)
    }

    function applyMattePreset(preset: MatteTuningPreset) {
        setConfig(prev => ({
            ...prev,
            refine: {
                ...prev.refine,
                unknown_band_px: preset.unknown_band_px,
            },
            matte_tuning: {
                ...prev.matte_tuning,
                enabled: true,
                shrink_grow_px: preset.shrink_grow_px,
                feather_px: preset.feather_px,
                offset_x_px: preset.offset_x_px,
                offset_y_px: preset.offset_y_px,
            },
        }))
        setStatus(
            `Applied matte preset '${preset.label}': band=${preset.unknown_band_px}, shrink/grow=${preset.shrink_grow_px}, feather=${preset.feather_px}, offset=(${preset.offset_x_px}, ${preset.offset_y_px}).`
        )
    }

    function resetMattePreset() {
        setConfig(prev => ({
            ...prev,
            refine: {
                ...prev.refine,
                unknown_band_px: DEFAULT_CONFIG.refine.unknown_band_px,
            },
            matte_tuning: {
                ...prev.matte_tuning,
                enabled: DEFAULT_CONFIG.matte_tuning.enabled,
                shrink_grow_px: DEFAULT_CONFIG.matte_tuning.shrink_grow_px,
                feather_px: DEFAULT_CONFIG.matte_tuning.feather_px,
                offset_x_px: DEFAULT_CONFIG.matte_tuning.offset_x_px,
                offset_y_px: DEFAULT_CONFIG.matte_tuning.offset_y_px,
            },
        }))
        setStatus("Reset matte tuning controls to defaults.")
    }

    function isMattePresetActive(preset: MatteTuningPreset): boolean {
        return (
            config.refine.unknown_band_px === preset.unknown_band_px &&
            config.matte_tuning.shrink_grow_px === preset.shrink_grow_px &&
            config.matte_tuning.feather_px === preset.feather_px &&
            config.matte_tuning.offset_x_px === preset.offset_x_px &&
            config.matte_tuning.offset_y_px === preset.offset_y_px
        )
    }

    async function handleImportAssignment() {
        setError(null)
        setStatus(null)
        if (!assignmentMaskPath.trim()) {
            setError("Mask path is required before import.")
            return
        }
        setAssignmentLoading(true)
        try {
            const res = await fetch('/api/assignments/import', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    config,
                    frame: assignmentFrame,
                    mask_path: assignmentMaskPath.trim(),
                    source: "ui",
                    kind: assignmentKind,
                })
            })
            if (!res.ok) throw new Error(await parseApiError(res))
            const data = await res.json()
            const range = data.suggested_reprocess_range as SuggestedReprocessRange | undefined
            setProjectSummary({
                project_path: data.project_path,
                keyframe_count: data.keyframe_count,
                keyframes: data.keyframes || [],
                require_assignment: data.require_assignment ?? true
            })
            if (range) {
                setSuggestedRange(range)
                if (assignmentKind === 'correction' && autoApplySuggestedRange) {
                    applySuggestedRange(range)
                    setStatus(
                        `Imported correction anchor at frame ${assignmentFrame}. Applied reprocess range ${range.frame_start}..${range.frame_end}.`
                    )
                } else {
                    setStatus(`Imported keyframe mask at frame ${assignmentFrame}. Suggested range: ${range.frame_start}..${range.frame_end}.`)
                }
            } else {
                setStatus(`Imported keyframe mask at frame ${assignmentFrame}.`)
            }
        } catch (err: any) {
            setError(err.message)
        } finally {
            setAssignmentLoading(false)
        }
    }

    useEffect(() => {
        refreshProjectSummary().catch(() => { })
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    async function handleSubmit(e: React.FormEvent) {
        e.preventDefault()
        setLoading(true)
        setError(null)
        setStatus(null)

        try {
            if (config.assignment.require_assignment) {
                const summary = await refreshProjectSummary(config)
                if (!summary.keyframe_count) {
                    throw new Error("Assignment required. Import at least one keyframe mask in 'Subject Assignment' before starting.")
                }
            }

            const res = await fetch('/api/jobs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config })
            })

            if (!res.ok) throw new Error(await parseApiError(res))

            onSuccess()
        } catch (err: any) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const activeBuilderBox = builderDraftBox ?? builderBox
    const assignmentBusy =
        assignmentLoading || builderLoadingFrame || builderBuildingMask || builderSuggestingBoxes || propagateRunning

    return (
        <div className="space-y-4 pb-20">
            <div className="border-b border-gray-800 pb-2 flex justify-between items-end">
                <div>
                    <h2 className="text-xl font-bold text-white">New Matting Job</h2>
                    <p className="text-sm text-gray-400">Configure and start a new video matting pipeline.</p>
                </div>
                {!showAdvanced && (
                    <div className="text-xs text-gray-500 italic bg-gray-800 px-2 py-1 rounded">
                        Advanced hidden
                    </div>
                )}
            </div>

            {error && (
                <div className="bg-red-500/10 border border-red-500/20 text-red-500 p-3 rounded-lg flex items-center gap-2 text-sm">
                    <FaExclamationCircle />
                    {error}
                </div>
            )}
            {status && (
                <div className="bg-green-500/20 border border-green-500/20 text-green-400 p-3 rounded-lg text-sm">
                    {status}
                </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-2">

                {/* 1. IO Section */}
                <Section title="Input / Output" defaultOpen={true} tooltip="Configure file paths, frame ranges, and formats.">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
                        {/* Drop zone for input file */}
                        <div
                            className="col-span-2"
                            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
                            onDragLeave={() => setDragOver(false)}
                            onDrop={handleDrop}
                        >
                            {!config.io.input ? (
                                <div className={`border-2 border-dashed rounded-lg p-4 text-center transition-colors cursor-pointer ${dragOver ? 'border-brand-500 bg-brand-500/5' : 'border-gray-700 hover:border-gray-500'}`}
                                     onClick={() => {
                                         const el = document.getElementById('input-path-field')
                                         if (el) el.focus()
                                     }}
                                >
                                    <FaFileVideo className={`mx-auto text-2xl mb-2 ${dragOver ? 'text-brand-400' : 'text-gray-500'}`} />
                                    <p className="text-sm text-gray-400">Drop a video file here or type the path below</p>
                                    <input
                                        id="input-path-field"
                                        value={config.io.input}
                                        onChange={e => updateConfig('io', 'input', e.target.value)}
                                        placeholder="videos/my_video.mp4"
                                        className="mt-2 w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-brand-500"
                                    />
                                </div>
                            ) : (
                                <Input
                                    label="Input Path"
                                    name="input"
                                    value={config.io.input}
                                    onChange={e => updateConfig('io', 'input', e.target.value)}
                                    placeholder="videos/my_video.mp4"
                                    tooltip="Path to the video file or image sequence (e.g., 'frame_%05d.png')."
                                />
                            )}
                        </div>
                        <Input
                            label="Output Directory"
                            value={config.io.output_dir}
                            onChange={e => updateConfig('io', 'output_dir', e.target.value)}
                            tooltip="Directory where results will be saved."
                        />
                        <Input
                            label="Alpha Pattern"
                            value={config.io.output_alpha}
                            onChange={e => updateConfig('io', 'output_alpha', e.target.value)}
                            tooltip="Naming pattern for output alpha frames (printf format)."
                        />
                        <div className="grid grid-cols-2 gap-2">
                            <Input
                                label="Start Frame"
                                type="number"
                                value={config.io.frame_start}
                                onChange={e => updateConfig('io', 'frame_start', parseInt(e.target.value))}
                                tooltip="Frame index to start processing from (0-based)."
                            />
                            <Input
                                label="End Frame"
                                type="number"
                                value={config.io.frame_end}
                                onChange={e => updateConfig('io', 'frame_end', parseInt(e.target.value))}
                                tooltip="Frame index to end at. Use -1 to process until end of video."
                            />
                        </div>
                        <Select
                            label="Shot Type"
                            value={config.io.shot_type}
                            onChange={e => updateConfig('io', 'shot_type', e.target.value)}
                            options={[
                                { value: 'locked_off', label: 'Locked Off' },
                                { value: 'moving', label: 'Moving' },
                                { value: 'unknown', label: 'Unknown' }
                            ]}
                            tooltip="Camera motion type. 'Locked Off' assumes a static camera."
                        />
                        <div className="grid grid-cols-2 gap-2">
                            <Select
                                label="Alpha Format"
                                value={config.io.alpha_format}
                                onChange={e => updateConfig('io', 'alpha_format', e.target.value)}
                                options={[
                                    { value: 'png16', label: 'PNG 16-bit' },
                                    { value: 'png8', label: 'PNG 8-bit' },
                                    { value: 'dwaa', label: 'EXR DWAA' }
                                ]}
                                tooltip="File format for the alpha matte."
                            />
                            <Input
                                label="DWAA Quality"
                                type="number"
                                step="0.1"
                                value={config.io.alpha_dwaa_quality}
                                onChange={e => updateConfig('io', 'alpha_dwaa_quality', parseFloat(e.target.value))}
                                tooltip="Compression quality for EXR DWAA. Higher is better."
                            />
                        </div>
                        <Switch
                            label="Force Overwrite"
                            checked={config.io.force_overwrite}
                            onChange={v => updateConfig('io', 'force_overwrite', v)}
                            tooltip="Overwrite existing files in the output directory."
                        />
                    </div>
                </Section>

                {/* 2. Subject Assignment */}
                <Section title="Subject Assignment (Mask-First)" defaultOpen={true} tooltip="Import one or more keyframe masks before running.">
                    <div className="space-y-3">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
                            <Input
                                label="Project File (.vmhqproj)"
                                value={config.project.path}
                                onChange={e => updateConfig('project', 'path', e.target.value)}
                                placeholder="output/project.vmhqproj"
                                tooltip="Optional explicit project path. Leave blank to auto-use output directory."
                            />
                            <Switch
                                label="Require Assignment (--require-assignment/--allow-empty-assignment)"
                                checked={config.assignment.require_assignment}
                                onChange={v => updateConfig('assignment', 'require_assignment', v)}
                                tooltip="CLI parity: --require-assignment / --allow-empty-assignment. Blocks run until at least one keyframe mask is imported."
                            />
                            <Input
                                label="Keyframe Index"
                                type="number"
                                value={assignmentFrame}
                                onChange={e => setAssignmentFrame(parseInt(e.target.value || "0"))}
                                tooltip="Frame index this mask corresponds to."
                            />
                            <Select
                                label="Anchor Type (--assign-kind)"
                                value={assignmentKind}
                                onChange={e => setAssignmentKind(e.target.value as 'initial' | 'correction')}
                                options={[
                                    { value: 'initial', label: 'Initial Anchor' },
                                    { value: 'correction', label: 'Correction Anchor' },
                                ]}
                                tooltip="CLI parity: --assign-kind {initial,correction}. Use correction anchors to patch drift and reprocess only the affected range."
                            />
                            <Input
                                label="Mask Path"
                                value={assignmentMaskPath}
                                onChange={e => setAssignmentMaskPath(e.target.value)}
                                placeholder="D:\\path\\to\\mask.png"
                                tooltip="Filesystem path to keyframe mask image."
                            />
                            <Switch
                                label="Auto-Apply Suggested Range (--apply-suggested-range/--no-apply-suggested-range)"
                                checked={autoApplySuggestedRange}
                                onChange={setAutoApplySuggestedRange}
                                tooltip="CLI parity: --apply-suggested-range / --no-apply-suggested-range when importing correction anchors."
                            />
                        </div>
                        <div className="flex gap-2">
                            <button
                                type="button"
                                onClick={handleImportAssignment}
                                disabled={assignmentBusy}
                                className="px-3 py-2 rounded bg-brand-500 hover:bg-brand-600 text-white text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                            >
                                {assignmentLoading ? <FaSpinner className="animate-spin" /> : <FaUpload />}
                                Import Mask
                            </button>
                            <button
                                type="button"
                                onClick={() => {
                                    setAssignmentLoading(true)
                                    refreshProjectSummary().finally(() => setAssignmentLoading(false))
                                }}
                                disabled={assignmentBusy}
                                className="px-3 py-2 rounded bg-gray-700 hover:bg-gray-600 text-gray-200 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                            >
                                <FaSync />
                                Refresh
                            </button>
                        </div>
                        <div className="rounded border border-gray-700 bg-gray-900/50 p-3 space-y-3">
                            <div className="flex flex-wrap gap-2 items-center justify-between">
                                <div className="text-sm font-semibold text-gray-200">Initial Mask Builder (Phase 3)</div>
                                <div className="flex gap-2">
                                    <button
                                        type="button"
                                        onClick={handleLoadBuilderFrame}
                                        disabled={assignmentBusy}
                                        className="px-3 py-2 rounded bg-gray-700 hover:bg-gray-600 text-gray-100 text-xs font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        {builderLoadingFrame ? "Loading..." : "Load Frame"}
                                    </button>
                                    <button
                                        type="button"
                                        onClick={handleBuildMaskFromPrompts}
                                        disabled={assignmentBusy || !builderFrameDataUrl || !builderBox}
                                        className="px-3 py-2 rounded bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        {builderBuildingMask ? "Building..." : "Build + Import Mask"}
                                    </button>
                                </div>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-5 gap-2">
                                <div className="md:col-span-4">
                                    <Input
                                        label="Prompt Auto-Detect"
                                        value={builderPrompt}
                                        onChange={e => setBuilderPrompt(e.target.value)}
                                        placeholder="person, person left, person center"
                                        tooltip="Suggest subject boxes from text prompt. Works best with people prompts."
                                    />
                                </div>
                                <div className="flex items-end">
                                    <button
                                        type="button"
                                        onClick={handleSuggestBuilderBoxes}
                                        disabled={assignmentBusy || !builderFrameDataUrl}
                                        className="w-full px-3 py-2 rounded bg-purple-600 hover:bg-purple-500 text-white text-xs font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        {builderSuggestingBoxes ? "Detecting..." : "Suggest Boxes"}
                                    </button>
                                </div>
                            </div>
                            {builderCandidates.length > 0 && (
                                <div className="rounded border border-gray-700/80 bg-gray-900/70 p-2">
                                    <div className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-2">
                                        Prompt Candidates
                                    </div>
                                    <div className="flex flex-wrap gap-2">
                                        {builderCandidates.map((cand, idx) => (
                                            <button
                                                key={`cand-${idx}`}
                                                type="button"
                                                onClick={() => applyBuilderCandidate(cand, idx)}
                                                className="px-2.5 py-1.5 rounded border border-gray-600 bg-gray-800 hover:bg-gray-700 text-xs text-gray-100"
                                                title={`${cand.source} (${cand.label}) score=${cand.score.toFixed(3)}`}
                                            >
                                                #{idx + 1} {cand.label} ({Math.round(cand.score * 100)}%)
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}
                            <div className="grid grid-cols-1 md:grid-cols-5 gap-2">
                                <Select
                                    label="Mask Builder Backend"
                                    value={builderBackend}
                                    onChange={e => setBuilderBackend(e.target.value as BuilderBackend)}
                                    options={[
                                        { value: 'grabcut', label: 'GrabCut (Fast)' },
                                        { value: 'sam', label: 'SAM (Phase 3)' },
                                    ]}
                                    tooltip="Phase 3: optional SAM backend for higher-quality prompt masks."
                                />
                                <Select
                                    label="Builder Tool"
                                    value={builderTool}
                                    onChange={e => setBuilderTool(e.target.value as BuilderTool)}
                                    options={[
                                        { value: 'box', label: 'Draw Box' },
                                        { value: 'fg', label: 'Add FG Points' },
                                        { value: 'bg', label: 'Add BG Points' },
                                    ]}
                                    tooltip="Draw one rough subject box first, then add positive/negative points."
                                />
                                <Input
                                    label="Point Radius (px)"
                                    type="number"
                                    value={builderPointRadius}
                                    onChange={e => setBuilderPointRadius(parseInt(e.target.value || "8"))}
                                    tooltip="Brush size for FG/BG point prompts."
                                />
                                <Input
                                    label="Iterations"
                                    type="number"
                                    value={builderIterCount}
                                    onChange={e => setBuilderIterCount(parseInt(e.target.value || "5"))}
                                    tooltip="Higher can improve difficult edges (slower)."
                                />
                                <div className="flex items-end">
                                    <button
                                        type="button"
                                        onClick={clearBuilderPrompts}
                                        disabled={assignmentBusy}
                                        className="w-full px-3 py-2 rounded bg-gray-800 hover:bg-gray-700 text-gray-100 text-xs font-semibold border border-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Clear Prompts
                                    </button>
                                </div>
                            </div>
                            {builderBackend === 'sam' && (
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 rounded border border-gray-700/70 bg-gray-900/70 p-2">
                                    <Input
                                        label="SAM Model ID / Path"
                                        value={builderSamModelId}
                                        onChange={e => setBuilderSamModelId(e.target.value)}
                                        placeholder="facebook/sam-vit-base"
                                        tooltip="Use a local path or HuggingFace model id for SAM."
                                    />
                                    <Switch
                                        label="Local Files Only"
                                        checked={builderSamLocalOnly}
                                        onChange={setBuilderSamLocalOnly}
                                        tooltip="Recommended: ON. Avoids downloads and uses only local model files."
                                    />
                                </div>
                            )}
                            {builderFrameDataUrl && builderFrameSize ? (
                                <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
                                    <div>
                                        <div className="text-xs text-gray-400 mb-1">
                                            Step 1: Prompt auto-detect or draw a subject box. Step 2: Add FG/BG points if needed.
                                        </div>
                                        <div
                                            className={`relative inline-block border border-gray-700 rounded overflow-hidden ${builderTool === 'box' ? 'cursor-crosshair' : 'cursor-cell'}`}
                                            onMouseDown={handleBuilderMouseDown}
                                            onMouseMove={handleBuilderMouseMove}
                                            onMouseUp={handleBuilderMouseUp}
                                            onMouseLeave={handleBuilderMouseLeave}
                                            onClick={handleBuilderClick}
                                        >
                                            <img
                                                ref={builderImgRef}
                                                src={builderFrameDataUrl}
                                                alt="Assignment frame preview"
                                                className="block max-h-[420px] w-auto select-none"
                                                draggable={false}
                                            />
                                            <svg
                                                className="absolute inset-0 w-full h-full pointer-events-none"
                                                viewBox={`0 0 ${builderFrameSize.width} ${builderFrameSize.height}`}
                                                preserveAspectRatio="xMinYMin meet"
                                            >
                                                {activeBuilderBox && (
                                                    <rect
                                                        x={Math.min(activeBuilderBox.x0, activeBuilderBox.x1)}
                                                        y={Math.min(activeBuilderBox.y0, activeBuilderBox.y1)}
                                                        width={Math.max(1, Math.abs(activeBuilderBox.x1 - activeBuilderBox.x0))}
                                                        height={Math.max(1, Math.abs(activeBuilderBox.y1 - activeBuilderBox.y0))}
                                                        fill="rgba(56, 189, 248, 0.18)"
                                                        stroke="rgb(56, 189, 248)"
                                                        strokeWidth={2}
                                                    />
                                                )}
                                                {builderFgPoints.map((p, idx) => (
                                                    <circle
                                                        key={`fg-${idx}`}
                                                        cx={p.x}
                                                        cy={p.y}
                                                        r={Math.max(2, builderPointRadius)}
                                                        fill="rgba(34, 197, 94, 0.55)"
                                                        stroke="rgb(34, 197, 94)"
                                                        strokeWidth={1.5}
                                                    />
                                                ))}
                                                {builderBgPoints.map((p, idx) => (
                                                    <circle
                                                        key={`bg-${idx}`}
                                                        cx={p.x}
                                                        cy={p.y}
                                                        r={Math.max(2, builderPointRadius)}
                                                        fill="rgba(239, 68, 68, 0.55)"
                                                        stroke="rgb(239, 68, 68)"
                                                        strokeWidth={1.5}
                                                    />
                                                ))}
                                            </svg>
                                        </div>
                                    </div>
                                    <div>
                                        <div className="text-xs text-gray-400 mb-1">
                                            Prompt summary: box={builderBox ? "yes" : "no"}, FG points={builderFgPoints.length}, BG points={builderBgPoints.length}
                                        </div>
                                        {builderMaskPreviewUrl ? (
                                            <img
                                                src={builderMaskPreviewUrl}
                                                alt="Built mask preview"
                                                className="block max-h-[420px] w-auto border border-gray-700 rounded"
                                            />
                                        ) : (
                                            <div className="h-[220px] border border-dashed border-gray-700 rounded flex items-center justify-center text-xs text-gray-500 px-3 text-center">
                                                Built mask preview appears here after you click "Build + Import Mask".
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ) : (
                                <div className="text-xs text-gray-500">
                                    Click "Load Frame" to start building an initial mask from box and points.
                                </div>
                            )}
                        </div>
                        <div className="rounded border border-gray-700 bg-gray-900/50 p-3 space-y-3">
                            <div className="flex items-center justify-between gap-2">
                                <div className="text-sm font-semibold text-gray-200">
                                    Phase 4: Long-Range Propagation Assist
                                </div>
                                <button
                                    type="button"
                                    onClick={handlePropagateAssignments}
                                    disabled={assignmentBusy}
                                    className="px-3 py-2 rounded bg-teal-600 hover:bg-teal-500 text-white text-xs font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    {propagateRunning ? "Propagating..." : "Propagate Keyframes"}
                                </button>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-5 gap-2">
                                <Select
                                    label="Backend"
                                    value={propagateBackend}
                                    onChange={e => setPropagateBackend(e.target.value as PropagationBackend)}
                                    options={[
                                        { value: 'flow', label: 'Flow (Built-in)' },
                                        { value: 'sam2_video_predictor', label: 'SAM2VideoPredictor' },
                                        { value: 'cutie', label: 'Cutie' },
                                    ]}
                                    tooltip="Phase 4 backend. SAM2/Cutie currently fallback to flow when unavailable."
                                />
                                <Input
                                    label="Range Start"
                                    type="number"
                                    value={propagateFrameStart}
                                    onChange={e => setPropagateFrameStart(parseInt(e.target.value || "0"))}
                                    tooltip="Absolute frame index to start propagation."
                                />
                                <Input
                                    label="Range End"
                                    type="number"
                                    value={propagateFrameEnd}
                                    onChange={e => setPropagateFrameEnd(parseInt(e.target.value || "-1"))}
                                    tooltip="Absolute frame index to end propagation."
                                />
                                <Input
                                    label="Stride"
                                    type="number"
                                    value={propagateStride}
                                    onChange={e => setPropagateStride(parseInt(e.target.value || "8"))}
                                    tooltip="Insert one propagated keyframe every N frames."
                                />
                                <Input
                                    label="Max New Keyframes"
                                    type="number"
                                    value={propagateMaxNewKeyframes}
                                    onChange={e => setPropagateMaxNewKeyframes(parseInt(e.target.value || "24"))}
                                    tooltip="Cap how many propagated correction anchors are added."
                                />
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-4 gap-2">
                                <Input
                                    label="Flow Downscale"
                                    type="number"
                                    step="0.05"
                                    value={propagateFlowDownscale}
                                    onChange={e => setPropagateFlowDownscale(parseFloat(e.target.value || "0.5"))}
                                    tooltip="Smaller is faster; larger can track finer motion."
                                />
                                <Input
                                    label="Min Coverage"
                                    type="number"
                                    step="0.001"
                                    value={propagateFlowMinCoverage}
                                    onChange={e => setPropagateFlowMinCoverage(parseFloat(e.target.value || "0.002"))}
                                    tooltip="Reject propagated masks that become too small."
                                />
                                <Input
                                    label="Max Coverage"
                                    type="number"
                                    step="0.01"
                                    value={propagateFlowMaxCoverage}
                                    onChange={e => setPropagateFlowMaxCoverage(parseFloat(e.target.value || "0.98"))}
                                    tooltip="Reject propagated masks that become unrealistically large."
                                />
                                <Input
                                    label="Flow Feather (px)"
                                    type="number"
                                    value={propagateFlowFeatherPx}
                                    onChange={e => setPropagateFlowFeatherPx(parseInt(e.target.value || "1"))}
                                    tooltip="Softening applied during propagation smoothing."
                                />
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                                <Switch
                                    label="Fallback To Flow"
                                    checked={propagateFallbackToFlow}
                                    onChange={setPropagateFallbackToFlow}
                                    tooltip="If SAM2/Cutie is unavailable, continue with flow backend."
                                />
                                <Switch
                                    label="Overwrite Existing Frames"
                                    checked={propagateOverwriteExisting}
                                    onChange={setPropagateOverwriteExisting}
                                    tooltip="Replace existing assignments if propagated frame indices collide."
                                />
                            </div>
                            <div className="text-xs text-gray-400">
                                Uses current <span className="font-mono">Keyframe Index</span> as the anchor frame.
                            </div>
                        </div>
                        <div className="text-xs text-gray-400">
                            Project: <span className="font-mono text-gray-300">{projectSummary?.project_path || "(not resolved yet)"}</span>
                        </div>
                        <div className="text-xs text-gray-400">
                            Imported keyframes: <span className="font-semibold text-gray-200">{projectSummary?.keyframe_count ?? 0}</span>
                        </div>
                        {suggestedRange && (
                            <div className="text-xs text-blue-400 bg-blue-500/10 border border-blue-500/30 rounded p-2 flex items-center justify-between gap-2">
                                <span>
                                    Suggested reprocess range: <span className="font-mono">{suggestedRange.frame_start}..{suggestedRange.frame_end}</span>
                                </span>
                                <button
                                    type="button"
                                    onClick={() => {
                                        applySuggestedRange(suggestedRange)
                                        setStatus(`Applied suggested reprocess range ${suggestedRange.frame_start}..${suggestedRange.frame_end}.`)
                                    }}
                                    className="px-2 py-1 rounded bg-blue-500 hover:bg-blue-600 text-white text-xs font-medium"
                                >
                                    Apply Range
                                </button>
                            </div>
                        )}
                        {(projectSummary?.keyframes?.length ?? 0) > 0 && (
                            <div className="max-h-32 overflow-auto border border-gray-700 rounded p-2 text-xs">
                                {projectSummary!.keyframes.map(kf => (
                                    <div key={`${kf.frame}:${kf.mask_asset}`} className="text-gray-300 font-mono">
                                        frame={kf.frame} kind={kf.kind} mask={kf.mask_asset}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </Section>

                {/* 3. Background */}
                <Section title="Background Plate" tooltip="Settings for clean plate estimation and handling via inpainting.">
                    <div className="space-y-2">
                        <Switch
                            label="Enable Plate Estimation"
                            checked={config.background.enabled}
                            onChange={v => updateConfig('background', 'enabled', v)}
                            tooltip="Estimate a clean background plate from the video footage."
                        />
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
                            <Input
                                label="Sample Count" type="number"
                                value={config.background.sample_count}
                                onChange={e => updateConfig('background', 'sample_count', parseInt(e.target.value))}
                                tooltip="Number of frames to sample for plate estimation."
                            />
                            <Select
                                label="Occlusion Fallback"
                                value={config.background.occlusion_fallback}
                                onChange={e => updateConfig('background', 'occlusion_fallback', e.target.value)}
                                options={[
                                    { value: 'auto', label: 'Auto' },
                                    { value: 'temporal_extremes', label: 'Temporal Extremes' },
                                    { value: 'patch_inpaint', label: 'Patch Inpaint' },
                                    { value: 'ai_inpaint', label: 'AI Inpaint' }
                                ]}
                                tooltip="Method to fill in background areas that are always occluded."
                            />
                            <Input
                                label="Manual Plate Path"
                                value={config.background.manual_plate_path}
                                onChange={e => updateConfig('background', 'manual_plate_path', e.target.value)}
                                className="col-span-2"
                                tooltip="Optional: Path to a pre-generated clean plate image."
                            />
                        </div>
                    </div>
                </Section>

                {/* 3. ROI */}
                <Section title="ROI & Tracking" tooltip="Subject detection and Region of Interest tracking settings.">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
                        <Input
                            label="Detect Every (frames)" type="number"
                            value={config.roi.detect_every}
                            onChange={e => updateConfig('roi', 'detect_every', parseInt(e.target.value))}
                            tooltip="Run object detection every N frames. Interpolates in between."
                        />
                        <Input
                            label="Pad Ratio" type="number" step="0.05"
                            value={config.roi.pad_ratio}
                            onChange={e => updateConfig('roi', 'pad_ratio', parseFloat(e.target.value))}
                            tooltip="Padding added around the detected subject."
                        />
                        <Select
                            label="Multi-Person Mode"
                            value={config.roi.multi_person}
                            onChange={e => updateConfig('roi', 'multi_person', e.target.value)}
                            options={[
                                { value: 'union_k', label: 'Union of Top K' },
                                { value: 'single', label: 'Single Largest' }
                            ]}
                            tooltip="How to handle multiple people. 'Union of Top K' merges masks."
                        />
                        <Input
                            label="Max Subjects (K)" type="number"
                            value={config.roi.k}
                            onChange={e => updateConfig('roi', 'k', parseInt(e.target.value))}
                            tooltip="Maximum number of subjects to track."
                        />
                    </div>
                </Section>

                {/* 4. Global Pass */}
                <Section title="Global Pass (Pass A)" tooltip="First pass: Generates a coarse, full-frame matte." defaultOpen={true}>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
                        <Select
                            label="Global Model"
                            value={config.global.model}
                            onChange={e => updateConfig('global', 'model', e.target.value)}
                            options={[
                                { value: 'rvm', label: 'RVM (Robust Video Matting)' },
                                { value: 'modnet', label: 'MODNet (Fast)' }
                            ]}
                            tooltip="The AI model used for the initial coarse matte."
                        />
                        <Input
                            label="Long Side (px)" type="number"
                            value={config.global.long_side}
                            onChange={e => updateConfig('global', 'long_side', parseInt(e.target.value))}
                            tooltip="Processing resolution for the global pass. Lower is faster."
                        />
                        <div className="grid grid-cols-2 gap-2">
                            <Input
                                label="Chunk Length" type="number"
                                value={config.global.chunk_len}
                                onChange={e => updateConfig('global', 'chunk_len', parseInt(e.target.value))}
                                tooltip="Frames to process in one batch for temporal consistency."
                            />
                            <Input
                                label="Chunk Overlap" type="number"
                                value={config.global.chunk_overlap}
                                onChange={e => updateConfig('global', 'chunk_overlap', parseInt(e.target.value))}
                                tooltip="Overlap between chunks to blend transitions."
                            />
                        </div>
                    </div>
                </Section>

                {/* 5. Intermediate Pass */}
                {showAdvanced && (
                    <Section title="Intermediate (Pass A')" tooltip="Refines the coarse matte and applies temporal smoothing.">
                        <div className="space-y-2">
                            <Switch
                                label="Enable Intermediate Pass"
                                checked={config.intermediate.enabled}
                                onChange={v => updateConfig('intermediate', 'enabled', v)}
                                tooltip="Turn on/off the intermediate stabilization step."
                            />
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
                                <Select
                                    label="Temporal Smooth"
                                    value={config.intermediate.temporal_smooth}
                                    onChange={e => updateConfig('intermediate', 'temporal_smooth', e.target.value)}
                                    options={[
                                        { value: 'flow', label: 'Optical Flow' },
                                        { value: 'ema', label: 'EMA' },
                                        { value: 'none', label: 'None' }
                                    ]}
                                    tooltip="Method used to smooth the matte over time."
                                />
                                <Input
                                    label="Smooth Strength" type="number" step="0.1"
                                    value={config.intermediate.smooth_strength}
                                    onChange={e => updateConfig('intermediate', 'smooth_strength', parseFloat(e.target.value))}
                                    tooltip="Identify how strongly to smooth. 0 is none."
                                />
                                <Input
                                    label="Processing Resolution" type="number"
                                    value={config.intermediate.long_side}
                                    onChange={e => updateConfig('intermediate', 'long_side', parseInt(e.target.value))}
                                    tooltip="Resolution for this intermediate step."
                                />
                            </div>
                        </div>
                    </Section>
                )}

                {/* 6. Band & Trimap */}
                {showAdvanced && (
                    <Section title="Band & Trimap" tooltip="Generates the trimap (unknown region) for detail refinement.">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
                            <Select
                                label="Band Mode"
                                value={config.band.mode}
                                onChange={e => updateConfig('band', 'mode', e.target.value)}
                                options={[
                                    { value: 'adaptive', label: 'Adaptive' },
                                    { value: 'fixed', label: 'Fixed' }
                                ]}
                                tooltip="'Adaptive' calculates band width based on image content."
                            />
                            <Input
                                label="Feather (px)" type="number"
                                value={config.band.feather_px}
                                onChange={e => updateConfig('band', 'feather_px', parseInt(e.target.value))}
                                tooltip="Pixel width to feather the edges of the band."
                            />
                            <Select
                                label="Trimap Method"
                                value={config.trimap.method}
                                onChange={e => updateConfig('trimap', 'method', e.target.value)}
                                options={[
                                    { value: 'distance_transform', label: 'Distance Transform' },
                                    { value: 'erosion', label: 'Erosion' }
                                ]}
                                tooltip="Algorithm to generate the trimap from the coarse matte."
                            />
                            <Input
                                label="Unknown Width (px)" type="number"
                                value={config.trimap.unknown_width}
                                onChange={e => updateConfig('trimap', 'unknown_width', parseInt(e.target.value))}
                                tooltip="Width of the 'unknown' gray region in the trimap."
                            />
                        </div>
                    </Section>
                )}

                {/* 7. Refine */}
                <Section title="Detail Refinement (Pass B)" tooltip="Second pass: High-resolution matting on image tiles.">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
                        <Select
                            label="Refine Model"
                            value={config.refine.model}
                            onChange={e => updateConfig('refine', 'model', e.target.value)}
                            options={[
                                { value: 'vitmatte', label: 'ViTMatte' },
                                { value: 'matformer', label: 'MatteFormer' }
                            ]}
                            tooltip="High-quality matting model for detail refinement."
                        />
                        <div className="grid grid-cols-2 gap-2">
                            <Input
                                label="Tile Size" type="number"
                                value={config.tiles.tile_size}
                                onChange={e => updateConfig('tiles', 'tile_size', parseInt(e.target.value))}
                                tooltip="Size of image tiles (e.g., 1024, 2048)."
                            />
                            <Input
                                label="Tile Overlap" type="number"
                                value={config.tiles.overlap}
                                onChange={e => updateConfig('tiles', 'overlap', parseInt(e.target.value))}
                                tooltip="Pixel overlap between tiles to prevent seams."
                            />
                        </div>
                    </div>
                </Section>

                <Section
                    title="Matte Tuning"
                    tooltip="Artist-facing matte tuning: trimap width, choke/expand, feather, and XY offsets."
                    defaultOpen={showAdvanced}
                >
                    <div className="space-y-2">
                        <div className="rounded border border-gray-700/60 bg-gray-900/60 p-3">
                            <div className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-2">
                                Presets
                            </div>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                                {MATTE_TUNING_PRESETS.map(preset => {
                                    const active = isMattePresetActive(preset)
                                    return (
                                        <button
                                            key={preset.id}
                                            type="button"
                                            onClick={() => applyMattePreset(preset)}
                                            title={preset.description}
                                            className={`px-3 py-2 rounded text-xs font-semibold border transition-colors ${active
                                                ? 'bg-brand-500/20 border-brand-400 text-brand-200'
                                                : 'bg-gray-800 border-gray-700 text-gray-200 hover:border-gray-500 hover:bg-gray-700'
                                                }`}
                                        >
                                            {preset.label}
                                        </button>
                                    )
                                })}
                                <button
                                    type="button"
                                    onClick={resetMattePreset}
                                    className="px-3 py-2 rounded text-xs font-semibold border border-gray-700 text-gray-200 bg-gray-800 hover:border-gray-500 transition-colors"
                                >
                                    Reset
                                </button>
                            </div>
                            <div className="mt-2 text-xs text-gray-500">
                                Presets set trimap width + matte tuning values together. You can edit any field afterward.
                            </div>
                        </div>
                        <Switch
                            label="Enable Matte Tuning (--matte-tuning/--no-matte-tuning)"
                            checked={config.matte_tuning.enabled}
                            onChange={v => updateConfig('matte_tuning', 'enabled', v)}
                            tooltip="CLI parity: --matte-tuning / --no-matte-tuning."
                        />
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
                            <Input
                                label="Refine Trimap Width (--unknown-band-px / --mt-trimap-width-px)"
                                type="number"
                                value={config.refine.unknown_band_px}
                                onChange={e => updateConfig('refine', 'unknown_band_px', parseInt(e.target.value || "0"))}
                                tooltip="Controls boundary band width for high-res refinement (maps to refine.unknown_band_px)."
                            />
                            <Input
                                label="Mask Shrink/Grow (--mt-shrink-grow-px)"
                                type="number"
                                value={config.matte_tuning.shrink_grow_px}
                                onChange={e => updateConfig('matte_tuning', 'shrink_grow_px', parseInt(e.target.value || "0"))}
                                tooltip="Positive grows matte, negative shrinks matte."
                            />
                            <Input
                                label="Edge Feather (--mt-feather-px)"
                                type="number"
                                value={config.matte_tuning.feather_px}
                                onChange={e => updateConfig('matte_tuning', 'feather_px', parseInt(e.target.value || "0"))}
                                tooltip="Applies final Gaussian feather to matte edges."
                            />
                            <Input
                                label="Offset X (--mt-offset-x-px)"
                                type="number"
                                value={config.matte_tuning.offset_x_px}
                                onChange={e => updateConfig('matte_tuning', 'offset_x_px', parseInt(e.target.value || "0"))}
                                tooltip="Shifts matte in X pixels."
                            />
                            <Input
                                label="Offset Y (--mt-offset-y-px)"
                                type="number"
                                value={config.matte_tuning.offset_y_px}
                                onChange={e => updateConfig('matte_tuning', 'offset_y_px', parseInt(e.target.value || "0"))}
                                tooltip="Shifts matte in Y pixels."
                            />
                        </div>
                    </div>
                </Section>


                {/* 8. Temporal Stability */}
                {showAdvanced && (
                    <Section title="Temporal Stability (Pass C)" tooltip="Final pass: Ensures temporal consistency on the high-res matte.">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
                            <Select
                                label="Method"
                                value={config.temporal.method}
                                onChange={e => updateConfig('temporal', 'method', e.target.value)}
                                options={[
                                    { value: 'frequency_separation', label: 'Frequency Separation' },
                                    { value: 'none', label: 'None' }
                                ]}
                                tooltip="Algorithm for temporal stability."
                            />
                            <Input
                                label="Detail Blend Strength" type="number" step="0.05"
                                value={config.temporal.detail_blend_strength}
                                onChange={e => updateConfig('temporal', 'detail_blend_strength', parseFloat(e.target.value))}
                                tooltip="Strength of blending for high-frequency details."
                            />
                            <Input
                                label="Structural Blend Strength" type="number" step="0.05"
                                value={config.temporal.structural_blend_strength}
                                onChange={e => updateConfig('temporal', 'structural_blend_strength', parseFloat(e.target.value))}
                                tooltip="Strength of blending for structural elements."
                            />
                        </div>
                    </Section>
                )}

                {/* 9. Despill & Output */}
                <Section title="Post-Processing" tooltip="Final color correction (despill) and outputting foreground.">
                    <div className="space-y-2">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
                            <Switch
                                label="Enable Despill"
                                checked={config.postprocess.despill.enabled}
                                onChange={v => updateNestedConfig('postprocess', 'despill', 'enabled', v)}
                                tooltip="Remove green/blue spill from the foreground subject."
                            />
                            {showAdvanced && (
                                <>
                                    <Select
                                        label="Despill Method"
                                        value={config.postprocess.despill.method}
                                        onChange={e => updateNestedConfig('postprocess', 'despill', 'method', e.target.value)}
                                        options={[
                                            { value: 'advanced', label: 'Advanced' },
                                            { value: 'simple', label: 'Simple' },
                                            { value: 'none', label: 'None' }
                                        ]}
                                        tooltip="Algorithm used for despill operation."
                                    />
                                    <Input
                                        label="Luma Bias" type="number" step="0.05"
                                        value={config.postprocess.despill.luma_bias}
                                        onChange={e => updateNestedConfig('postprocess', 'despill', 'luma_bias', parseFloat(e.target.value))}
                                        tooltip="Bias to adjust brightness of despilled areas."
                                    />
                                </>
                            )}
                            <div className="col-span-2 border-t border-gray-700/50 pt-2">
                                <Switch
                                    label="Generate FG Output"
                                    checked={config.postprocess.fg_output.enabled}
                                    onChange={v => updateNestedConfig('postprocess', 'fg_output', 'enabled', v)}
                                    tooltip="Save the despilled foreground as a separate image sequence."
                                />
                            </div>
                        </div>
                    </div>
                </Section>

                {/* 10. Preview & Runtime */}
                <Section title="Runtime & Preview" tooltip="Hardware configuration and live preview settings." defaultOpen={true}>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
                        <Select
                            label="Device"
                            value={config.runtime.device}
                            onChange={e => updateConfig('runtime', 'device', e.target.value)}
                            options={[
                                { value: 'cuda', label: 'CUDA (NVIDIA)' },
                                { value: 'cpu', label: 'CPU' }
                            ]}
                            tooltip="Hardware to run the inference on."
                        />
                        <Select
                            label="Precision"
                            value={config.runtime.precision}
                            onChange={e => updateConfig('runtime', 'precision', e.target.value)}
                            options={[
                                { value: 'fp16', label: 'FP16 (Half)' },
                                { value: 'fp32', label: 'FP32 (Full)' }
                            ]}
                            tooltip="FP16 is faster and uses less VRAM; FP32 is more precise."
                        />
                        {showAdvanced && (
                            <Input
                                label="IO Workers" type="number"
                                value={config.runtime.workers_io}
                                onChange={e => updateConfig('runtime', 'workers_io', parseInt(e.target.value))}
                                tooltip="Number of threads for image loading/saving."
                            />
                        )}
                        <div className="col-span-2 space-y-2 border-t border-gray-700/50 pt-2 mt-1">
                            <Switch
                                label="Enable Live Preview"
                                checked={config.preview.enabled}
                                onChange={v => updateConfig('preview', 'enabled', v)}
                                tooltip="Show processing progress window."
                            />
                            <div className="grid grid-cols-2 gap-2">
                                {showAdvanced && (
                                    <>
                                        <Input
                                            label="Preview Scale" type="number" step="0.1"
                                            value={config.preview.scale}
                                            onChange={e => updateConfig('preview', 'scale', parseFloat(e.target.value))}
                                            tooltip="Scale factor for the preview window (0.5 = half size)."
                                        />
                                        <Input
                                            label="Update Every (frames)" type="number"
                                            value={config.preview.every || 1}
                                            onChange={e => updateConfig('preview', 'every', parseInt(e.target.value))}
                                            tooltip="Update preview window every N frames."
                                        />
                                        <Input
                                            label="Preview Modes"
                                            value={config.preview.modes?.join(", ") || ""}
                                            onChange={e => updateConfig('preview', 'modes', e.target.value.split(",").map(s => s.trim()))}
                                            placeholder="checker, alpha, white"
                                            className="col-span-2"
                                            tooltip="Visualization modes (comma separated): checker, alpha, white, etc."
                                        />
                                    </>
                                )}
                            </div>
                        </div>
                    </div>
                </Section>

                <Section
                    title="QC & Regression Gates"
                    tooltip="Option B QC metrics and regression thresholds. Enable hard-fail to stop runs that exceed limits."
                    defaultOpen={showAdvanced}
                >
                    <div className="space-y-2">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
                            <Switch
                                label="Enable QC Metrics (--qc/--no-qc)"
                                checked={config.qc.enabled}
                                onChange={v => updateConfig('qc', 'enabled', v)}
                                tooltip="CLI parity: --qc / --no-qc. Compute per-frame QC metrics and write QC artifacts."
                            />
                            <Switch
                                label="Fail On Regression (--qc-fail-on-regression/--no-qc-fail-on-regression)"
                                checked={config.qc.fail_on_regression}
                                onChange={v => updateConfig('qc', 'fail_on_regression', v)}
                                tooltip="CLI parity: --qc-fail-on-regression / --no-qc-fail-on-regression. If any QC gate fails, the job fails."
                            />
                        </div>
                        {showAdvanced && (
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
                                <Input
                                    label="QC Output Subdir (config: qc.output_subdir)"
                                    value={config.qc.output_subdir}
                                    onChange={e => updateConfig('qc', 'output_subdir', e.target.value)}
                                    tooltip="Config field parity: qc.output_subdir (YAML/UI). No direct CLI flag."
                                />
                                <Input
                                    label="Metrics Filename (config: qc.metrics_filename)"
                                    value={config.qc.metrics_filename}
                                    onChange={e => updateConfig('qc', 'metrics_filename', e.target.value)}
                                    tooltip="Config field parity: qc.metrics_filename (YAML/UI). No direct CLI flag."
                                />
                                <Input
                                    label="Report Filename (config: qc.report_filename)"
                                    value={config.qc.report_filename}
                                    onChange={e => updateConfig('qc', 'report_filename', e.target.value)}
                                    tooltip="Config field parity: qc.report_filename (YAML/UI). No direct CLI flag."
                                />
                                <Input
                                    label="Output Roundtrip Samples (--qc-sample-output-frames)"
                                    type="number"
                                    value={config.qc.sample_output_frames}
                                    onChange={e => updateConfig('qc', 'sample_output_frames', parseInt(e.target.value || "0"))}
                                    tooltip="CLI parity: --qc-sample-output-frames. Number of output frames to sample for roundtrip MAE checks."
                                />
                                <Input
                                    label="Max Roundtrip MAE (--qc-max-output-roundtrip-mae)"
                                    type="number"
                                    step="0.0001"
                                    value={config.qc.max_output_roundtrip_mae}
                                    onChange={e => updateConfig('qc', 'max_output_roundtrip_mae', parseFloat(e.target.value))}
                                    tooltip="CLI parity: --qc-max-output-roundtrip-mae. Maximum allowed MAE between in-memory alpha and written output."
                                />
                                <Input
                                    label="Alpha Range Epsilon (--qc-alpha-range-eps)"
                                    type="number"
                                    step="0.0001"
                                    value={config.qc.alpha_range_eps}
                                    onChange={e => updateConfig('qc', 'alpha_range_eps', parseFloat(e.target.value))}
                                    tooltip="CLI parity: --qc-alpha-range-eps. Allowed alpha range slack outside [0,1]."
                                />
                                <Input
                                    label="Max P95 Flicker (--qc-max-p95-flicker)"
                                    type="number"
                                    step="0.0001"
                                    value={config.qc.max_p95_flicker}
                                    onChange={e => updateConfig('qc', 'max_p95_flicker', parseFloat(e.target.value))}
                                    tooltip="CLI parity: --qc-max-p95-flicker. Maximum allowed 95th percentile frame-to-frame flicker."
                                />
                                <Input
                                    label="Max P95 Edge Flicker (--qc-max-p95-edge-flicker)"
                                    type="number"
                                    step="0.0001"
                                    value={config.qc.max_p95_edge_flicker}
                                    onChange={e => updateConfig('qc', 'max_p95_edge_flicker', parseFloat(e.target.value))}
                                    tooltip="CLI parity: --qc-max-p95-edge-flicker. Maximum allowed 95th percentile edge-band flicker."
                                />
                                <Input
                                    label="Min Mean Edge Confidence (--qc-min-mean-edge-confidence)"
                                    type="number"
                                    step="0.0001"
                                    value={config.qc.min_mean_edge_confidence}
                                    onChange={e => updateConfig('qc', 'min_mean_edge_confidence', parseFloat(e.target.value))}
                                    tooltip="CLI parity: --qc-min-mean-edge-confidence. Minimum allowed mean edge confidence."
                                />
                                <Input
                                    label="Band Spike Ratio (--qc-band-spike-ratio)"
                                    type="number"
                                    step="0.1"
                                    value={config.qc.band_spike_ratio}
                                    onChange={e => updateConfig('qc', 'band_spike_ratio', parseFloat(e.target.value))}
                                    tooltip="CLI parity: --qc-band-spike-ratio. Coverage ratio above running mean that counts as a spike."
                                />
                                <Input
                                    label="Max Band Spike Frames (--qc-max-band-spike-frames)"
                                    type="number"
                                    value={config.qc.max_band_spike_frames}
                                    onChange={e => updateConfig('qc', 'max_band_spike_frames', parseInt(e.target.value || "0"))}
                                    tooltip="CLI parity: --qc-max-band-spike-frames. Maximum number of spike frames allowed before gate fail."
                                />
                            </div>
                        )}
                    </div>
                </Section>

                <div className="pt-4 sticky bottom-6 z-10">
                    <button
                        type="submit"
                        disabled={loading || !config.io.input}
                        className="w-full py-4 bg-brand-500 hover:bg-brand-600 text-white rounded-lg font-bold text-lg shadow-lg shadow-brand-500/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 transition-colors backdrop-blur-sm"
                    >
                        {loading ? <FaSpinner className="animate-spin" /> : <FaPlay />}
                        Start Pipeline
                    </button>
                </div>
            </form >
        </div >
    )
}
