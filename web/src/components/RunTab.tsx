import { useState, useEffect, useCallback } from 'react'
import { FaPlay, FaSpinner, FaExclamationCircle, FaFileVideo } from 'react-icons/fa'
import { VideoMatteConfig } from '../types'
import { Section } from './ui/Section'
import { Input } from './ui/Input'
import { Select } from './ui/Select'
import { Switch } from './ui/Switch'

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
    const [showAdvanced, setShowAdvanced] = useState(true)

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

    async function handleSubmit(e: React.FormEvent) {
        e.preventDefault()
        setLoading(true)
        setError(null)

        try {
            const res = await fetch('/api/jobs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config })
            })

            if (!res.ok) throw new Error(await res.text())

            onSuccess()
        } catch (err: any) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

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

                {/* 2. Background */}
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
