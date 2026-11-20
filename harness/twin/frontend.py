"""Capture front end, as Vocali's VoiceEngine.accept: 512-sample blocks (32 ms at 16 kHz); the VAD sees the block
after a 150 Hz high-pass and a per-block peak gain (to 0.85, at most x128); the recogniser gets the unfiltered
block. Silero VAD through sherpa-onnx with the app's settings (Sources/SpeakerNative/VoiceNative.c).

Output: one (end_sample, speech) per block, `speech` = the detector's state after the block, which is exactly
what Transcriber.accept receives (`inSpeech && detected` reduces to `detected`).
"""
import math

import numpy as np

SR = 16000
BLOCK = 512


def vad_settings(vad):
    """HarnessConfig.vad -> sherpa-onnx SileroVadModelConfig fields (window 512 fixed, as in VoiceNative.c)."""
    return {"threshold": vad["threshold"], "min_silence_duration": vad["minSilence"],
            "min_speech_duration": vad["minSpeech"], "window_size": 512, "max_speech_duration": vad["maxSpeech"]}


RESET_SAMPLES = 128000  # the detector is reset between utterances after this many samples


def high_pass(audio, f=150.0, rate=16000.0):
    """Vocali's HighPass (two biquads, Butterworth 4th order at 150 Hz). The filter is linear and its state carries
    across blocks, so filtering the whole signal at once equals the block-by-block Swift code (up to float32
    rounding)."""
    from scipy.signal import sosfilt

    sos = []
    for q in (0.5412, 1.3066):
        w = 2 * math.pi * f / rate
        alpha, c = math.sin(w) / (2 * q), math.cos(w)
        a0 = 1 + alpha
        sos.append([(1 + c) / 2 / a0, -(1 + c) / a0, (1 + c) / 2 / a0, 1.0, -2 * c / a0, (1 - alpha) / a0])
    return sosfilt(np.array(sos), np.asarray(audio, np.float64)).astype(np.float32)


def vad_blocks(audio, vad_model, vad):
    """audio: float32 mono 16 kHz; vad: HarnessConfig.vad. Returns list of (end_sample, speech)."""
    import sherpa_onnx

    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = vad_model
    for k, v in vad_settings(vad).items():
        setattr(cfg.silero_vad, k, v)
    cfg.sample_rate = SR
    det = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)
    filtered = high_pass(audio)
    out, since_reset, in_speech = [], 0, False
    for i in range(len(audio) // BLOCK):
        raw = filtered[i * BLOCK:(i + 1) * BLOCK]
        peak = float(np.max(np.abs(raw)))
        gain = min(128.0, 0.85 / max(peak, 1e-8))
        det.accept_waveform((raw * np.float32(gain)).astype(np.float32))
        speech = bool(det.is_speech_detected())
        while not det.empty():  # VLVoiceClear: drop finished segments
            det.pop()
        since_reset += BLOCK
        out.append(((i + 1) * BLOCK, speech))
        if speech and not in_speech:
            in_speech = True
        elif not speech and in_speech:
            in_speech = False
        if since_reset >= RESET_SAMPLES and not in_speech:
            det.reset()
            since_reset = 0
    return out
