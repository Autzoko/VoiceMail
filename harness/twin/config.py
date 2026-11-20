"""HarnessConfig, Python side: same sections, keys, defaults and validation as Vocali's
Sources/SpeakerCore/HarnessConfig.swift (missing keys keep defaults, unknown keys are an error).
`default.json` here is a copy of Vocali's config/harness/default.json; tests/test_twin.py checks they agree
when the Vocali checkout is present.
"""
import copy
import json
from pathlib import Path

DEFAULTS = {
    "router": {"margin": 1.5, "decideWithin": 3.0, "speakerWeight": 1.0, "recentWeight": 0.5, "scriptWeight": 1.0,
               "arabicShareHigh": 0.6, "arabicShareLow": 0.1, "scriptMinLetters": 3, "rememberEvidence": 1.0,
               "probabilityFloor": 0.0025},
    "transcription": {"partialInterval": None, "firstPartialAfter": 0.8, "lockAfter": 2.0, "switchConfidence": 0.75,
                      "maximumUtterance": 12.0, "commitMargin": 0.8, "partialCostFactor": 1.2},
    "activity": {"on": 0.5, "off": 0.35, "confirm": 0.6, "hold": 0.24},
    "voices": {"accept": 0.55, "keep": 0.42, "learn": 0.65},
    "translation": {"contextSentences": 2},
}
# Not in the Swift HarnessConfig (yet): engine constants the twin needs, fixed to the app's values.
ENGINE = {
    "decoderPartialInterval": 0.9,   # QwenDecoder .large / .audarTurbo (RoutedDecoder takes the max)
    "prerollSamples": 8192,          # VoiceEngine asrPreroll
    "minFinalSeconds": 0.4,          # Transcriber.finalize
    "dropBehind": 3,                 # Transcriber: finals skipped when >= 3 newer requests wait
    "alignerWindow": 0.16,           # AppModel: TranscriptAligner window with Nemotron
}


class ConfigError(ValueError):
    pass


def load(source=None) -> dict:
    """source: None (defaults), a path, or a dict (partial)."""
    cfg = copy.deepcopy(DEFAULTS)
    if source is None:
        return cfg
    data = source if isinstance(source, dict) else json.loads(Path(source).read_text())
    for section, values in data.items():
        if section not in cfg:
            raise ConfigError(f"unknown section {section!r}")
        for k, v in values.items():
            if k not in cfg[section]:
                raise ConfigError(f"unknown key {section}.{k}")
            cfg[section][k] = v
    return cfg


def dump(cfg) -> str:
    """Canonical JSON (sorted keys, nulls dropped), as HarnessConfig.json() writes it."""
    clean = {s: {k: v for k, v in sorted(vals.items()) if v is not None} for s, vals in sorted(cfg.items())}
    return json.dumps(clean, indent=2, sort_keys=True)
