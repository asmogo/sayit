# Fish Audio Drama 3 on your Mac

Say It runs the community [Q4](https://huggingface.co/cguldogan/drama-3-mlx-q4)
and [Q8](https://huggingface.co/cguldogan/drama-3-mlx-q8) conversions directly
in its native Swift/MLX backend. No Python installation, companion process,
local HTTP service, or terminal command is required.

Both variants have the same 35B-parameter architecture, with 3B active
parameters per token. Q4 uses smaller expert weights.

| Variant | Approximate download | Suggested unified memory |
| --- | --- | --- |
| Drama 3 Q4 | 25.9 GB | 48 GB or more |
| Drama 3 Q8 | 42.0 GB | 64 GB or more |

These memory recommendations come from the conversion's author. Both models
remain experimental while broader audio qualification is pending.

## Use

Open Settings → Models, find **Fish Audio Drama 3 Q4** or **Q8**, review the
license, download one variant, and select it. Existing installations of these
conversions work unchanged. Updating the app does not require a new model
download. The first load can take a while.

The development app uses a separate model folder from the installed release.
Keep existing downloads; use the appropriate app profile or import the local
snapshot instead of downloading another copy.

## Supported behavior

- Buffered read-aloud, generated voices, voice discovery, and voice cloning.
- A single reference recording of up to 30 seconds, with its exact transcript.
- Native tokenizer, Qwen3.5 mixture-of-experts backbone, temporal audio-code
  decoder, and Fish S1 DAC encode/decode, all on your Mac.
- Mixed per-layer quantization: Q4 expert matrices with Q8 embeddings, or Q8.
- Temperature, top-p, top-k, and seeded sampling.
- Cancellation between prompt chunks and generated frames; normal model
  switching and idle unloading release model memory.
- Text is split into bounded chunks by Say It's speech pipeline.

The original FP8 `fishaudio/drama-3` checkpoint is not compatible with this
adapter. Use the pinned converted Q4/Q8 checkpoints. Multi-reference editing
and multi-speaker workflows are not exposed in Say It's voice UI.

The model weights retain the Fish Audio Community License. Native code
attributions are in [ThirdPartyNotices.md](../ThirdPartyNotices.md).

## Developer validation

`swift test --disable-sandbox --filter Drama3Tests` checks synthetic Q4/Q8
weights against reference backbone, cache, and audio-decoder outputs, plus
prompt formatting, validation and sampling behavior. Fixtures are deterministic
and do not require internet access, Python, or downloaded model weights.

`Scripts/generate-drama3-fixtures.py` is an optional developer tool for
regenerating these tiny fixtures with the pinned Python reference implementation.
It is not part of the app runtime and never modifies downloaded checkpoints.

The opt-in `InstalledModelSmokeTests` exercise the normal synthesis pipeline
with `SAYIT_MODEL_AUDIT_ID` and `SAYIT_MODEL_AUDIT_ROOT`. Never remove model
snapshots after validation; reuse them for subsequent tests and local app use.
