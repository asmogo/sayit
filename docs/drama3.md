# Fish Audio Drama 3 on your Mac

Say It runs Drama 3 through the native Swift/MLX `DramaTTS` library from
[ArrayPress swift-tts](https://github.com/arraypress/swift-tts), pinned to
v3.2.1 (`338c0bc8794ada080ffbf7f492a8b4044f31f709`). No Python runtime,
companion, or local HTTP server is required.

## Existing models

The existing `cguldogan/drama-3-mlx-q4` and `cguldogan/drama-3-mlx-q8`
snapshots are reused directly. Updating the app does not require downloading,
converting, or replacing them. The loader adapts names, semantic-head row
ordering, and zero-centred audio-tower normalization in memory. Per-layer
quantization is preserved. No checkpoint files are written.

| Variant | Approximate download | Suggested unified memory |
| --- | --- | --- |
| Drama 3 Q4 | 25.9 GB | 48 GB or more |
| Drama 3 Q8 | 42.0 GB | 64 GB or more |

The original FP8 Fish Audio release is not supported by this adapter. Model
weights retain the Fish Audio Community License. Installation already records
licence acceptance. See `ThirdPartyNotices.md` for code attribution.

## Inference and playback

`DramaLibraryModel` adapts the library's actor and streamed sample buffers to
Say It's synthesis pipeline. The custom backbone and audio tower have been
replaced by the library. Only its DramaTTS and SpeechSynthesizer targets are
included under `Packages/DramaTTS`; unrelated engines and dependencies are
excluded. `UPSTREAM.md` records the source revision and local compatibility
patches so later library updates can be reviewed against a known version.

- Temperature 1.0, top-p 0.9 and top-k 30 remain the defaults; voice tuning
  overrides still apply. The library's 1024-frame cap bounds each internal chunk.
- Voice references and prompt state are cached in memory by the library.
  Automatic voice mode still creates one reference for the request; saved
  profiles reuse their supplied reference. References remain limited to 30 seconds.
- The library decodes pieces of approximately 0.74 seconds while generating the
  rest. Say It's existing adaptive buffering, PCM conditioning, playback rate,
  and paragraph pauses remain in control of playback.
- On a fresh request, delivered-piece timing can qualify playback before the
  first logical text chunk finishes. Startup requires at least three seconds of
  delivered audio, 25% post-prefill generation headroom at the selected playback speed, and
  a reserve covering at least two observed delivery gaps. Slow or insufficient
  measurements retain the existing completed-chunk buffering decision.
- Embedded newlines are flattened before prompting, following the upstream
  Drama 3 fix for prematurely ended turns. Paragraph metadata remains separate.
- The library's bounded silence-rewind protection is enabled. Exhausting the
  frame limit or retries returns an error instead of silently claiming that
  incomplete speech succeeded.
- The backend disables TF32 before MLX initialization. On supported hardware,
  reduced-precision float32 multiplication otherwise introduces a larger
  difference between streamed and complete codec decoding.

The evaluated dependencies follow upstream Say It: MLX Audio Swift `624fd54`
and MLX Swift `ac4ae9a3`. MLX Swift LM is pinned to unmodified upstream revision
`ee673d6a`, before its requirement for newer MLX APIs. Its gated-delta and
expert-layer implementations are identical to those in 3.32.3. The sampling implementation and runtime differ from the
previous custom adapter: generated takes are not promised to be bit-identical.
Keeping the weights and sampling controls is not by itself a proof of identical
subjective voice quality; validate speech completeness and listen to new builds.

## Validation

The package suite includes the library's upstream tests and Say It's checkpoint
compatibility tests. Real model checks are opt-in and read existing snapshots:

```sh
MLX_ENABLE_TF32=0 \
SAYIT_DRAMATTS_MODEL='/path/to/existing/q4/snapshot' \
SAYIT_DRAMATTS_FIXTURE='/path/to/baseline.safetensors' \
swift test --disable-sandbox -c release --filter DramaLibraryTests
```

The fixture contains a fixed reference recording and its encoded codes. It is a
local validation artifact outside the model directory. GPU tests need Metal
shaders matching the linked MLX version; the app build produces them.

`DramaLibraryTests` covers checkpoint row ordering, normalization and quantization
mapping, streamed-versus-whole codec fidelity across the attention window,
finite Q4 output, cancellation and recovery. `InstalledModelSmokeTests` exercises
the production pipeline; use `SAYIT_MODEL_AUDIT_PLAYBACK=1` to check muted live
playback and rebuffer counts. Q8 inference was not exercised for this migration.

Settings → Diagnostics retains model-load, voice-preparation, first-audio,
total-generation, first-playback and rebuffer events. A fresh model load,
automatic voice creation, longer text, and other GPU work can increase startup.
Never remove downloaded snapshots after validation.
