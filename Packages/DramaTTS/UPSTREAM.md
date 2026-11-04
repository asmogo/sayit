# DramaTTS source dependency

Source: https://github.com/arraypress/swift-tts
Revision: 338c0bc8794ada080ffbf7f492a8b4044f31f709 (v3.2.1).

Only DramaTTS and its dependency-free SpeechSynthesizer target are included.
The package manifest omits unrelated engines and pins the evaluated MLX versions.
Original MIT notices are preserved. This is not an independently developed model port.

Local patch: WeightLoading maps cguldogan fish_omni backbone keys and moves the
EOS row of the dense lm_head to the library's final semantic-head row, in memory.
Per-layer quantization uses the original checkpoint keys. Model files are read only.

The original fish_omni tower stores zero-centred normalization weights; these
are shifted to float32 gamma only for that model_type, matching the existing
adapter. Native fish_drama3 exports are not shifted again. The normalization
mapping, semantic row ordering, and per-layer quantization have regression tests.

The AVAudioConverter input callback uses a mutex-protected buffer state to
compile with current Swift concurrency checking. No resampling algorithm changed.

The app retains temperature/top-p/top-k settings, uses the upstream silence-rewind
protection, and uses a 1024-frame cap. A capped or unrecoverable silent chunk is an
error, not a successful truncated response. Embedded newlines are flattened as in
the later mlx-audio fix. The backend disables TF32 before MLX initialization so
streamed DAC output agrees with complete decoding at approximately 120 dB SNR.
