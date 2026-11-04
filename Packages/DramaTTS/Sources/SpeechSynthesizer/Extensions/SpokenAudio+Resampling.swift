//
//  SpokenAudio+Resampling.swift
//  SpeechSynthesizer
//
//  Created by David Sherlock on 2026.
//
//  Reference audio for cloning arrives at whatever rate it was recorded at;
//  each model wants its own. Shared by DramaTTS and OmniVoiceTTS.
//

import AVFoundation
import Foundation
import Synchronization

public extension SpokenAudio {

    /// The same audio at `rate` Hz (AVAudioConverter, best quality), or nil
    /// when the conversion cannot be set up.
    func resampled(to rate: Double) -> SpokenAudio? {
        guard rate != sampleRate, !samples.isEmpty else { return SpokenAudio(samples: samples, sampleRate: rate) }
        guard let input = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1,
                                        interleaved: false),
              let output = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1,
                                         interleaved: false),
              let converter = AVAudioConverter(from: input, to: output),
              let source = AVAudioPCMBuffer(pcmFormat: input, frameCapacity: AVAudioFrameCount(samples.count))
        else { return nil }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        source.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { source.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
        let capacity = AVAudioFrameCount(Double(samples.count) * rate / sampleRate) + 1024
        guard let target = AVAudioPCMBuffer(pcmFormat: output, frameCapacity: capacity) else { return nil }
        // AVAudioConverter consumes this buffer synchronously. The callback's
        // state is nevertheless Sendable under the current SDK signature.
        let inputState = Mutex(ResamplingInput(buffer: source))
        var failure: NSError?
        let status = converter.convert(to: target, error: &failure) { _, state in
            inputState.withLock { input in
                if input.supplied {
                    state.pointee = .endOfStream
                    return nil
                }
                input.supplied = true
                state.pointee = .haveData
                return input.buffer
            }
        }
        guard status != .error else { return nil }
        return SpokenAudio(samples: Array(UnsafeBufferPointer(start: target.floatChannelData![0],
                                                              count: Int(target.frameLength))),
                           sampleRate: rate)
    }
}

// The buffer is immutable after construction and accessed only during conversion.
private struct ResamplingInput: @unchecked Sendable {
    let buffer: AVAudioPCMBuffer
    var supplied = false
}
