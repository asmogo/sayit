import AVFoundation
import Foundation
import MLX
import MLXAudioCore
import MLXAudioCodecs
import MLXLMCommon
import Testing
@testable import DramaTTS
@testable import SayItBackend

@Suite("DramaTTS integration", .serialized)
struct DramaLibraryTests {
    @Test("Existing checkpoint keys and EOS rows map without changing values")
    func checkpointMapping() throws {
        #expect(DramaLibraryModel.promptText("One.\n\nTwo.") == "One.  Two.")
        let source = MLXArray([Float(10), 11, 20, 21, 30, 31], [3, 2])
        let (name, mapped) = try #require(WeightLoading.canonicalWeight("lm_head.weight", source))
        #expect(name == "semantic_head.weight")
        #expect(mapped.asArray(Float.self) == [20, 21, 30, 31, 10, 11])
        #expect(source.asArray(Float.self) == [10, 11, 20, 21, 30, 31])
        let key = "backbone.layers.0.mlp.switch_mlp.up_proj.weight"
        #expect(WeightLoading.canonicalWeight(key, source)?.0 == "backbone.model.layers.0.mlp.switch_mlp.up_proj.weight")
        #expect(WeightLoading.canonicalWeight("backbone.model.norm.weight", source)?.0 == "backbone.model.norm.weight")
        let norm = MLXArray([Float(0), 0.125]).asType(.bfloat16)
        let shifted = try #require(WeightLoading.canonicalWeight("tower.shared_layers.0.enorm.weight", norm, fishOmni: true))
        #expect(shifted.1.dtype == .float32)
        #expect(shifted.1.asArray(Float.self) == [1, 1.125])
        let unchanged = try #require(WeightLoading.canonicalWeight("tower.shared_layers.0.enorm.weight", norm))
        #expect(unchanged.1.asType(.float32).asArray(Float.self) == [0, 0.125])
        let table = QuantizationTable(standard: .init(groupSize: 64, bits: 8),
            layers: ["backbone.layers.0.mlp.switch_mlp.up_proj": .init(groupSize: 64, bits: 4)])
        let q = WeightLoading.setting(for: "backbone.model.layers.0.mlp.switch_mlp.up_proj", table: table, hasScales: { _ in true })
        #expect(q?.bits == 4)
        #expect(WeightLoading.setting(for: "tower.head", table: table, hasScales: { _ in false }) == nil)
    }

    @Test("Incremental codec preserves the complete decode waveform")
    func codecWaveform() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["SAYIT_DRAMATTS_MODEL"],
              let fixture = environment["SAYIT_DRAMATTS_FIXTURE"] else { return }
        let arrays = try loadArrays(url: URL(fileURLWithPath: fixture))
        let referenceCodes = try #require(arrays["reference_codes"])
        // Cross the codec attention window as well as multiple piece boundaries.
        let codes = concatenated(Array(repeating: referenceCodes, count: 4), axis: 1)
        let codec = try FishS1DAC.fromModelDirectory(URL(fileURLWithPath: path))
        let (whole, _) = codec.decode(codes, featureLengths: MLXArray([Int32(codes.dim(1))]))
        let expected = whole[0, 0].asType(.float32).asArray(Float.self)
        let stream = CodecStream(codec: codec)
        var actual: [Float] = []
        for start in stride(from: 0, to: codes.dim(1), by: 16) {
            let end = min(start + 16, codes.dim(1))
            actual += stream.step(codes[0..., start..<end].expandedDimensions(axis: 0))
                .asType(.float32).asArray(Float.self)
        }
        #expect(actual.count == expected.count)
        let signal = expected.reduce(0.0) { $0 + Double($1) * Double($1) }
        let noise = zip(actual, expected).reduce(0.0) { $0 + pow(Double($1.0) - Double($1.1), 2) }
        let snr = 10 * log10(signal / max(noise, 1e-30))
        #expect(snr > 90, "Streaming changed the decoded waveform")
        print("DRAMATTS_CODEC snr=\(snr) samples=\(actual.count)")
    }

    @Test("Installed Q4 streams finite speech and recovers after cancellation")
    func installedQ4() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["SAYIT_DRAMATTS_MODEL"],
              let fixture = environment["SAYIT_DRAMATTS_FIXTURE"] else { return }
        func seconds(_ start: ContinuousClock.Instant) -> Double {
            let d = start.duration(to: .now)
            return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
        }
        let reference = try #require(loadArrays(url: URL(fileURLWithPath: fixture))["reference"]).asArray(Float.self)
        let referenceText = "Hello, this is Drama Three speaking natively on this Mac."
        let started = ContinuousClock.now
        let model = try await DramaLibraryModel.fromDirectory(URL(fileURLWithPath: path))
        print("DRAMATTS_LOAD seconds=\(seconds(started))")
        var parameters = model.defaultGenerationParameters
        parameters.seed = 42
        var first: Double?
        var samples: [Float] = []
        var pieces = 0
        let speechStarted = ContinuousClock.now
        for try await event in model.generateStream(
            text: "A familiar voice can begin reading while the next sentence is prepared.",
            voice: nil, refAudio: MLXArray(reference), refText: referenceText,
            language: nil, generationParameters: parameters) {
            if case .audio(let audio) = event {
                if first == nil { first = seconds(speechStarted) }
                samples += audio.asArray(Float.self)
                pieces += 1
            }
        }
        let elapsed = seconds(speechStarted)
        #expect(pieces > 1)
        #expect(!samples.isEmpty && samples.allSatisfy(\.isFinite))
        let rms = sqrt(samples.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(samples.count))
        #expect(rms > 0.001)
        print("DRAMATTS_Q4 firstAudio=\(first ?? -1) total=\(elapsed) audio=\(Double(samples.count)/44100) rtf=\(elapsed/(Double(samples.count)/44100)) pieces=\(pieces) rms=\(rms)")
        let format = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = buffer.frameCapacity
        samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: $0.count) }
        let file = try AVAudioFile(forWriting: URL(fileURLWithPath: "/tmp/sayit-dramatts-q4.wav"), settings: format.settings)
        try file.write(from: buffer)
        let task = Task {
            for try await _ in model.generateStream(text: String(repeating: "This is a cancellation test. ", count: 20),
                voice: nil, refAudio: MLXArray(reference), refText: referenceText,
                language: nil, generationParameters: parameters) {
                try Task.checkCancellation()
            }
            try Task.checkCancellation()
        }
        try await Task.sleep(for: .seconds(1))
        let cancelStarted = ContinuousClock.now
        task.cancel()
        do { try await task.value; Issue.record("Cancelled synthesis unexpectedly succeeded") }
        catch is CancellationError { }
        let recovery = try await model.generate(text: "Ready again.", voice: nil, refAudio: MLXArray(reference),
            refText: referenceText, language: nil, generationParameters: parameters)
        #expect(recovery.size > 0)
        print("DRAMATTS_CANCEL_AND_RECOVER seconds=\(seconds(cancelStarted))")
        let anchor = Task {
            let audio = try await model.generate(text: String(repeating: "An automatic voice must cancel promptly. ", count: 20),
                voice: nil, refAudio: nil, refText: nil, language: nil, generationParameters: parameters)
            return audio.size
        }
        try await Task.sleep(for: .seconds(1))
        anchor.cancel()
        do { _ = try await anchor.value; Issue.record("Cancelled anchor unexpectedly succeeded") }
        catch is CancellationError { }

    }
}
