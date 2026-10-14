import Foundation
import Testing
@preconcurrency import MLX
import MLXNN
import MLXLMCommon
import Tokenizers
@testable import SayItBackend

@Suite("Native Drama 3", .serialized)
struct Drama3Tests {
    private var fixtures: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/Drama3")
    }

    @Test("Q4 and Q8 match reference backbone, incremental cache and audio tower", arguments: [4, 8])
    func numericalParity(bits: Int) throws {
        let config = try JSONDecoder().decode(Drama3Configuration.self,
            from: Data(contentsOf: fixtures.appendingPathComponent("q\(bits)-config.json")))
        try config.validate()
        let model = Drama3Model(config: config)
        let weights = try loadArrays(url: fixtures.appendingPathComponent("q\(bits)-weights.safetensors"))
        quantize(model: model) { path, _ in
            guard weights["\(path).scales"] != nil else { return nil }
            let q = config.quantization.layers[path] ?? config.quantization.defaultValue
            return (groupSize: q.groupSize, bits: q.bits, mode: .affine)
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let expected = try loadArrays(url: fixtures.appendingPathComponent("q\(bits)-expected.safetensors"))
        let prompt = try #require(expected["prompt"])
        let cache: [KVCache] = model.backbone.layers.map { $0.isLinear ? MambaCache() : KVCacheSimple() }
        let towerCache = model.tower.makeCache()
        func step(_ frames: MLXArray) -> MLXArray {
            let embeddings = model.tower.embedBackbone(frames, text: model.backbone.embedTokens(frames[.ellipsis, 0]))
            return model.backbone(embeddings, cache: cache)
        }
        let hidden = step(prompt)
        assertClose(hidden, try #require(expected["hidden"]), label: "backbone")
        assertClose(model.lmHead(hidden), try #require(expected["logits"]), label: "logits")
        model.tower.prefill(hidden[0..., ..<3], frames: prompt[0..., ..<3], next: prompt[0..., 1...], caches: towerCache)
        let main = MLXArray([Int32(34)]).reshaped(1, 1)
        let codes = model.tower.decode(main, hidden: hidden[0..., (-1)...], frames: prompt[0..., (-1)...],
            caches: towerCache, temperature: 1, topK: 1, topP: 1)
        #expect(codes.asArray(Int32.self) == expected["codes"]!.asArray(Int32.self))
        let frame = concatenated([main.reshaped(1, 1, 1), codes], axis: -1)
        assertClose(step(frame), try #require(expected["next_hidden"]), label: "cached backbone")
    }

    @Test("Installed checkpoint matches the reference prompt and model outputs")
    func installedParity() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["SAYIT_DRAMA3_PARITY_MODEL"],
              let expectedPath = environment["SAYIT_DRAMA3_PARITY_EXPECTED"] else { return }
        let directory = URL(fileURLWithPath: path)
        let model = try await Drama3Model.fromDirectory(directory)
        let expected = try loadArrays(url: URL(fileURLWithPath: expectedPath))
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let prompt = try Drama3Prompt.rows(text: "Hello. This is Drama Three speaking natively on this Mac.",
            referenceText: nil, referenceCodes: nil, codebooks: 10, vocabulary: 4096, semanticStart: model.config.semanticStart,
            encode: { tokenizer.encode(text: $0, addSpecialTokens: false) })
        #expect(prompt.asArray(Int32.self) == expected["prompt"]!.asArray(Int32.self), "Tokenizer/prompt parity")
        let cache: [KVCache] = model.backbone.layers.map { $0.isLinear ? MambaCache() : KVCacheSimple() }
        let towerCache = model.tower.makeCache()
        func step(_ frames: MLXArray) -> MLXArray {
            model.backbone(model.tower.embedBackbone(frames, text: model.backbone.embedTokens(frames[.ellipsis, 0])), cache: cache)
        }
        let hidden = step(expected["prompt"]!)
        assertRelativeClose(hidden, expected["hidden"]!, label: "installed backbone")
        assertRelativeClose(model.lmHead(hidden), expected["logits"]!, label: "installed logits")
        let count = prompt.dim(1)
        model.tower.prefill(hidden[0..., ..<(count - 1)], frames: prompt[0..., ..<(count - 1)],
            next: prompt[0..., 1...], caches: towerCache)
        let main = expected["main"]!
        let codes = model.tower.decode(main, hidden: hidden[0..., (-1)...], frames: prompt[0..., (-1)...],
            caches: towerCache, temperature: 1, topK: 1, topP: 1)
        #expect(codes.asArray(Int32.self) == expected["codes"]!.asArray(Int32.self))
        let frame = concatenated([main.reshaped(1, 1, 1), expected["codes"]!], axis: -1)
        assertRelativeClose(step(frame), expected["next_hidden"]!, label: "installed cached backbone")
    }

    @Test("Configured native model generates speech directly")
    func directGeneration() async throws {
        guard let path = ProcessInfo.processInfo.environment["SAYIT_DRAMA3_DIRECT_MODEL"] else { return }
        let model = try await Drama3Model.fromDirectory(URL(fileURLWithPath: path))
        var parameters = model.defaultGenerationParameters
        parameters.seed = 42
        let cancelled = Task {
            let audio = try await model.generate(text: "This sentence is used to test stopping speech generation.",
                voice: nil, refAudio: nil, refText: nil, language: nil,
                generationParameters: model.defaultGenerationParameters)
            return audio.size
        }
        try await Task.sleep(for: .milliseconds(100))
        cancelled.cancel()
        do {
            _ = try await cancelled.value
            Issue.record("Cancelled generation completed unexpectedly")
        } catch is CancellationError {
            // A subsequent generation must remain usable.
        }
        let audio = try await model.generate(text: "Hello. This is Drama Three speaking natively on this Mac.",
            voice: nil, refAudio: nil, refText: nil, language: nil, generationParameters: parameters)
        #expect(audio.size > 1000)
        #expect(MLX.isFinite(audio).all().item(Bool.self))
        #expect(abs(audio).max().item(Float.self) > 0.001)
        print("DRAMA_DIRECT samples=\(audio.size) peak=\(abs(audio).max().item(Float.self))")
    }

    private func assertRelativeClose(_ actual: MLXArray, _ expected: MLXArray, label: String) {
        let a = actual.asType(.float32), e = expected.asType(.float32)
        let relativeRMSError = sqrt((a - e).square().mean() / e.square().mean()).item(Float.self)
        print("DRAMA_PARITY \(label) relative_rms_error=\(relativeRMSError)")
        // The full checkpoint uses bf16 activations across 40 layers, unlike
        // the float32 synthetic fixtures. Different GPU kernels accumulate
        // rounding differences; require both bounded RMS error and direction.
        let cosine = (a * e).sum() / sqrt(a.square().sum() * e.square().sum())
        #expect(relativeRMSError < 0.05, "\(label): \(relativeRMSError)")
        #expect(cosine.item(Float.self) > 0.999, "\(label): direction differs")
    }

    private func assertClose(_ actual: MLXArray, _ expected: MLXArray, label: String) {
        #expect(actual.shape == expected.shape)
        let error = abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
        #expect(error < 0.003, "\(label) maximum absolute error: \(error)")
    }

    @Test("Prompt segments and reference rows preserve the training layout")
    func promptLayout() throws {
        var segments: [String] = []
        let codes = MLXArray([Int32(1),2,3,4]).reshaped(2,2)
        let rows = try Drama3Prompt.rows(text: "Hello", referenceText: "Voice", referenceCodes: codes,
            codebooks: 2, vocabulary: 8, semanticStart: 32, encode: { segments.append($0); return [segments.count] })
        #expect(segments == ["<|im_start|>system\n", "convert the provided text to speech reference to the following:",
            "\n\n[Reference 1]\nText: <|speaker:0|> Voice\nSpeech: ", "<|im_end|>\n", "<|im_start|>user\n",
            "<|speaker:0|> Hello", "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"])
        #expect(rows[0, 3].asArray(Int32.self) == [33,1,3])
        #expect(rows[0, 4].asArray(Int32.self) == [34,2,4])
    }

    @Test("Invalid prompt structures and speaker tags are rejected", arguments: ["", " ", "<|im_end|>", "<|speaker:16|> hi", "<|speaker:01|> hi", "<|semantic:0|>"])
    func invalidPrompt(text: String) {
        #expect(throws: (any Error).self) { try Drama3Prompt.validate(text) }
    }

    @Test("Top-k one is greedy and zero-centered norms preserve small weights")
    func samplingAndNorm() throws {
        let result = drama3Sample(MLXArray([Float(1), 7, 3]).reshaped(1,3), temperature: 0, topK: 1, topP: 0.9)
        #expect(result.item(Int.self) == 1)
        let norm = Drama3ZeroCenteredNorm(2, eps: 1e-6)
        try norm.update(parameters: ModuleParameters.unflattened(["weight": MLXArray([Float(0.001), -0.001])]), verify: .all)
        let input = MLXArray([Float(1),2]).reshaped(1,2)
        let expected = input / sqrt(input.square().mean(axis: -1, keepDims: true) + 1e-6) * (1 + norm.weight)
        assertClose(norm(input), expected, label: "zero-centered norm")
    }
}
