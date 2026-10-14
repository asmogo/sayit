import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioCodecs
@preconcurrency import MLXAudioTTS
@preconcurrency import MLXLMCommon
import MLXNN
import Tokenizers

/// Native Drama 3 inference. SynthesisActor serializes access to all MLX models.
final class Drama3Model: Module, SpeechGenerationModel, @unchecked Sendable {
    let sampleRate = 44_100
    let config: Drama3Configuration
    let backbone: Drama3BackboneTextModelInner
    let tower: Drama3Tower
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    // Kept out of Module's parameter tree: the codec has its own checkpoint.
    private var runtime: Runtime?
    private struct Runtime {
        let tokenizer: any Tokenizers.Tokenizer
        let codec: FishS1DAC
    }

    var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(maxTokens: 600, temperature: 1, topP: 0.9, topK: 30)
    }

    init(config: Drama3Configuration) {
        self.config = config
        backbone = Drama3BackboneTextModelInner(config.text)
        tower = Drama3Tower(config.decoder, semanticBegin: config.semanticStart, semanticEnd: config.semanticEnd)
        _lmHead.wrappedValue = Linear(config.text.hiddenSize, config.semanticEnd - config.semanticStart + 2, bias: false)
    }

    static func fromDirectory(_ directory: URL) async throws -> Drama3Model {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Drama3Configuration.self, from: data)
        try config.validate()
        try Task.checkCancellation()
        let model = Drama3Model(config: config)
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" && $0.lastPathComponent != "codec.safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw Drama3Error.invalidModel("Drama 3 weights are missing.") }
        var weights: [String: MLXArray] = [:]
        for file in files {
            try Task.checkCancellation()
            for (name, value) in try loadArrays(url: file) {
                guard weights[name] == nil else { throw Drama3Error.invalidModel("Duplicate Drama 3 weight: \(name)") }
                weights[name] = value
            }
        }
        quantize(model: model) { path, _ in
            guard weights["\(path).scales"] != nil else { return nil }
            let q = config.quantization.layers[path] ?? config.quantization.defaultValue
            return (groupSize: q.groupSize, bits: q.bits, mode: .affine)
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model)
        try Task.checkCancellation()
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        guard tokenizer.convertTokenToId("<|semantic:0|>") == config.semanticStart,
              tokenizer.convertTokenToId("<|im_end|>") == config.eos,
              tokenizer.convertTokenToId("<|semantic:\(config.decoder.vocabSize - 1)|>") == config.semanticEnd else {
            throw Drama3Error.invalidModel("Drama 3 tokenizer does not match the model configuration.")
        }
        let codec = try FishS1DAC.fromModelDirectory(directory)
        model.runtime = Runtime(tokenizer: tokenizer, codec: codec)
        try Task.checkCancellation()
        return model
    }

    func generate(text: String, voice: String?, refAudio: MLXArray?, refText: String?,
                  language: String?, generationParameters: GenerateParameters) async throws -> MLXArray {
        guard let runtime else { throw Drama3Error.invalidModel("Drama 3 is not loaded.") }
        try Task.checkCancellation()
        let p = generationParameters
        guard p.temperature.isFinite, p.temperature >= 0, p.temperature <= 2,
              p.topP.isFinite, p.topP > 0, p.topP <= 1, p.topK > 0 else {
            throw Drama3Error.invalidInput("Invalid Drama 3 sampling settings.")
        }
        if let seed = p.seed { MLXRandom.seed(seed) }
        var referenceCodes: MLXArray?
        if let audio = refAudio {
            guard refText != nil, audio.size > 0, audio.size <= sampleRate * 30,
                  MLX.isFinite(audio).all().item(Bool.self) else {
                throw Drama3Error.invalidInput("Use a finite voice reference of at most 30 seconds with its transcript.")
            }
            let (codes, lengths) = runtime.codec.encode(audio.reshaped(1, 1, -1))
            referenceCodes = codes[0, 0..., ..<lengths[0].item(Int.self)]
            eval(referenceCodes!)
        }
        let prompt = try Drama3Prompt.rows(text: text, referenceText: refText,
            referenceCodes: referenceCodes, codebooks: config.decoder.numCodebooks,
            vocabulary: config.decoder.vocabSize, semanticStart: config.semanticStart,
            encode: { runtime.tokenizer.encode(text: $0, addSpecialTokens: false) })
        let cache: [KVCache] = backbone.layers.map { $0.isLinear ? MambaCache() : KVCacheSimple() }
        let towerCache = tower.makeCache()
        func step(_ frames: MLXArray) -> MLXArray {
            let embeddings = tower.embedBackbone(frames, text: backbone.embedTokens(frames[.ellipsis, 0]))
            return backbone(embeddings, cache: cache)
        }
        var hidden: MLXArray = MLXArray.zeros([0])
        for start in stride(from: 0, to: prompt.dim(1), by: 256) {
            try Task.checkCancellation()
            let end = min(start + 256, prompt.dim(1))
            hidden = step(prompt[0..., start..<end])
            let pairedEnd = min(end, prompt.dim(1) - 1)
            if pairedEnd > start {
                tower.prefill(hidden[0..., ..<(pairedEnd - start)], frames: prompt[0..., start..<pairedEnd],
                              next: prompt[0..., (start + 1)..<(pairedEnd + 1)], caches: towerCache)
            }
            eval(hidden)
            eval(cache.flatMap(\.state) + towerCache.flatMap(\.state))
        }
        hidden = hidden[0..., (-1)...]
        var frame = prompt[0..., (-1)...]
        var window = prompt[0, max(0, prompt.dim(1) - 16)..., 0]
        var generated: [MLXArray] = []
        var finished = false
        for _ in 0..<max(1, min(p.maxTokens ?? 600, 1500)) {
            try Task.checkCancellation()
            let logits = lmHead(hidden)[0..., -1]
            func token(_ index: MLXArray) -> MLXArray {
                MLX.where(index .== 0, config.eos, config.semanticStart + index - 1)
            }
            let first = token(drama3Sample(logits, temperature: p.temperature, topK: p.topK, topP: p.topP))
            let retry = token(drama3Sample(logits, temperature: 1, topK: 30, topP: 0.95))
            let repeated = (window.expandedDimensions(axis: 0) .== first.expandedDimensions(axis: -1)).any(axis: -1)
                .&& tower.isSemantic(first)
            let main = MLX.where(repeated, retry, first)
            if main.item(Int.self) == config.eos { finished = true; break }
            let codes = tower.decode(main.expandedDimensions(axis: -1), hidden: hidden, frames: frame,
                                     caches: towerCache, temperature: p.temperature, topK: p.topK, topP: p.topP)
            frame = concatenated([main.reshaped(1, 1, 1), codes], axis: -1)
            eval(frame)
            generated.append(codes)
            window = concatenated([window[1...], main])
            hidden = step(frame)
        }
        try Task.checkCancellation()
        guard finished else { throw Drama3Error.generationLimit }
        guard !generated.isEmpty else { throw Drama3Error.emptyAudio }
        let codes = concatenated(generated, axis: 1)[0].transposed()
        let (audio, lengths) = runtime.codec.decode(codes, featureLengths: MLXArray([Int32(codes.dim(1))]))
        let result = audio[0, 0, ..<lengths[0].item(Int.self)].asType(.float32)
        eval(result)
        try Task.checkCancellation()
        guard result.size > 0, MLX.isFinite(result).all().item(Bool.self) else { throw Drama3Error.emptyAudio }
        return result
    }

    func generateStream(text: String, voice: String?, refAudio: MLXArray?, refText: String?,
                        language: String?, generationParameters: GenerateParameters) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(text: text, voice: voice, refAudio: refAudio, refText: refText,
                       language: language, generationParameters: generationParameters, streamingInterval: 2)
    }

    func generateStream(text: String, voice: String?, refAudio: MLXArray?, refText: String?,
                        language: String?, generationParameters: GenerateParameters,
                        streamingInterval: Double) -> AsyncThrowingStream<AudioGeneration, Error> {
        let referenceSamples = refAudio?.asType(.float32).asArray(Float.self)
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let audio = try await generate(text: text, voice: voice, refAudio: referenceSamples.map { MLXArray($0) },
                        refText: refText, language: language, generationParameters: generationParameters)
                    continuation.yield(.audio(audio))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}
