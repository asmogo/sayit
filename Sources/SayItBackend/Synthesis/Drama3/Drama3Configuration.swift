import Foundation

struct Drama3Configuration: Decodable {
    let modelType: String
    let text: Drama3BackboneTextConfiguration
    let decoder: Drama3DecoderConfiguration
    let semanticStart: Int, semanticEnd: Int, eos: Int
    let quantization: Drama3Quantization
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", text = "text_config", decoder = "audio_decoder_config"
        case semanticStart = "semantic_start_token_id", semanticEnd = "semantic_end_token_id"
        case eos = "eos_token_id", quantization
    }
    func validate() throws {
        guard modelType == "fish_omni", eos >= 0, eos < semanticStart,
              semanticEnd - semanticStart + 1 == decoder.vocabSize,
              semanticEnd < text.vocabularySize,
              decoder.numCodebooks > 1, decoder.numCodebooks <= 32,
              decoder.dim == text.hiddenSize,
              decoder.vqBackboneCodebooks > 0,
              decoder.vqBackboneCodebooks < decoder.numCodebooks - 1,
              decoder.sharedEnrichmentLayers > 0,
              decoder.nHead > 0, decoder.nLocalHeads > 0,
              decoder.nHead % decoder.nLocalHeads == 0,
              decoder.headDim > 0, decoder.rotaryPercent > 0, decoder.rotaryPercent <= 1,
              text.hiddenLayers >= text.fullAttentionInterval, text.fullAttentionInterval > 1,
              text.linearNumKeyHeads > 0,
              text.linearNumValueHeads % text.linearNumKeyHeads == 0,
              text.numExperts > 0, text.numExpertsPerTok > 0,
              text.numExpertsPerTok <= text.numExperts else {
            throw Drama3Error.invalidModel("Unsupported Drama 3 model configuration. Use the MLX Q4 or Q8 conversion.")
        }
        try quantization.validate()
    }
}

struct Drama3Quantization: Decodable {
    struct Value: Decodable {
        let groupSize: Int
        let bits: Int
        let mode: String?
        enum CodingKeys: String, CodingKey { case groupSize = "group_size", bits, mode }
    }
    let defaultValue: Value
    let layers: [String: Value]
    private struct Key: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
    init(from decoder: Decoder) throws {
        defaultValue = try Value(from: decoder)
        let container = try decoder.container(keyedBy: Key.self)
        var values: [String: Value] = [:]
        for key in container.allKeys where !["group_size", "bits", "mode"].contains(key.stringValue) {
            values[key.stringValue] = try container.decode(Value.self, forKey: key)
        }
        layers = values
    }
    func validate() throws {
        for q in [defaultValue] + Array(layers.values) {
            guard [4, 8].contains(q.bits), [32, 64, 128].contains(q.groupSize),
                  q.mode == nil || q.mode == "affine" else {
                throw Drama3Error.invalidModel("Unsupported Drama 3 quantization. Use affine Q4 or Q8 weights.")
            }
        }
    }
}
