// Ported from cguldogan/mlx-audio fish_omni/tower.py (e748893).
// MIT license: see ThirdPartyNotices.md.
import Foundation
@preconcurrency import MLX
import MLXFast
import MLXLMCommon
import MLXNN

struct Drama3DecoderConfiguration: Decodable {
    let dim: Int
    let vocabSize: Int
    let numCodebooks: Int
    let nHead: Int
    let nLocalHeads: Int
    let headDim: Int
    let intermediateSize: Int
    let normEps: Float
    let ropeBase: Float
    let rotaryPercent: Float
    let sharedEnrichmentLayers: Int
    let vqBackboneCodebooks: Int

    enum CodingKeys: String, CodingKey {
        case dim
        case vocabSize = "vocab_size", numCodebooks = "num_codebooks"
        case nHead = "n_head", nLocalHeads = "n_local_heads", headDim = "head_dim"
        case intermediateSize = "intermediate_size", normEps = "norm_eps"
        case ropeBase = "rope_base", rotaryPercent = "rotary_percent"
        case sharedEnrichmentLayers = "shared_enrichment_layers"
        case vqBackboneCodebooks = "vq_backbone_codebooks"
    }
}

final class Drama3ZeroCenteredNorm: Module, UnaryLayer {
    @ParameterInfo var weight: MLXArray
    let eps: Float

    init(_ dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.zeros([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normalized = MLXFast.rmsNorm(x.asType(.float32), weight: .mlxNone, eps: eps)
        return (normalized * (1 + weight.asType(.float32))).asType(x.dtype)
    }
}

/// The reference drops the token crossing top-p, but always retains rank zero.
func drama3Sample(_ logits: MLXArray, temperature: Float, topK: Int, topP: Float) -> MLXArray {
    let scaled = logits.asType(.float32) / max(temperature, 1e-5)
    let k = min(max(topK, 1), scaled.dim(-1))
    var ids = argPartition(-scaled, kth: k - 1, axis: -1)[.ellipsis, ..<k]
    var values = takeAlong(scaled, ids, axis: -1)
    let order = argSort(-values, axis: -1)
    ids = takeAlong(ids, order, axis: -1)
    values = takeAlong(values, order, axis: -1)
    let cumulative = cumsum(exp(values - logSumExp(scaled, axis: -1, keepDims: true)), axis: -1)
    let drop = (cumulative .> topP) .&& (MLXArray(0..<k) .> 0)
    let masked = MLX.where(drop, -Float.infinity, values)
    let choice = MLXRandom.categorical(masked, axis: -1)
    return takeAlong(ids, choice.expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
}

final class Drama3TowerAttention: Module {
    let heads: Int, localHeads: Int, headDim: Int, qWidth: Int, kvWidth: Int
    let scale: Float
    let rope: RoPE
    let wqkv: Linear
    let wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Drama3ZeroCenteredNorm
    @ModuleInfo(key: "k_norm") var kNorm: Drama3ZeroCenteredNorm

    init(_ c: Drama3DecoderConfiguration) {
        heads = c.nHead; localHeads = c.nLocalHeads; headDim = c.headDim
        qWidth = c.nHead * c.headDim; kvWidth = c.nLocalHeads * c.headDim
        scale = pow(Float(c.headDim), -0.5)
        wqkv = Linear(c.dim, 2 * qWidth + 2 * kvWidth, bias: false)
        wo = Linear(qWidth, c.dim, bias: false)
        _qNorm.wrappedValue = Drama3ZeroCenteredNorm(c.headDim, eps: c.normEps)
        _kNorm.wrappedValue = Drama3ZeroCenteredNorm(c.headDim, eps: c.normEps)
        rope = RoPE(dimensions: Int(Float(c.headDim) * c.rotaryPercent), traditional: true, base: c.ropeBase)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
                        cache: KVCache) -> MLXArray {
        let b = x.dim(0), l = x.dim(1)
        let parts = split(wqkv(x), indices: [qWidth, 2 * qWidth, 2 * qWidth + kvWidth], axis: -1)
        let q = rope(qNorm(parts[0].reshaped(b, l, heads, headDim)).transposed(0, 2, 1, 3), offset: cache.offset)
        let k = rope(kNorm(parts[2].reshaped(b, l, localHeads, headDim)).transposed(0, 2, 1, 3), offset: cache.offset)
        let v = parts[3].reshaped(b, l, localHeads, headDim).transposed(0, 2, 1, 3)
        let output = attentionWithCacheUpdate(queries: q, keys: k, values: v, cache: cache,
                                             scale: scale, mask: mask)
            .transposed(0, 2, 1, 3).reshaped(b, l, qWidth)
        return wo(output * sigmoid(parts[1]))
    }
}

final class Drama3TowerMLP: Module, UnaryLayer {
    let w1: Linear, w2: Linear, w3: Linear
    init(_ c: Drama3DecoderConfiguration) {
        w1 = Linear(c.dim, c.intermediateSize, bias: false)
        w3 = Linear(c.dim, c.intermediateSize, bias: false)
        w2 = Linear(c.intermediateSize, c.dim, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x)) * w3(x)) }
}

final class Drama3TowerBlock: Module {
    let attention: Drama3TowerAttention
    @ModuleInfo(key: "attention_norm") var attentionNorm: Drama3ZeroCenteredNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: Drama3ZeroCenteredNorm
    @ModuleInfo(key: "feed_forward") var feedForward: Drama3TowerMLP
    init(_ c: Drama3DecoderConfiguration) {
        attention = Drama3TowerAttention(c)
        _attentionNorm.wrappedValue = Drama3ZeroCenteredNorm(c.dim, eps: c.normEps)
        _ffnNorm.wrappedValue = Drama3ZeroCenteredNorm(c.dim, eps: c.normEps)
        _feedForward.wrappedValue = Drama3TowerMLP(c)
    }
    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
                        cache: KVCache) -> MLXArray {
        let h = x + attention(attentionNorm(x), mask: mask, cache: cache)
        return h + feedForward(ffnNorm(h))
    }
}

final class Drama3TowerLayer: Module {
    let vocabulary: Int
    @ModuleInfo(key: "semantic_embedding") var semanticEmbedding: Embedding?
    @ModuleInfo(key: "codebook_embedding") var codebookEmbedding: Embedding?
    let enorm: Drama3ZeroCenteredNorm, hnorm: Drama3ZeroCenteredNorm
    @ModuleInfo(key: "eh_proj") var ehProj: Linear
    @ModuleInfo(key: "transformer_layer") var transformer: Drama3TowerBlock
    @ModuleInfo(key: "final_layernorm") var finalNorm: Drama3ZeroCenteredNorm

    init(_ c: Drama3DecoderConfiguration, shared: Bool) {
        vocabulary = c.vocabSize
        let embedding = Embedding(embeddingCount: c.vocabSize, dimensions: c.dim)
        if shared { _semanticEmbedding.wrappedValue = embedding }
        else { _codebookEmbedding.wrappedValue = embedding }
        enorm = Drama3ZeroCenteredNorm(c.dim, eps: c.normEps)
        hnorm = Drama3ZeroCenteredNorm(c.dim, eps: c.normEps)
        _ehProj.wrappedValue = Linear(2 * c.dim, c.dim, bias: false)
        _transformer.wrappedValue = Drama3TowerBlock(c)
        _finalNorm.wrappedValue = Drama3ZeroCenteredNorm(c.dim, eps: c.normEps)
    }

    func callAsFunction(_ hidden: MLXArray, codes: MLXArray, semantic: MLXArray,
                        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache,
                        sharedInput: MLXArray? = nil) -> MLXArray {
        let embedding = semanticEmbedding ?? codebookEmbedding!
        var e = embedding(clip(codes, min: 0, max: vocabulary - 1))
        e = MLX.where(semantic.expandedDimensions(axis: -1), e, 0)
        if let sharedInput { e = e + sharedInput }
        let mixed = ehProj(concatenated([enorm(e), hnorm(hidden)], axis: -1))
        return finalNorm(transformer(mixed, mask: mask, cache: cache))
    }
}

final class Drama3Tower: Module {
    let config: Drama3DecoderConfiguration
    let semanticBegin: Int, semanticEnd: Int
    @ModuleInfo(key: "codebook_embeddings") var embeddings: Embedding
    @ModuleInfo(key: "shared_layers") var sharedLayers: [Drama3TowerLayer]
    @ModuleInfo(key: "residual_layers") var residualLayers: [Drama3TowerLayer]
    @ModuleInfo(key: "codebook_heads") var heads: [Linear]

    init(_ c: Drama3DecoderConfiguration, semanticBegin: Int, semanticEnd: Int) {
        config = c; self.semanticBegin = semanticBegin; self.semanticEnd = semanticEnd
        _embeddings.wrappedValue = Embedding(embeddingCount: c.vocabSize * c.numCodebooks, dimensions: c.dim)
        _sharedLayers.wrappedValue = (0..<c.sharedEnrichmentLayers).map { _ in Drama3TowerLayer(c, shared: true) }
        _residualLayers.wrappedValue = (1..<c.numCodebooks).map { _ in Drama3TowerLayer(c, shared: false) }
        _heads.wrappedValue = (1..<c.numCodebooks).map { _ in Linear(c.dim, c.vocabSize, bias: false) }
    }

    func makeCache() -> [KVCache] {
        (0..<(sharedLayers.count + residualLayers.count)).map { _ in KVCacheSimple() }
    }
    func isSemantic(_ tokens: MLXArray) -> MLXArray {
        (tokens .>= semanticBegin) .&& (tokens .<= semanticEnd)
    }
    func embeddingSum(_ frames: MLXArray, first: Int, end: Int) -> MLXArray {
        let offsets = MLXArray(first..<end) * config.vocabSize
        return embeddings(frames[.ellipsis, (1 + first)..<(1 + end)] + offsets).sum(axis: -2)
    }
    func embedBackbone(_ frames: MLXArray, text: MLXArray) -> MLXArray {
        let count = config.vqBackboneCodebooks
        let extra = embeddingSum(frames, first: 1, end: 1 + count).asType(text.dtype)
        let mixed = (text + extra) / sqrt(Float(1 + count))
        return MLX.where(isSemantic(frames[.ellipsis, 0]).expandedDimensions(axis: -1), mixed, text)
    }
    func runShared(_ hidden: MLXArray, c0: MLXArray, semantic: MLXArray, frames: MLXArray,
                   mask: MLXFast.ScaledDotProductAttentionMaskMode, caches: [KVCache]) -> MLXArray {
        var h = hidden
        var unseen = embeddingSum(frames, first: 1 + config.vqBackboneCodebooks, end: config.numCodebooks)
        unseen = MLX.where(isSemantic(frames[.ellipsis, 0]).expandedDimensions(axis: -1), unseen, 0)
        for (i, layer) in sharedLayers.enumerated() {
            h = layer(h, codes: c0, semantic: semantic, mask: mask, cache: caches[i],
                      sharedInput: i == 0 ? unseen : nil)
        }
        return h
    }
    func prefill(_ hidden: MLXArray, frames: MLXArray, next: MLXArray, caches: [KVCache]) {
        let mask = createAttentionMask(h: hidden, cache: caches[0])
        let semantic = isSemantic(next[.ellipsis, 0])
        var h = runShared(hidden, c0: next[.ellipsis, 1], semantic: semantic,
                          frames: frames, mask: mask, caches: caches)
        for (i, layer) in residualLayers.enumerated() {
            h = layer(h, codes: next[.ellipsis, i + 1], semantic: semantic, mask: mask,
                      cache: caches[sharedLayers.count + i])
        }
    }
    func decode(_ main: MLXArray, hidden: MLXArray, frames: MLXArray, caches: [KVCache],
                temperature: Float, topK: Int, topP: Float) -> MLXArray {
        let c0 = clip(main - semanticBegin, min: 0, max: config.vocabSize - 1)
        let semantic = isSemantic(main)
        var h = runShared(hidden, c0: c0, semantic: semantic, frames: frames, mask: .none, caches: caches)
        var codes = [c0]
        for (i, layer) in residualLayers.enumerated() {
            h = layer(h, codes: codes.last!, semantic: semantic, mask: .none,
                      cache: caches[sharedLayers.count + i])
            codes.append(drama3Sample(heads[i](h)[0..., -1], temperature: temperature,
                                      topK: topK, topP: topP).expandedDimensions(axis: -1))
        }
        return stacked(codes, axis: -1)
    }
}
