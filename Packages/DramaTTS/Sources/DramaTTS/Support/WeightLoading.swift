//
//  WeightLoading.swift
//  DramaTTS
//
//  Created by David Sherlock on 2026.
//
//  Loads the 4-bit checkpoint as it is: each layer that has `.scales` in the
//  shards is swapped for its quantised form (group size and bits from the
//  config's per-layer table, else its default), then every array is loaded
//  and checked — a missing or misshapen key is an error, not a silent zero.
//

import Foundation
import MLX
import MLXNN

/// Reads Drama 3's shards into a ``DramaNetwork``.
enum WeightLoading {

    /// Arrays the network owns; the codec's are loaded separately.
    static let keptPrefixes = ["backbone.", "semantic_head.", "tower."]

    /// The model shards in a folder, in order.
    static func shards(in folder: URL) -> [URL] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? []
        return names.filter { $0.hasPrefix("model") && $0.hasSuffix(".safetensors") }.sorted()
            .map { folder.appendingPathComponent($0) }
    }

    /// The quantisation for a layer path, or nil to leave it dense.
    static func setting(for path: String, table: QuantizationTable?,
                        hasScales: (String) -> Bool) -> QuantizationTable.Setting? {
        guard hasScales(path) else { return nil }
        guard let table else { return QuantizationTable.Setting(groupSize: 64, bits: 4) }
        let originalPath = path.hasPrefix("backbone.model.")
            ? "backbone." + path.dropFirst("backbone.model.".count) : path
        return table.layers[path] ?? table.layers[originalPath] ?? table.standard
    }

    /// Maps the earlier fish_omni conversion without modifying its stored tensors.
    static func canonicalWeight(_ key: String, _ value: MLXArray, fishOmni: Bool = false) -> (String, MLXArray)? {
        if key.hasPrefix("backbone."), !key.hasPrefix("backbone.model.") {
            return ("backbone.model." + key.dropFirst("backbone.".count), value)
        }
        if key == "lm_head.weight" {
            // fish_omni stores [EOS, semantic...]; upstream stores [semantic..., EOS].
            return ("semantic_head.weight", concatenated([value[1...], value[0..<1]], axis: 0))
        }
        let towerNorms: Set<String> = ["enorm", "hnorm", "final_layernorm", "attention_norm", "ffn_norm", "q_norm", "k_norm"]
        if fishOmni, key.hasPrefix("tower."), key.hasSuffix(".weight"),
           let layer = key.split(separator: ".").dropLast().last, towerNorms.contains(String(layer)) {
            // This conversion stores zero-centred tower norms; DramaTTS expects
            // their float32 gamma. Never apply this to an already-shifted export.
            return (key, 1 + value.asType(.float32))
        }
        return keptPrefixes.contains(where: key.hasPrefix) ? (key, value) : nil
    }

    /// Quantises and loads the network.
    static func load(_ network: DramaNetwork, from folder: URL, configuration: DramaConfiguration) throws {
        let files = shards(in: folder)
        guard !files.isEmpty else { throw DramaError.modelMissing("no model-*.safetensors in \(folder.path)") }
        var weights: [String: MLXArray] = [:]
        for file in files {
            let arrays: [String: MLXArray]
            do { arrays = try loadArrays(url: file) } catch {
                throw DramaError.modelMissing("\(file.lastPathComponent) is unreadable: \(error)")
            }
            for (key, value) in arrays {
                if let (name, tensor) = canonicalWeight(key, value, fishOmni: configuration.modelType == "fish_omni") {
                    guard weights[name] == nil else {
                        throw DramaError.unsupportedModel("duplicate weight: \(name)")
                    }
                    weights[name] = tensor
                }
            }
        }
        quantize(model: network, filter: { path, _ in
            setting(for: path, table: configuration.quantization, hasScales: { weights["\($0).scales"] != nil })
                .map { ($0.groupSize, $0.bits, .affine) }
        })
        do {
            try network.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        } catch {
            throw DramaError.unsupportedModel("the weights do not fit the network: \(error)")
        }
        eval(network.parameters())
    }
}
