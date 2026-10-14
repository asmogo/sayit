// Ported from cguldogan/mlx-audio fish_omni/prompt.py (e748893).
// MIT license: see ThirdPartyNotices.md.
import Foundation
@preconcurrency import MLX

struct Drama3Prompt {
    static func validate(_ text: String) throws {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw Drama3Error.invalidInput("Text must not be empty.")
        }
        for token in ["<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:"] {
            guard !text.contains(token) else {
                throw Drama3Error.invalidInput("Text contains a reserved Drama 3 prompt token.")
            }
        }
        let markers = try NSRegularExpression(pattern: "<\\|speaker[^>]*>?")
        for match in markers.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            guard let range = Range(match.range, in: text),
                  (0...15).contains(where: { text[range] == "<|speaker:\($0)|>" }) else {
                throw Drama3Error.invalidInput("Speaker tags must be <|speaker:0|> through <|speaker:15|>.")
            }
        }
    }

    static func tagged(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("<|speaker:")
            ? text : "<|speaker:0|> " + text
    }

    static func rows(text: String, referenceText: String?, referenceCodes: MLXArray?,
                     codebooks: Int, vocabulary: Int, semanticStart: Int,
                     encode: (String) -> [Int]) throws -> MLXArray {
        try validate(text)
        var rows: [Int32] = []
        func append(_ text: String) {
            // Tokenize each structural segment separately, matching the training template.
            for id in encode(text) {
                rows.append(Int32(id))
                rows.append(contentsOf: repeatElement(0, count: codebooks))
            }
        }
        append("<|im_start|>system\n")
        append(referenceCodes == nil ? "convert the provided text to speech"
               : "convert the provided text to speech reference to the following:")
        if let codes = referenceCodes {
            guard let referenceText else { throw Drama3Error.invalidInput("A voice reference needs its transcript.") }
            try validate(referenceText)
            guard codes.ndim == 2, codes.dim(0) == codebooks, codes.dim(1) > 0,
                  codes.min().item(Int.self) >= 0, codes.max().item(Int.self) < vocabulary else {
                throw Drama3Error.invalidInput("Invalid reference audio codes.")
            }
            var transcript = tagged(referenceText)
            let body = transcript.drop(while: { $0.isWhitespace })
            let leading = transcript.prefix(transcript.count - body.count)
            if let end = body.range(of: "|>")?.upperBound {
                transcript = String(body[..<end]) + leading + body[end...]
            }
            append("\n\n[Reference 1]\nText: \(transcript)\nSpeech: ")
            let frames = codes.transposed().asType(.int32).asArray(Int32.self)
            for i in stride(from: 0, to: frames.count, by: codebooks) {
                rows.append(Int32(semanticStart) + frames[i])
                rows.append(contentsOf: frames[i..<(i + codebooks)])
            }
        }
        append("<|im_end|>\n")
        append("<|im_start|>user\n")
        append(tagged(text))
        append("<|im_end|>\n")
        append("<|im_start|>assistant\n<|voice|>")
        let count = rows.count / (codebooks + 1)
        guard count <= 8192 else { throw Drama3Error.invalidInput("Drama 3's prompt is too long. Use a shorter text or reference.") }
        return MLXArray(rows).reshaped(1, count, codebooks + 1)
    }
}

enum Drama3Error: LocalizedError {
    case invalidInput(String)
    case invalidModel(String)
    case emptyAudio
    case generationLimit

    var errorDescription: String? {
        switch self {
        case .invalidInput(let detail), .invalidModel(let detail): return detail
        case .emptyAudio: return "Drama 3 did not generate audio. Try a different text or voice reference."
        case .generationLimit: return "Drama 3 reached its generation limit. Try shorter text chunks."
        }
    }
}
