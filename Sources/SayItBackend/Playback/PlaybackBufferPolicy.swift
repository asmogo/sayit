import Foundation
import SayItCore

struct PlaybackBufferPolicy: Sendable {
    static let progressiveBaseLead: TimeInterval = 1.2
    static let bufferedBaseLead: TimeInterval = 2.5
    static let minimumSustainableStartBuffer: TimeInterval = 0.15
    static let generationSafetyMargin: TimeInterval = 0.4
    static let sustainableLoadLimit = 0.92
    static let wholeChunkVariabilityReserve = 1.5
    static let unstartedChunkVariabilityReserve = 2.0

    let mode: PlaybackMode
    let rate: Double
    let estimator: SynthesisPerformanceEstimator
    var receivesWholeChunks = false

    var preferredSourceLead: TimeInterval {
        guard mode != .completeFirst else { return .infinity }
        let baseLead: TimeInterval
        switch mode {
        case .progressive:
            baseLead = Self.progressiveBaseLead
        case .buffered, .adaptive:
            baseLead = Self.bufferedBaseLead
        case .completeFirst:
            baseLead = 0
        }
        let observedLead = estimator.hasObservations
            ? estimator.conservativeGenerationDuration
                + Self.generationSafetyMargin
            : 0
        return max(baseLead, observedLead) * max(rate, 1)
    }

    var streamingIsSustainable: Bool {
        guard mode != .completeFirst else { return false }
        if mode == .adaptive, !estimator.hasObservations {
            return false
        }
        guard estimator.hasObservations else { return true }
        return estimator.conservativeRealTimeFactor * max(rate, 0)
            < Self.sustainableLoadLimit
    }

    /// Qualify startup from delivered pieces before the first logical chunk ends.
    /// Frame-only progress must never qualify: those samples are not playable yet.
    func shouldStartStreaming(_ progress: SynthesisProgress,
                              bufferedDuration: TimeInterval) -> Bool {
        guard mode == .adaptive, !receivesWholeChunks,
              progress.deliveredAudioPieces >= 4,
              progress.generatedAudioDuration.isFinite, progress.generatedAudioDuration >= 3,
              progress.generationDuration.isFinite, progress.generationDuration > 0,
              progress.recentRealTimeFactor.isFinite, progress.recentRealTimeFactor > 0,
              progress.maximumDeliveryGap.isFinite, progress.maximumDeliveryGap > 0,
              progress.initialGenerationDuration.isFinite, progress.initialGenerationDuration >= 0,
              progress.initialGenerationDuration < progress.generationDuration,
              progress.initialAudioDuration.isFinite, progress.initialAudioDuration >= 0,
              progress.initialAudioDuration < progress.generatedAudioDuration,
              bufferedDuration.isFinite, rate.isFinite, rate > 0 else { return false }
        let cumulativeFactor = (progress.generationDuration - progress.initialGenerationDuration)
            / (progress.generatedAudioDuration - progress.initialAudioDuration)
        let ongoingFactor = max(progress.recentRealTimeFactor, estimator.conservativeRealTimeFactor)
        // Prefill has already been paid for and its samples are buffered. Require
        // cumulative sustainability and 25% headroom for the recent delivery rate.
        // Two observed delivery gaps reserve time for a later chunk's prefill.
        guard cumulativeFactor * rate < Self.sustainableLoadLimit,
              ongoingFactor * rate * 1.25 < Self.sustainableLoadLimit else { return false }
        let reserve = max(Self.bufferedBaseLead,
                          progress.maximumDeliveryGap * 2 + Self.generationSafetyMargin)
        return bufferedDuration >= reserve * max(rate, 1)
    }

    /// Predict each remaining whole-chunk arrival using current frame progress.
    /// Reassessing completed work avoids applying an old slow chunk's full cost
    /// again while the next chunk is already being generated.
    func requiredProgressLead(_ progress: SynthesisProgress,
                              completedAudioDuration: TimeInterval) -> TimeInterval? {
        guard mode == .adaptive, receivesWholeChunks,
              progress.chunkIndex >= 2, progress.completedTextCharacters > 0,
              !progress.pendingTextCharacters.isEmpty,
              progress.pendingTextCharacters.allSatisfy({ $0 > 0 }),
              completedAudioDuration.isFinite, completedAudioDuration > 0,
              progress.generatedAudioDuration.isFinite, progress.generatedAudioDuration >= 2,
              progress.generationDuration.isFinite, progress.generationDuration > 0,
              progress.recentRealTimeFactor.isFinite, progress.recentRealTimeFactor > 0,
              progress.maximumAudioDuration.isFinite,
              progress.maximumAudioDuration >= progress.generatedAudioDuration else { return nil }
        let pace = completedAudioDuration / Double(progress.completedTextCharacters)
        let factor = max(estimator.conservativeRealTimeFactor, progress.recentRealTimeFactor,
            progress.generationDuration / progress.generatedAudioDuration)
            * Self.wholeChunkVariabilityReserve
        // Allow a sentence to outlast the text-based estimate. Never treat
        // reaching that estimate as EOS; only the model can finish a chunk.
        let currentAudio = min(progress.maximumAudioDuration,
            max(Double(progress.pendingTextCharacters[0]) * pace,
                progress.generatedAudioDuration * 1.25))
        let audio = [currentAudio] + progress.pendingTextCharacters.dropFirst().map { Double($0) * pace }
        var arrival = 0.0
        var delivered = 0.0
        var required = 0.0
        for (index, duration) in audio.enumerated() {
            let remaining = index == 0 ? max(duration - progress.generatedAudioDuration, 0) : duration
            // A short future sentence can still take as long as earlier chunks.
            // Only subtract measured work from the chunk actually in progress;
            // retain a wall-clock budget for chunks that have not started.
            let generation = index == 0 ? remaining * factor : max(
                remaining * factor,
                estimator.conservativeGenerationDuration * Self.unstartedChunkVariabilityReserve)
            // Include a full second per arrival for decoding and delivery.
            arrival += generation + 1
            required = max(required, arrival * max(rate, 0) - delivered)
            delivered += duration
        }
        let result = required + Self.generationSafetyMargin * max(rate, 1)
        return result.isFinite ? result : nil
    }

    func shouldStart(
        synthesisIsComplete: Bool,
        bufferedDuration: TimeInterval,
        estimatedRemainingDuration: TimeInterval = 0
    ) -> Bool {
        if synthesisIsComplete {
            return true
        }
        if mode == .adaptive, receivesWholeChunks, estimator.hasObservations {
            // Later sentences can take substantially longer as speech pace
            // and context length vary. A percentile of early chunks alone
            // underestimates that delivery gap; reserve 50% for whole chunks.
            let factor = estimator.conservativeRealTimeFactor * Self.wholeChunkVariabilityReserve
            let adjustedLoad = factor * max(rate, 0)
            let remaining = max(estimatedRemainingDuration, 0)
            // Audio arrives in whole chunks. Cover the next delivery gap as
            // well as any accumulated deficit while generating later chunks.
            let nextAudio = factor > 0
                ? min(remaining, estimator.conservativeGenerationDuration
                    * Self.wholeChunkVariabilityReserve / factor)
                : 0
            let requiredLead = nextAudio * adjustedLoad
                + max(adjustedLoad - 1, 0) * max(remaining - nextAudio, 0)
                + Self.generationSafetyMargin * max(rate, 1)
            return bufferedDuration >= requiredLead
        }
        if mode == .adaptive, estimator.hasObservations, !streamingIsSustainable {
            let adjustedLoad = estimator.conservativeRealTimeFactor * max(rate, 0)
            let requiredLead = max(adjustedLoad - 1, 0) * max(estimatedRemainingDuration, 0)
                + Self.generationSafetyMargin * max(rate, 1)
            return bufferedDuration >= requiredLead
        }
        guard streamingIsSustainable else { return false }
        if estimator.hasObservations {
            return bufferedDuration >= Self.minimumSustainableStartBuffer
        }
        return bufferedDuration >= preferredSourceLead
    }
}
