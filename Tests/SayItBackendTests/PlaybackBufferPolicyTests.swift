import Foundation
import SayItCore
import Testing
@testable import SayItBackend

@Suite("Playback buffer policy")
struct PlaybackBufferPolicyTests {
    @Test("Paid prefill does not delay streaming until the full first chunk")
    func prefillAlreadyBuffered() {
        let streaming = PlaybackBufferPolicy(mode: .adaptive, rate: 1.2,
            estimator: SynthesisPerformanceEstimator())
        // Captured Q4 trace: 2.14s first delivery, then sustained faster output.
        let progress = SynthesisProgress(chunkIndex: 0, generatedAudioDuration: 6.679,
            generationDuration: 6.416, recentRealTimeFactor: 0.6004,
            maximumAudioDuration: 6.679, completedTextCharacters: 0, pendingTextCharacters: [],
            deliveredAudioPieces: 8, maximumDeliveryGap: 2.137,
            initialGenerationDuration: 2.137, initialAudioDuration: 0.735)
        #expect(streaming.shouldStartStreaming(progress, bufferedDuration: 6.679))
        #expect(!streaming.shouldStartStreaming(progress, bufferedDuration: 5))
        let notDelivered = SynthesisProgress(chunkIndex: 0, generatedAudioDuration: 6.679,
            generationDuration: 6.416, recentRealTimeFactor: 0.6004,
            maximumAudioDuration: 6.679, completedTextCharacters: 0, pendingTextCharacters: [],
            deliveredAudioPieces: 8, maximumDeliveryGap: 2.137,
            initialGenerationDuration: 6.416, initialAudioDuration: 6.679)
        #expect(!streaming.shouldStartStreaming(notDelivered, bufferedDuration: 100))
    }

    @Test("Delivered streaming pieces qualify startup without a complete text chunk")
    func deliveredStreamingStartup() {
        let estimator = SynthesisPerformanceEstimator()
        let streaming = PlaybackBufferPolicy(mode: .adaptive, rate: 1.2, estimator: estimator)
        func progress(audio: Double = 4, elapsed: Double = 1.6, recent: Double = 0.4,
                      pieces: Int = 6, gap: Double = 0.6) -> SynthesisProgress {
            SynthesisProgress(chunkIndex: 0, generatedAudioDuration: audio,
                generationDuration: elapsed, recentRealTimeFactor: recent, maximumAudioDuration: audio,
                completedTextCharacters: 0, pendingTextCharacters: [],
                deliveredAudioPieces: pieces, maximumDeliveryGap: gap)
        }
        #expect(!streaming.shouldStart(synthesisIsComplete: false, bufferedDuration: 4))
        #expect(streaming.shouldStartStreaming(progress(), bufferedDuration: 4))
        #expect(!streaming.shouldStartStreaming(progress(pieces: 0), bufferedDuration: 100))
        #expect(!streaming.shouldStartStreaming(progress(audio: 2), bufferedDuration: 100))
        #expect(!streaming.shouldStartStreaming(progress(), bufferedDuration: 2.9))
        #expect(!streaming.shouldStartStreaming(progress(recent: 0.8), bufferedDuration: 100))
        #expect(!streaming.shouldStartStreaming(progress(elapsed: 4), bufferedDuration: 100))
        #expect(!streaming.shouldStartStreaming(progress(gap: 2), bufferedDuration: 4))
        #expect(!streaming.shouldStartStreaming(progress(recent: .nan), bufferedDuration: 100))
        #expect(!streaming.shouldStartStreaming(progress(), bufferedDuration: .infinity))
        #expect(!PlaybackBufferPolicy(mode: .completeFirst, rate: 1, estimator: estimator)
            .shouldStartStreaming(progress(), bufferedDuration: 100))
        #expect(!PlaybackBufferPolicy(mode: .adaptive, rate: 1, estimator: estimator, receivesWholeChunks: true)
            .shouldStartStreaming(progress(), bufferedDuration: 100))
        #expect(!PlaybackBufferPolicy(mode: .adaptive, rate: 2, estimator: estimator)
            .shouldStartStreaming(progress(), bufferedDuration: 100))
        var slow = estimator
        slow.record(SynthesisMetrics(chunkIndex: 0, generationDuration: 10, audioDuration: 8))
        #expect(!PlaybackBufferPolicy(mode: .adaptive, rate: 1, estimator: slow)
            .shouldStartStreaming(progress(), bufferedDuration: 100))
    }

    @Test("Early streaming reserve covers variable arrivals and chunk-boundary delays")
    func streamingDeliveryTimeline() {
        let streaming = PlaybackBufferPolicy(mode: .adaptive, rate: 1.2,
            estimator: SynthesisPerformanceEstimator())
        var elapsed = 0.0
        var delivered = 0.0
        var firstStart: Double?
        var largestGap = 0.0
        var recent: [Double] = []
        for index in 0..<80 {
            let gap = index == 0 ? 0.8 : index.isMultiple(of: 20) ? 1.4 : 0.25
            elapsed += gap
            if let firstStart { #expect(delivered >= (elapsed - firstStart) * 1.2) }
            delivered += 0.74
            largestGap = max(largestGap, gap)
            recent.append(gap)
            if recent.count > 4 { recent.removeFirst() }
            let progress = SynthesisProgress(chunkIndex: 0, generatedAudioDuration: delivered,
                generationDuration: elapsed, recentRealTimeFactor: recent.reduce(0, +) / (Double(recent.count) * 0.74),
                maximumAudioDuration: delivered, completedTextCharacters: 0, pendingTextCharacters: [],
                deliveredAudioPieces: index + 1, maximumDeliveryGap: largestGap)
            if firstStart == nil, streaming.shouldStartStreaming(progress, bufferedDuration: delivered) {
                firstStart = elapsed
            }
        }
        #expect(firstStart != nil)
        #expect((firstStart ?? elapsed) < 3)
    }

    @Test("Playback modes choose distinct baseline leads")
    func modeBaselines() {
        let estimator = SynthesisPerformanceEstimator()

        #expect(
            policy(.progressive, estimator: estimator).preferredSourceLead
                == 1.2
        )
        #expect(
            policy(.buffered, estimator: estimator).preferredSourceLead
                == 2.5
        )
        #expect(
            policy(.completeFirst, estimator: estimator).preferredSourceLead
                == .infinity
        )
    }

    @Test("Complete-first waits for synthesis completion")
    func completeFirst() {
        let policy = policy(.completeFirst)

        #expect(
            !policy.shouldStart(
                synthesisIsComplete: false,
                bufferedDuration: 10_000
            )
        )
        #expect(
            policy.shouldStart(
                synthesisIsComplete: true,
                bufferedDuration: 0
            )
        )
    }

    @Test("Adaptive playback waits for a measured generation rate")
    func adaptiveWaitsForMeasurement() {
        let unmeasured = policy(.adaptive)
        #expect(!unmeasured.streamingIsSustainable)
        #expect(
            !unmeasured.shouldStart(
                synthesisIsComplete: false,
                bufferedDuration: 100
            )
        )

        var estimator = SynthesisPerformanceEstimator()
        estimator.record(
            SynthesisMetrics(
                chunkIndex: 0,
                generationDuration: 1,
                audioDuration: 4
            )
        )
        let measured = policy(.adaptive, estimator: estimator)
        #expect(measured.streamingIsSustainable)
        #expect(
            measured.shouldStart(
                synthesisIsComplete: false,
                bufferedDuration: 4
            )
        )
    }

    @Test("Adaptive playback builds enough lead for a slow model")
    func adaptiveSlowModelLead() {
        var estimator = SynthesisPerformanceEstimator()
        estimator.record(
            SynthesisMetrics(
                chunkIndex: 0,
                generationDuration: 8,
                audioDuration: 4
            )
        )
        let measured = policy(.adaptive, estimator: estimator)

        #expect(
            !measured.shouldStart(
                synthesisIsComplete: false,
                bufferedDuration: 16.39,
                estimatedRemainingDuration: 6
            )
        )
        #expect(
            measured.shouldStart(
                synthesisIsComplete: false,
                bufferedDuration: 16.41,
                estimatedRemainingDuration: 6
            )
        )
    }

    @Test("Existing adaptive streaming retains its smaller startup buffer")
    func incrementalStreamingUnchanged() {
        var estimator = SynthesisPerformanceEstimator()
        estimator.record(SynthesisMetrics(chunkIndex: 0, generationDuration: 3, audioDuration: 4))
        let streaming = PlaybackBufferPolicy(mode: .adaptive, rate: 1, estimator: estimator)
        #expect(streaming.shouldStart(synthesisIsComplete: false,
            bufferedDuration: 0.15, estimatedRemainingDuration: 20))
        estimator = SynthesisPerformanceEstimator()
        estimator.record(SynthesisMetrics(chunkIndex: 0, generationDuration: 8, audioDuration: 4))
        let slowStreaming = PlaybackBufferPolicy(mode: .adaptive, rate: 1, estimator: estimator)
        #expect(slowStreaming.shouldStart(synthesisIsComplete: false,
            bufferedDuration: 6.41, estimatedRemainingDuration: 6))
    }

    @Test("Adaptive playback covers the next whole chunk even for a fast model")
    func adaptiveDeliveryGap() {
        var estimator = SynthesisPerformanceEstimator()
        estimator.record(SynthesisMetrics(chunkIndex: 0, generationDuration: 3, audioDuration: 4))
        let measured = policy(.adaptive, estimator: estimator)
        #expect(!measured.shouldStart(synthesisIsComplete: false,
            bufferedDuration: 0.15, estimatedRemainingDuration: 20))
        #expect(measured.shouldStart(synthesisIsComplete: false,
            bufferedDuration: 6.91, estimatedRemainingDuration: 20))
        let fasterPlayback = policy(.adaptive, rate: 2, estimator: estimator)
        #expect(!fasterPlayback.shouldStart(synthesisIsComplete: false,
            bufferedDuration: 29.79, estimatedRemainingDuration: 20))
        #expect(fasterPlayback.shouldStart(synthesisIsComplete: false,
            bufferedDuration: 29.81, estimatedRemainingDuration: 20))
    }

    @Test("Slow chunk arrivals can start before completion without exhausting the buffer")
    func adaptiveChunkTimeline() {
        var estimator = SynthesisPerformanceEstimator()
        var firstStart: Double?
        let totalAudio = 32.0
        for index in 0..<8 {
            let arrival = Double(index + 1) * 8
            let generated = Double(index + 1) * 4
            if let firstStart {
                // Immediately before this chunk arrives, previously generated
                // audio must still cover everything played since startup.
                #expect(generated - 4 >= arrival - firstStart)
            }
            estimator.record(SynthesisMetrics(chunkIndex: index, generationDuration: 8, audioDuration: 4))
            if firstStart == nil, policy(.adaptive, estimator: estimator).shouldStart(
                synthesisIsComplete: index == 7, bufferedDuration: generated,
                estimatedRemainingDuration: totalAudio - generated) {
                firstStart = arrival
            }
        }
        #expect(firstStart == 48)
        #expect(firstStart! < 64)
        #expect(!policy(.buffered, estimator: estimator).shouldStart(
            synthesisIsComplete: false, bufferedDuration: 28, estimatedRemainingDuration: 4))
    }

    @Test("Measured Q4 slowdown keeps playback ahead of later chunk arrivals")
    func variableWholeChunkTimeline() {
        // Regression: starting after chunk two exhausted the queue before
        // the slower third chunk arrived, despite the early RTF being stable.
        let generation = [10.275, 14.045, 22.615, 17.486]
        let audio = [9.048, 12.159, 16.060, 11.566]
        var estimator = SynthesisPerformanceEstimator()
        var arrival = 0.0
        var generated = 0.0
        var firstStart: Double?
        for index in generation.indices {
            arrival += generation[index]
            if let firstStart { #expect(generated >= arrival - firstStart) }
            generated += audio[index]
            estimator.record(SynthesisMetrics(chunkIndex: index,
                generationDuration: generation[index], audioDuration: audio[index]))
            let fraction = Double(index + 1) / Double(generation.count)
            let remaining = generated * (1 - fraction) / fraction
            if firstStart == nil, policy(.adaptive, estimator: estimator).shouldStart(
                synthesisIsComplete: index == generation.count - 1,
                bufferedDuration: generated, estimatedRemainingDuration: remaining) {
                firstStart = arrival
            }
        }
        #expect(firstStart == generation.prefix(3).reduce(0, +))
        #expect(firstStart! < arrival)
    }

    @Test("Live progress starts earlier on both recorded Q4 traces without underruns")
    func progressTimelines() {
        for (generation, audio) in [
            ([10.275, 14.045, 22.615, 17.486], [9.048, 12.159, 16.060, 11.566]),
            ([9.538, 14.065, 49.139, 22.723], [8.955, 13.042, 17.732, 13.005])
        ] {
            var estimator = SynthesisPerformanceEstimator()
            var arrival = 0.0
            var generated = 0.0
            var firstStart: Double?
            var baselineStart: Double?
            for index in generation.indices {
                for elapsed in stride(from: 0.5, to: generation[index], by: 0.5) {
                    let progress = SynthesisProgress(chunkIndex: index,
                        generatedAudioDuration: audio[index] * elapsed / generation[index],
                        generationDuration: elapsed,
                        recentRealTimeFactor: generation[index] / audio[index],
                        maximumAudioDuration: 28, completedTextCharacters: index * 200,
                        pendingTextCharacters: Array(repeating: 200, count: generation.count - index))
                    if firstStart == nil,
                       let lead = policy(.adaptive, estimator: estimator)
                        .requiredProgressLead(progress, completedAudioDuration: generated), generated >= lead {
                        firstStart = arrival + elapsed
                    }
                }
                arrival += generation[index]
                if let firstStart { #expect(generated >= arrival - firstStart) }
                generated += audio[index]
                estimator.record(SynthesisMetrics(chunkIndex: index,
                    generationDuration: generation[index], audioDuration: audio[index]))
                let remaining = generated * Double(generation.count - index - 1) / Double(index + 1)
                if policy(.adaptive, estimator: estimator).shouldStart(
                    synthesisIsComplete: index == generation.count - 1,
                    bufferedDuration: generated, estimatedRemainingDuration: remaining) {
                    if firstStart == nil { firstStart = arrival }
                    if baselineStart == nil { baselineStart = arrival }
                }
            }
            #expect(firstStart != nil)
            #expect(firstStart! + 1 < baselineStart!)
        }
    }

    @Test("Progress rejects invalid or insufficient observations and accounts for playback rate")
    func progressValidation() {
        func progress(index: Int = 2, elapsed: Double = 6, recent: Double = 1,
                      audio: Double = 6, pending: [Int] = [100]) -> SynthesisProgress {
            SynthesisProgress(chunkIndex: index, generatedAudioDuration: audio,
                generationDuration: elapsed, recentRealTimeFactor: recent,
                maximumAudioDuration: 28, completedTextCharacters: 200, pendingTextCharacters: pending)
        }
        let normal = policy(.adaptive)
        #expect(normal.requiredProgressLead(progress(index: 1), completedAudioDuration: 20) == nil)
        #expect(normal.requiredProgressLead(progress(elapsed: .nan), completedAudioDuration: 20) == nil)
        #expect(normal.requiredProgressLead(progress(recent: .infinity), completedAudioDuration: 20) == nil)
        #expect(normal.requiredProgressLead(progress(audio: 1), completedAudioDuration: 20) == nil)
        #expect(normal.requiredProgressLead(progress(pending: []), completedAudioDuration: 20) == nil)
        #expect(normal.requiredProgressLead(progress(), completedAudioDuration: 0) == nil)
        let lead = normal.requiredProgressLead(progress(), completedAudioDuration: 20)!
        #expect(policy(.adaptive, rate: 2).requiredProgressLead(progress(), completedAudioDuration: 20)! == lead * 2)
        let streaming = PlaybackBufferPolicy(mode: .adaptive, rate: 1, estimator: SynthesisPerformanceEstimator())
        #expect(streaming.requiredProgressLead(progress(), completedAudioDuration: 20) == nil)
    }

    @Test("A short final chunk retains a generation budget before it starts")
    func shortTailTimingRegression() {
        // Recorded Q4 trace: the last, shortest chunk took 22s to generate,
        // plus 6.3s before the consumer finished receiving the final events.
        let generation = [14.825, 11.241, 20.663, 22.061]
        let deliveryDelay = [0.0, 0, 0, 6.314]
        let audio = [9.001, 8.490, 13.738, 5.805]
        let characters = [156, 178, 243, 115]
        var estimator = SynthesisPerformanceEstimator()
        var arrival = 0.0
        var generated = 0.0
        var firstStart: Double?
        for index in generation.indices {
            for elapsed in stride(from: 0.5, to: generation[index], by: 0.5) {
                let progress = SynthesisProgress(chunkIndex: index,
                    generatedAudioDuration: audio[index] * elapsed / generation[index],
                    generationDuration: elapsed,
                    recentRealTimeFactor: generation[index] / audio[index],
                    maximumAudioDuration: 28,
                    completedTextCharacters: characters.prefix(index).reduce(0, +),
                    pendingTextCharacters: Array(characters[index...]))
                if firstStart == nil,
                   let lead = policy(.adaptive, estimator: estimator)
                    .requiredProgressLead(progress, completedAudioDuration: generated),
                   generated >= lead {
                    firstStart = arrival + elapsed
                }
            }
            arrival += generation[index] + deliveryDelay[index]
            if let firstStart {
                #expect(generated >= arrival - firstStart, "Playback must cover the next delivery")
            }
            generated += audio[index]
            estimator.record(SynthesisMetrics(chunkIndex: index,
                generationDuration: generation[index], audioDuration: audio[index]))
            let remaining = generated / Double(characters.prefix(index + 1).reduce(0, +))
                * Double(characters.dropFirst(index + 1).reduce(0, +))
            if firstStart == nil, policy(.adaptive, estimator: estimator).shouldStart(
                synthesisIsComplete: index == generation.count - 1,
                bufferedDuration: generated, estimatedRemainingDuration: remaining) {
                firstStart = arrival
            }
        }
        #expect(firstStart != nil)
        #expect(firstStart! < arrival, "Playback should still overlap the final generation")
    }

    @Test("Known-sustainable generation starts with minimal buffer")
    func observedLatency() {
        var estimator = SynthesisPerformanceEstimator()
        estimator.record(
            SynthesisMetrics(
                chunkIndex: 0,
                generationDuration: 3,
                audioDuration: 10
            )
        )
        let policy = policy(.progressive, rate: 1.5, estimator: estimator)

        #expect(abs(policy.preferredSourceLead - 5.1) < 0.000_001)
        #expect(
            !policy.shouldStart(
                synthesisIsComplete: false,
                bufferedDuration: 0.14
            )
        )
        #expect(
            policy.shouldStart(
                synthesisIsComplete: false,
                bufferedDuration: 0.15
            )
        )
    }

    @Test("Unsustainable generation waits for completion after an underrun")
    func unsustainableGeneration() {
        var estimator = SynthesisPerformanceEstimator()
        estimator.record(
            SynthesisMetrics(
                chunkIndex: 0,
                generationDuration: 9.5,
                audioDuration: 10
            )
        )
        let policy = policy(.progressive, estimator: estimator)

        #expect(!policy.streamingIsSustainable)
        #expect(
            !policy.shouldStart(
                synthesisIsComplete: false,
                bufferedDuration: 100
            )
        )
        #expect(
            policy.shouldStart(
                synthesisIsComplete: true,
                bufferedDuration: 0
            )
        )
    }

    @Test("Playback speed participates in sustainability")
    func rateAdjustedSustainability() {
        var estimator = SynthesisPerformanceEstimator()
        estimator.record(
            SynthesisMetrics(
                chunkIndex: 0,
                generationDuration: 6,
                audioDuration: 10
            )
        )

        #expect(policy(.progressive, rate: 1, estimator: estimator)
            .streamingIsSustainable)
        #expect(!policy(.progressive, rate: 2, estimator: estimator)
            .streamingIsSustainable)
    }

    @Test("Unknown generation speed waits for the baseline lead")
    func unknownSpeedWaitsForBaseline() {
        let policy = policy(.progressive)

        #expect(
            !policy.shouldStart(
                synthesisIsComplete: false,
                bufferedDuration: 1.19
            )
        )
        #expect(
            policy.shouldStart(
                synthesisIsComplete: false,
                bufferedDuration: 1.2
            )
        )
    }

    @Test("Estimator is bounded and uses a conservative percentile")
    func boundedEstimator() {
        var estimator = SynthesisPerformanceEstimator()
        for index in 0..<20 {
            estimator.record(
                SynthesisMetrics(
                    chunkIndex: index,
                    generationDuration: Double(index),
                    audioDuration: 10
                )
            )
        }

        #expect(
            estimator.observations.count
                == SynthesisPerformanceEstimator.maximumObservationCount
        )
        #expect(estimator.observations.first?.chunkIndex == 8)
        #expect(estimator.conservativeGenerationDuration == 18)
    }

    @Test("Invalid observations do not influence policy")
    func invalidObservations() {
        var estimator = SynthesisPerformanceEstimator()
        estimator.record(
            SynthesisMetrics(
                chunkIndex: 0,
                generationDuration: .nan,
                audioDuration: 1
            )
        )
        estimator.record(
            SynthesisMetrics(
                chunkIndex: 1,
                generationDuration: 1,
                audioDuration: 0
            )
        )

        #expect(!estimator.hasObservations)
    }

    private func policy(
        _ mode: PlaybackMode,
        rate: Double = 1,
        estimator: SynthesisPerformanceEstimator =
            SynthesisPerformanceEstimator()
    ) -> PlaybackBufferPolicy {
        PlaybackBufferPolicy(
            mode: mode,
            rate: rate,
            estimator: estimator,
            receivesWholeChunks: true
        )
    }
}
