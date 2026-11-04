import Foundation
import SayItCore
import SayItProtocol
import Testing
@testable import SayItBackend

@Suite("Installed model smoke tests", .serialized)
struct InstalledModelSmokeTests {
    @Test("Configured local snapshot imports into the isolated audit profile")
    func configuredModelImportsSnapshot() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelID = auditValue("SAYIT_MODEL_AUDIT_IMPORT_ID", in: environment),
              let source = auditValue("SAYIT_MODEL_AUDIT_IMPORT_SOURCE", in: environment) else {
            return
        }
#if SAYIT_MODEL_AUDIT_BUILD
        let catalog = try ModelCatalogLoader().bundledCatalog()
        let model = try #require(catalog.models.first { $0.id.rawValue == modelID })
        let directories = try AppDirectories.shared(appGroupIdentifier: "app.sayit.audit")
        let manager = ModelManager(
            catalog: catalog, directories: directories, activeModelID: ModelID("model-audit")
        )
        try await manager.importLocalModel(model, from: URL(filePath: source))
        #expect(await manager.installedURL(for: model.id) != nil)
        print("MODEL_AUDIT_IMPORT_RESULT id=\(model.id.rawValue) imported=true")
#else
        Issue.record("Snapshot import requires SAYIT_MODEL_AUDIT_BUILD isolation.")
#endif
    }

    @Test("Configured model completes its production download")
    func configuredModelCompletesDownload() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelID = auditValue(
            "SAYIT_MODEL_AUDIT_INSTALL_ID",
            in: environment
        ) else {
            return
        }

        let catalog = try ModelCatalogLoader().bundledCatalog()
        let model = try #require(catalog.models.first {
            $0.id.rawValue == modelID
        })
        let directories: AppDirectories
        if let installRoot = auditValue(
            "SAYIT_MODEL_AUDIT_INSTALL_ROOT",
            in: environment
        ) {
            directories = try AppDirectories.testing(
                root: URL(filePath: installRoot, directoryHint: .isDirectory)
            )
        } else {
            directories = try AppDirectories.shared(
                appGroupIdentifier: "app.sayit.audit"
            )
        }
        let manager = ModelManager(
            catalog: catalog,
            directories: directories,
            activeModelID: ModelID("model-audit")
        )
        try await manager.install(model.id)
        print("MODEL_AUDIT_DOWNLOAD_RESULT id=\(model.id.rawValue) downloaded=true")
        try await manager.markDependenciesVerified(model.id)

        #expect(await manager.installedURL(for: model.id) != nil)
        print("MODEL_AUDIT_INSTALL_RESULT id=\(model.id.rawValue) installed=true")
    }

    @Test("Configured model generates finite, audible samples")
    @MainActor
    func configuredModelGeneratesAudio() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelID = auditValue("SAYIT_MODEL_AUDIT_ID", in: environment),
              let modelsRoot = auditValue(
                  "SAYIT_MODEL_AUDIT_ROOT",
                  in: environment
              ) else {
            return
        }

        let model = try #require(
            ModelCatalogLoader().bundledCatalog().models.first {
                $0.id.rawValue == modelID
            }
        )
        let modelsRootURL = URL(filePath: modelsRoot, directoryHint: .isDirectory)
        setenv(
            "HF_HUB_CACHE",
            modelsRootURL.deletingLastPathComponent()
                .appending(path: "Hugging Face", directoryHint: .isDirectory)
                .path,
            1
        )
        let modelURL = modelsRootURL
            .appending(path: model.id.rawValue, directoryHint: .isDirectory)
            .appending(path: model.revision, directoryHint: .isDirectory)
        let synthesizer = SynthesisActor(
            modelURLProvider: { requestedID in
                requestedID == model.id ? modelURL : nil
            }
        )
        let voiceReference = auditValue(
            "SAYIT_MODEL_AUDIT_REFERENCE",
            in: environment
        ).map {
            VoiceReference(
                audioURL: URL(filePath: $0),
                transcript: auditValue(
                    "SAYIT_MODEL_AUDIT_REFERENCE_TEXT",
                    in: environment
                )
            )
        }
        let voices: [String?]
        if environment["SAYIT_MODEL_AUDIT_ALL_VOICES"] == "1",
           !model.voices.isEmpty {
            voices = model.voices.map(Optional.some)
        } else if let requestedVoice = auditValue(
            "SAYIT_MODEL_AUDIT_VOICE",
            in: environment
        ) {
            voices = [requestedVoice]
        } else {
            voices = [model.defaultVoice]
        }
        let auditText = auditValue("SAYIT_MODEL_AUDIT_TEXT", in: environment)
        let auditLanguage = auditValue(
            "SAYIT_MODEL_AUDIT_LANGUAGE",
            in: environment
        )
        let auditVoiceDescription = auditValue(
            "SAYIT_MODEL_AUDIT_VOICE_DESCRIPTION",
            in: environment
        )

        for voice in voices {
            let voiceName = voice ?? "none"
            let language = auditLanguage
                ?? model.inferredLanguage(forPresetVoice: voice)
                ?? model.defaultLanguage
            let request = SpeechRequest(
                cleanedText: CleanedText(
                    text: auditText ?? sampleText(for: model, language: language),
                    title: "Model audit",
                    detectedLanguage: language ?? "en",
                    cleanupSummary: CleanupSummary(sourceFormat: "plainText"),
                    requiresLongTextConfirmation: false
                ),
                model: model,
                voice: voice,
                language: language,
                voiceDescription: model.capabilities.voiceDescription
                    ? auditVoiceDescription
                        ?? "A calm, clear adult voice with natural pacing."
                    : nil,
                voiceMode: voiceReference == nil
                    ? model.capabilities.supportsRandomVoiceSampling
                        ? .automaticStable
                        : .standard
                    : .savedProfile,
                voiceReference: voiceReference,
                source: .preview
            )

            let startedAt = ContinuousClock.now
            let playback = environment["SAYIT_MODEL_AUDIT_PLAYBACK"] == "1" ? PlaybackController() : nil
            var firstPlaybackSeconds: Double?
            var rebufferCount = 0
            var baselineFirstStart: Double?
            var baselineEstimator = SynthesisPerformanceEstimator()
            var previousState = PlaybackState.idle
            func elapsedSeconds() -> Double {
                let elapsed = startedAt.duration(to: .now)
                return Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
            }
            playback?.onStateChange = { state in
                if state == .playing, firstPlaybackSeconds == nil { firstPlaybackSeconds = elapsedSeconds() }
                if state == .buffering, previousState == .playing { rebufferCount += 1 }
                previousState = state
            }
            playback?.volume = 0
            if let rate = auditValue("SAYIT_MODEL_AUDIT_PLAYBACK_RATE", in: environment).flatMap(Double.init) {
                playback?.rate = rate
            }
            playback?.setPlaybackMode(model.playbackMode, receivesWholeChunks: !model.capabilities.streaming)
            playback?.prepare(requestID: request.id, title: "Timing audit",
                estimatedDuration: Double(request.cleanedText.text.count) / 14, modelID: model.id.rawValue)
            playback?.setSpokenText(request.cleanedText.text)
            defer { playback?.stop() }
            let stream = await synthesizer.synthesize(request)
            var samples: [Float] = []
            var firstAudioSeconds: Double?
            var audioEventCount = 0
            var sampleRate = 0.0
            var synthesizedChunks: [SpeechChunk] = []
            var chunkDurations: [Double] = []
            for try await event in stream {
                switch event {
                case .chunkStarted(_, let chunk):
                    synthesizedChunks.append(chunk)
                    playback?.appendSpokenChunk(PlaybackTextChunk(
                        textStart: chunk.sourceRange.lowerBound, textEnd: chunk.sourceRange.upperBound,
                        audioStart: playback?.generatedDuration ?? 0))
                case .audio(let chunk):
                    if firstAudioSeconds == nil {
                        let elapsed = startedAt.duration(to: .now)
                        firstAudioSeconds = Double(elapsed.components.seconds)
                            + Double(elapsed.components.attoseconds) / 1e18
                    }
                    audioEventCount += 1
                    samples.append(contentsOf: chunk.samples)
                    sampleRate = chunk.sampleRate
                    try playback?.enqueue(chunk)
                    if playback?.shouldStartWhenBuffered == true { playback?.play() }
                case .progress(let progress):
                    playback?.observeSynthesisProgress(progress)
                    if playback?.shouldStartWhenBuffered == true { playback?.play() }
                    if environment["SAYIT_MODEL_AUDIT_COMPARE_STARTUP"] == "1" {
                        print("MODEL_AUDIT_PROGRESS index=\(progress.chunkIndex) elapsed=\(progress.generationDuration) audio=\(progress.generatedAudioDuration) rtf=\(progress.recentRealTimeFactor)")
                    }
                case .metrics(let metrics):
                    chunkDurations.append(metrics.audioDuration)
                    if let playback, let chunk = playback.spokenChunks.last {
                        playback.appendSpokenChunk(PlaybackTextChunk(textStart: chunk.textStart,
                            textEnd: chunk.textEnd, audioStart: chunk.audioStart,
                            audioEnd: playback.generatedDuration + metrics.trailingAudioDuration))
                        playback.observeSynthesisMetrics(metrics)
                        if playback.shouldStartWhenBuffered { playback.play() }
                    }
                    baselineEstimator.record(metrics)
                    if let playback, baselineFirstStart == nil,
                       PlaybackBufferPolicy(mode: model.playbackMode, rate: playback.rate,
                           estimator: baselineEstimator, receivesWholeChunks: !model.capabilities.streaming)
                        .shouldStart(synthesisIsComplete: false, bufferedDuration: playback.generatedDuration,
                            estimatedRemainingDuration: max(playback.estimatedDuration - playback.generatedDuration, 0)) {
                        baselineFirstStart = elapsedSeconds()
                    }
                    print("MODEL_AUDIT_CHUNK index=\(metrics.chunkIndex) generation=\(metrics.generationDuration) audio=\(metrics.audioDuration) rtf=\(metrics.realTimeFactor)")
                case .timing(let timing):
                    #expect(timing.duration.isFinite && timing.duration >= 0)
                    print("MODEL_AUDIT_TIMING phase=\(timing.phase.rawValue) seconds=\(timing.duration)")
                case .completed:
                    if baselineFirstStart == nil { baselineFirstStart = elapsedSeconds() }
                    playback?.finishBuffering()
                default:
                    break
                }
            }

            let elapsed = startedAt.duration(to: .now)
            let seconds = Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1e18
            if let playback {
                let deadline = ContinuousClock.now.advanced(by: .seconds(playback.generatedDuration + 10))
                while playback.state == .playing || playback.state == .buffering {
                    if ContinuousClock.now >= deadline { break }
                    try await Task.sleep(for: .milliseconds(100))
                }
                #expect(playback.state == .finished)
                #expect(rebufferCount == 0, "Adaptive playback ran out of audio")
                #expect(firstPlaybackSeconds != nil)
                if environment["SAYIT_MODEL_AUDIT_COMPARE_STARTUP"] == "1" {
                    let current = try #require(firstPlaybackSeconds)
                    let baseline = try #require(baselineFirstStart)
                    #expect(current + 0.5 < baseline, "Progress must start playback earlier on this same run")
                    print("MODEL_AUDIT_STARTUP baseline=\(baseline) progress=\(current) saved=\(baseline - current)")
                }
                print("MODEL_AUDIT_PLAYBACK firstStart=\(firstPlaybackSeconds ?? -1) synthesisComplete=\(seconds) rebuffers=\(rebufferCount)")
            }
            let finiteSamples = samples.filter(\.isFinite)
            let peak = finiteSamples.map { abs($0) }.max() ?? 0
            let rms = finiteSamples.isEmpty
                ? 0
                : sqrt(
                    finiteSamples.reduce(0.0) { sum, sample in
                        sum + Double(sample * sample)
                    } / Double(finiteSamples.count)
                )
            let audioDuration = sampleRate > 0
                ? Double(samples.count) / sampleRate
                : 0
            print(
                "MODEL_AUDIT_RESULT id=\(model.id.rawValue) "
                    + "voice=\(voiceName) "
                    + "samples=\(samples.count) rate=\(sampleRate) "
                    + "duration=\(audioDuration) peak=\(peak) rms=\(rms) "
                    + "elapsed=\(seconds) firstAudio=\(firstAudioSeconds ?? -1) "
                    + "audioEvents=\(audioEventCount) "
                    + "rtf=\(audioDuration > 0 ? seconds / audioDuration : -1)"
            )

            #expect(samples.count > 1_000, "No audio for voice \(voiceName)")
            #expect(
                finiteSamples.count == samples.count,
                "Non-finite audio for voice \(voiceName)"
            )
            #expect(sampleRate >= 8_000, "Invalid rate for voice \(voiceName)")
            #expect(peak > 0.001, "Silent audio for voice \(voiceName)")
            #expect(rms > 0.0001, "Silent audio for voice \(voiceName)")

            if let outputPath = auditValue(
                "SAYIT_MODEL_AUDIT_OUTPUT",
                in: environment
            ),
               voices.count == 1 {
                let outputURL = URL(filePath: outputPath)
                let outputDirectory = outputURL.deletingLastPathComponent()
                try FileManager.default.createDirectory(
                    at: outputDirectory,
                    withIntermediateDirectories: true
                )
                let archive = AudioArchive(directory: outputDirectory)
                try await archive.writeWAV(
                    samples: samples,
                    sampleRate: sampleRate,
                    destination: outputURL
                )
                let diagnostics = synthesizedChunks.enumerated().map {
                    index, chunk in
                    let duration = chunkDurations.indices.contains(index)
                        ? chunkDurations[index]
                        : 0
                    return "\(chunk.id)\t\(chunk.text.count)\t\(duration)\t\(chunk.text)"
                }.joined(separator: "\n")
                try diagnostics.write(
                    to: outputURL.appendingPathExtension("chunks.txt"),
                    atomically: true,
                    encoding: .utf8
                )
                print("MODEL_AUDIT_OUTPUT path=\(outputURL.path)")
            }
        }
        await synthesizer.unloadModel()
    }

    private func auditValue(
        _ key: String,
        in environment: [String: String]
    ) -> String? {
        guard let value = environment[key],
              !value.isEmpty,
              !value.hasPrefix("$(") else {
            return nil
        }
        return value
    }

    private func sampleText(
        for model: ModelDescriptor,
        language: String?
    ) -> String {
        switch language?.lowercased() {
        case "es":
            "Hoy es un día tranquilo. Esta es una prueba breve de una voz clara."
        case "fr":
            "Aujourd’hui est une journée calme. Ceci est un bref test vocal."
        case "ja":
            "今日は穏やかな一日です。自然で明瞭な声のテストをしています。"
        case "cmn", "zh":
            "今天是平静的一天。这是一次清晰自然的语音测试。"
        default:
            "Today is a calm day. This is a short test of clear, natural speech."
        }
    }
}
