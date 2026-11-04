// swift-tools-version: 6.2
import PackageDescription
let package = Package(
    name: "DramaTTS",
    platforms: [.macOS(.v15)],
    products: [.library(name: "DramaTTS", targets: ["DramaTTS"]),
               .library(name: "SpeechSynthesizer", targets: ["SpeechSynthesizer"])],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.32.3"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", exact: "3.32.3"),
        .package(url: "https://github.com/Blaizzy/mlx-audio-swift.git", exact: "0.1.5"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.3")
    ],
    targets: [
        .target(name: "SpeechSynthesizer"),
        .target(name: "DramaTTS", dependencies: ["SpeechSynthesizer",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            .product(name: "MLXAudioCodecs", package: "mlx-audio-swift"),
            .product(name: "Transformers", package: "swift-transformers")]),
        .testTarget(name: "DramaTTSTests", dependencies: ["DramaTTS", "SpeechSynthesizer",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift")], resources: [.copy("Fixtures")])
    ])
