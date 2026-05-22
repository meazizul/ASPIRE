// swift-tools-version:6.0
import PackageDescription

let package = Package(
  name: "SystemAudioDump",
  platforms: [
    .macOS(.v15)
  ],
  dependencies: [
    // no external deps
  ],
  targets: [
    .executableTarget(
      name: "SystemAudioDump",
      dependencies: [],
      path: "Sources/SystemAudioDump"
    )
  ]
)
