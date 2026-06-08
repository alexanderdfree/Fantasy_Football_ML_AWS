import Observation
import SwiftUI

/// Global, observable app state injected via `.environment`. Holds the two
/// cross-screen toggles (scoring format + model display) — persisted to
/// UserDefaults under the same keys the web frontend uses in localStorage —
/// plus the last-seen degraded positions and a `/health` probe.
@MainActor
@Observable
final class AppState {
    var scoring: ScoringFormat {
        didSet { UserDefaults.standard.set(scoring.rawValue, forKey: Keys.scoring) }
    }

    /// nil == "All models" (show all four), matching the web default.
    var selectedModel: PredictionModel? {
        didSet { UserDefaults.standard.set(selectedModel?.rawValue ?? "", forKey: Keys.model) }
    }

    var degradedPositions: [String] = []
    var health: LoadState<Health> = .idle

    private enum Keys {
        static let scoring = "scoringFormat"
        static let model = "modelDisplay"
    }

    init() {
        let s = UserDefaults.standard.string(forKey: Keys.scoring) ?? ""
        scoring = ScoringFormat(rawValue: s) ?? .ppr
        let m = UserDefaults.standard.string(forKey: Keys.model) ?? ""
        selectedModel = PredictionModel(rawValue: m)
    }

    func checkHealth() async {
        health = .loading
        do {
            health = .loaded(try await APIClient.shared.get(.health, as: Health.self))
        } catch {
            health = .failed((error as? APIError)?.errorDescription ?? error.localizedDescription)
        }
    }
}
