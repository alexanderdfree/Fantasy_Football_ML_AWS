import Observation
import Foundation

@MainActor
@Observable
final class PlayerDetailStore {
    private let api: APIClient
    private var loadGeneration = 0

    init(api: APIClient = .shared) { self.api = api }

    var detail: LoadState<PlayerDetail> = .idle
    /// nil = no week context (e.g. from Upcoming); otherwise the week breakdown.
    var breakdown: LoadState<Breakdown>?

    func load(playerID: String, week: Int?, scoring: ScoringFormat) async {
        loadGeneration += 1
        let generation = loadGeneration
        detail = .loading
        do {
            let value = try await api.get(.player(id: playerID, scoring: scoring), as: PlayerDetail.self)
            guard generation == loadGeneration else { return }
            detail = .loaded(value)
        } catch {
            guard generation == loadGeneration else { return }
            detail = .failed(message(error))
        }
        // Breakdown is scoring-invariant — fetch once when a week is known.
        if let week, breakdown?.value == nil {
            breakdown = .loading
            do {
                let value = try await api.get(.breakdown(playerID: playerID, week: week), as: Breakdown.self)
                guard generation == loadGeneration else { return }
                breakdown = .loaded(value)
            } catch {
                guard generation == loadGeneration else { return }
                breakdown = .failed(message(error))
            }
        }
    }

    private func message(_ error: Error) -> String {
        (error as? APIError)?.errorDescription ?? error.localizedDescription
    }
}
