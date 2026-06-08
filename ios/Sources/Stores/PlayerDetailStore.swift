import Observation
import Foundation

@MainActor
@Observable
final class PlayerDetailStore {
    private let api = APIClient.shared

    var detail: LoadState<PlayerDetail> = .idle
    /// nil = no week context (e.g. from Upcoming); otherwise the week breakdown.
    var breakdown: LoadState<Breakdown>?

    func load(playerID: String, week: Int?, scoring: ScoringFormat) async {
        detail = .loading
        do {
            detail = .loaded(try await api.get(.player(id: playerID, scoring: scoring), as: PlayerDetail.self))
        } catch {
            detail = .failed(message(error))
        }
        // Breakdown is scoring-invariant — fetch once when a week is known.
        if let week, breakdown?.value == nil {
            breakdown = .loading
            do {
                breakdown = .loaded(try await api.get(.breakdown(playerID: playerID, week: week), as: Breakdown.self))
            } catch {
                breakdown = .failed(message(error))
            }
        }
    }

    private func message(_ error: Error) -> String {
        (error as? APIError)?.errorDescription ?? error.localizedDescription
    }
}
