import Observation
import Foundation

@MainActor
@Observable
final class ComparisonStore {
    private let api = APIClient.shared
    var state: LoadState<Comparison> = .idle

    func load() async {
        if state.value != nil { return } // scoring-invariant (PPR-pinned) — fetch once
        state = .loading
        do {
            state = .loaded(try await api.get(.comparison, as: Comparison.self))
        } catch {
            state = .failed((error as? APIError)?.errorDescription ?? error.localizedDescription)
        }
    }
}
