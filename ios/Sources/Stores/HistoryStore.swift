import Observation
import Foundation

@MainActor
@Observable
final class HistoryStore {
    private let api = APIClient.shared
    var state: LoadState<BenchmarkHistory> = .idle

    func load() async {
        state = .loading
        do {
            state = .loaded(try await api.get(.benchmarkHistory, as: BenchmarkHistory.self))
        } catch {
            state = .failed((error as? APIError)?.errorDescription ?? error.localizedDescription)
        }
    }
}
