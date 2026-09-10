import Observation
import Foundation

@MainActor
@Observable
final class HistoryStore {
    private let api: APIClient
    private var loadGeneration = 0
    var state: LoadState<BenchmarkHistory> = .idle

    init(api: APIClient = .shared) { self.api = api }

    func load() async {
        guard !Task.isCancelled else { return }
        loadGeneration += 1
        let generation = loadGeneration
        state = .loading
        do {
            let value = try await api.get(.benchmarkHistory, as: BenchmarkHistory.self)
            guard generation == loadGeneration, !Task.isCancelled else { return }
            state = .loaded(value)
        } catch {
            guard generation == loadGeneration, !Task.isCancelled else { return }
            state = .failed((error as? APIError)?.errorDescription ?? error.localizedDescription)
        }
    }
}
