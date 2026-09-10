import Observation
import Foundation

@MainActor
@Observable
final class ArchitectureStore {
    private let api: APIClient
    private var loadGeneration = 0
    var state: LoadState<ModelArchitecture> = .idle

    init(api: APIClient = .shared) { self.api = api }

    func load() async {
        guard !Task.isCancelled else { return }
        if state.value != nil { return } // scoring-invariant — fetch once
        loadGeneration += 1
        let generation = loadGeneration
        state = .loading
        do {
            let value = try await api.get(.modelArchitecture, as: ModelArchitecture.self)
            guard generation == loadGeneration, !Task.isCancelled else { return }
            state = .loaded(value)
        } catch {
            guard generation == loadGeneration, !Task.isCancelled else { return }
            state = .failed((error as? APIError)?.errorDescription ?? error.localizedDescription)
        }
    }
}
