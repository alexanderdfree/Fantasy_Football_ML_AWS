import Observation
import Foundation

@MainActor
@Observable
final class WikiStore {
    private let api: APIClient
    private var loadGeneration = 0
    var index: LoadState<[WikiIndexEntry]> = .idle

    init(api: APIClient = .shared) { self.api = api }

    func loadIndex() async {
        guard !Task.isCancelled else { return }
        if index.value != nil { return }
        loadGeneration += 1
        let generation = loadGeneration
        index = .loading
        do {
            let value = try await api.get(.wikiIndex, as: [WikiIndexEntry].self)
            guard generation == loadGeneration, !Task.isCancelled else { return }
            index = .loaded(value)
        } catch {
            guard generation == loadGeneration, !Task.isCancelled else { return }
            index = .failed((error as? APIError)?.errorDescription ?? error.localizedDescription)
        }
    }
}
