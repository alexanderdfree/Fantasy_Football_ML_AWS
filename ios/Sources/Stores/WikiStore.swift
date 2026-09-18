import Observation
import Foundation

@MainActor
@Observable
final class WikiStore {
    private let api: any APIProviding
    private var loadGeneration = 0

    init(api: any APIProviding = APIClient.shared) { self.api = api }
    var index: LoadState<[WikiIndexEntry]> = .idle

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
