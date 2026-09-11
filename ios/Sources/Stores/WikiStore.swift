import Observation
import Foundation

@MainActor
@Observable
final class WikiStore {
    private let api: any APIProviding

    init(api: any APIProviding = APIClient.shared) { self.api = api }
    var index: LoadState<[WikiIndexEntry]> = .idle

    func loadIndex() async {
        if index.value != nil { return }
        index = .loading
        do {
            index = .loaded(try await api.get(.wikiIndex, as: [WikiIndexEntry].self))
        } catch {
            index = .failed((error as? APIError)?.errorDescription ?? error.localizedDescription)
        }
    }
}
