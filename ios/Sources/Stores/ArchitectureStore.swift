import Observation
import Foundation

@MainActor
@Observable
final class ArchitectureStore {
    private let api: any APIProviding

    init(api: any APIProviding = APIClient.shared) { self.api = api }
    var state: LoadState<ModelArchitecture> = .idle

    func load() async {
        if state.value != nil { return } // scoring-invariant — fetch once
        state = .loading
        do {
            state = .loaded(try await api.get(.modelArchitecture, as: ModelArchitecture.self))
        } catch {
            state = .failed((error as? APIError)?.errorDescription ?? error.localizedDescription)
        }
    }
}
