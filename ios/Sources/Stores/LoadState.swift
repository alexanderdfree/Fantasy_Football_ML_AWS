import Foundation

/// Generic async screen state. `failed` carries a display message (errors are
/// reduced to strings at the store boundary so views stay simple).
enum LoadState<Value> {
    case idle
    case loading
    case loaded(Value)
    case failed(String)

    var value: Value? {
        if case let .loaded(v) = self { return v }
        return nil
    }

    var isLoading: Bool {
        if case .loading = self { return true }
        return false
    }

    var errorMessage: String? {
        if case let .failed(m) = self { return m }
        return nil
    }
}
