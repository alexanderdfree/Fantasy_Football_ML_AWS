import SwiftUI

/// Renders a `LoadState`: spinner while loading, content when loaded, an
/// error state (with optional retry) on failure.
struct LoadStateView<Value, Content: View>: View {
    let state: LoadState<Value>
    var retry: (() -> Void)? = nil
    @ViewBuilder var content: (Value) -> Content

    var body: some View {
        switch state {
        case .idle, .loading:
            ProgressView()
                .tint(FFColor.accent)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 40)
        case let .loaded(value):
            content(value)
        case let .failed(message):
            EmptyStateView(
                icon: "wifi.exclamationmark",
                title: "Couldn't load",
                message: message,
                retry: retry
            )
        }
    }
}
