import SwiftUI

@main
struct FFPredictorApp: App {
    @State private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(appState)
                .preferredColorScheme(.dark)
                .tint(FFColor.accent)
                .task { await appState.checkHealth() }
        }
    }
}
