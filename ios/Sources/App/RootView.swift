import SwiftUI

/// Bottom tab bar. Five tabs, each its own NavigationStack so back-stacks are
/// independent. Player detail + wiki doc are value destinations registered on
/// every stack (player detail is reachable from several tabs).
struct RootView: View {
    var body: some View {
        TabView {
            stack { UpcomingView() }
                .tabItem { Label("Next Week", systemImage: "sportscourt.fill") }

            stack { LeadersView() }
                .tabItem { Label("Leaders", systemImage: "list.number") }

            stack { PerformanceView() }
                .tabItem { Label("Accuracy", systemImage: "chart.bar.fill") }

            stack { ComparisonView() }
                .tabItem { Label("Compare", systemImage: "scalemass.fill") }

            stack { MoreView() }
                .tabItem { Label("More", systemImage: "ellipsis.circle.fill") }
        }
        .tint(FFColor.accent)
    }

    private func stack<Root: View>(@ViewBuilder _ root: () -> Root) -> some View {
        NavigationStack {
            root()
                .navigationDestination(for: PlayerRoute.self) { PlayerDetailView(route: $0) }
                .navigationDestination(for: WikiRoute.self) { WikiDocView(slug: $0.slug, title: $0.name) }
        }
    }
}
