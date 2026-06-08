import SwiftUI

/// "More" hub — the lower-frequency, desktop-dense sections plus an about/legend.
struct MoreView: View {
    @Environment(AppState.self) private var app

    var body: some View {
        List {
            Section("Explore") {
                NavigationLink { ArchitectureView() } label: {
                    Label("Model Architecture", systemImage: "cpu")
                }
                NavigationLink { HistoryView() } label: {
                    Label("Training History", systemImage: "clock.arrow.circlepath")
                }
                NavigationLink { WikiIndexView() } label: {
                    Label("Docs & Wiki", systemImage: "book")
                }
            }
            .listRowBackground(FFColor.bgSecondary)

            Section("Models") {
                ForEach(PredictionModel.allCases) { model in
                    HStack {
                        ModelChip(model: model, showLabel: false)
                        Text(model.metricsDisplayName)
                        Spacer()
                        Text(model.shortLabel).foregroundStyle(FFColor.textMuted)
                    }
                }
                HStack {
                    Circle().fill(FFColor.seriesActual).frame(width: 8, height: 8)
                    Text("Actual (realized fantasy points)")
                    Spacer()
                }
            }
            .listRowBackground(FFColor.bgSecondary)

            Section("About") {
                VStack(alignment: .leading, spacing: FFSpacing.sm) {
                    Text("FF Predictor")
                        .font(.headline)
                    Text("Per-position machine-learning projections of weekly NFL fantasy points — a Ridge baseline, a multi-head neural net, an attention variant, and LightGBM, shown side by side (no blend). Toggle scoring (PPR / Half / Standard) from the top bar on any data screen.")
                        .font(.caption)
                        .foregroundStyle(FFColor.textSecondary)
                    Text("Data from fantasy.alexfree.me")
                        .font(.caption2)
                        .foregroundStyle(FFColor.textMuted)
                }
                .padding(.vertical, 4)
            }
            .listRowBackground(FFColor.bgSecondary)
        }
        .navigationTitle("More")
        .ffScreenBackground()
    }
}
