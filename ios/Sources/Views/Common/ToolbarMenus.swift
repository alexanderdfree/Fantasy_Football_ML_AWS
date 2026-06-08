import SwiftUI

/// Toolbar scoring-format picker (PPR / Half / Standard) bound to AppState.
struct ScoringMenu: View {
    @Environment(AppState.self) private var app

    var body: some View {
        @Bindable var app = app
        Menu {
            Picker("Scoring", selection: $app.scoring) {
                ForEach(ScoringFormat.allCases) { Text($0.displayName).tag($0) }
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "slider.horizontal.3")
                Text(app.scoring.shortName)
            }
            .font(.subheadline)
        }
        .accessibilityLabel("Scoring format: \(app.scoring.displayName)")
    }
}

/// Toolbar model-display picker (All / one model) bound to AppState.
struct ModelDisplayMenu: View {
    @Environment(AppState.self) private var app

    var body: some View {
        @Bindable var app = app
        Menu {
            Picker("Model", selection: $app.selectedModel) {
                Text("All models").tag(PredictionModel?.none)
                ForEach(PredictionModel.allCases) { model in
                    Text(model.shortLabel).tag(PredictionModel?.some(model))
                }
            }
        } label: {
            Image(systemName: "line.3.horizontal.decrease.circle")
        }
        .accessibilityLabel("Model display: \(app.selectedModel?.shortLabel ?? "All models")")
    }
}
