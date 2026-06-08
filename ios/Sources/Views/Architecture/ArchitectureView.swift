import SwiftUI

/// Model Architecture — overview, training loop, and a per-position spec sheet
/// (config key/values + feature accordions). Ports ARCH_CATEGORY_LABELS.
struct ArchitectureView: View {
    @State private var store = ArchitectureStore()

    private static let categories: [(key: String, label: String)] = [
        ("specific", "Position-specific"),
        ("rolling", "Rolling windows (L3 / L5 / L8)"),
        ("prior_season", "Prior season"),
        ("ewma", "EWMA"),
        ("trend", "Trend"),
        ("share", "Share / HHI"),
        ("matchup", "Matchup vs opponent"),
        ("defense", "Opponent defense"),
        ("contextual", "Contextual"),
        ("weather_vegas", "Weather / Vegas"),
        ("attention_history", "Attention history (per-game inputs)"),
        ("other", "Other"),
    ]

    var body: some View {
        ScrollView {
            LoadStateView(state: store.state, retry: { Task { await store.load() } }) { arch in
                VStack(spacing: FFSpacing.lg) {
                    overviewCard(arch.overview)
                    trainingCard(arch.trainingLoop)
                    ForEach(Position.displayOrder) { pos in
                        if let config = arch.positions[pos.rawValue] {
                            positionCard(pos, config)
                        }
                    }
                }
                .padding(FFSpacing.lg)
            }
        }
        .background(FFColor.bgPrimary)
        .navigationTitle("Architecture")
        .navigationBarTitleDisplayMode(.inline)
        .task { await store.load() }
    }

    private func overviewCard(_ o: ModelArchitecture.Overview) -> some View {
        Card {
            VStack(alignment: .leading, spacing: FFSpacing.sm) {
                SectionLabel(text: "Overview")
                kv("Framework", o.framework)
                kv("Device", o.device)
                kv("Data splits", o.dataSplits)
                Text("Ensemble").font(.caption2.weight(.semibold)).foregroundStyle(FFColor.textMuted).padding(.top, 2)
                ForEach(o.ensemble, id: \.self) { item in
                    Text("• \(item)").font(.caption).foregroundStyle(FFColor.textSecondary)
                }
            }
        }
    }

    private func trainingCard(_ t: ModelArchitecture.TrainingLoop) -> some View {
        Card {
            VStack(alignment: .leading, spacing: FFSpacing.sm) {
                SectionLabel(text: "Training Loop")
                kv("Optimizer", t.optimizer)
                kv("Loss", t.loss)
                kv("Gradient clip", t.gradientClip)
                kv("Feature scaling", t.featureScaling)
                kv("Early stopping", t.earlyStopping)
                kv("Checkpoint", t.checkpoint)
            }
        }
    }

    private func positionCard(_ pos: Position, _ p: ModelArchitecture.PositionArch) -> some View {
        Card {
            DisclosureGroup {
                VStack(alignment: .leading, spacing: FFSpacing.md) {
                    Grid(alignment: .leading, horizontalSpacing: FFSpacing.md, verticalSpacing: 6) {
                        configRow("Targets", p.targets.joined(separator: ", "))
                        configRow("Backbone", "[" + p.backboneLayers.map(String.init).joined(separator: ", ") + "]")
                        configRow("Head hidden", p.headHidden.map(String.init) ?? "—")
                        configRow("Dropout", Self.num(p.dropout))
                        configRow("Learning rate", Self.num(p.lr))
                        configRow("Weight decay", Self.num(p.weightDecay))
                        configRow("Batch size", p.batchSize.map(String.init) ?? "—")
                        configRow("Epochs", p.epochs.map(String.init) ?? "—")
                        configRow("Patience", p.patience.map(String.init) ?? "—")
                        configRow("Scheduler", p.scheduler)
                        configRow("Attention", p.attentionEnabled ? "Enabled" : "—")
                        configRow("LightGBM", p.lightgbmEnabled ? "Enabled" : "—")
                    }
                    SectionLabel(text: "Features by category")
                    ForEach(Self.categories, id: \.key) { cat in
                        if let feats = p.features[cat.key], !feats.isEmpty {
                            DisclosureGroup {
                                FeatureChips(features: feats)
                            } label: {
                                Text("\(cat.label) (\(feats.count))").font(.caption).foregroundStyle(FFColor.textSecondary)
                            }
                        }
                    }
                }
                .padding(.top, FFSpacing.sm)
            } label: {
                HStack {
                    PositionBadge(position: pos.rawValue)
                    Text(pos.fullName).font(.subheadline.weight(.semibold)).foregroundStyle(FFColor.textPrimary)
                    Spacer()
                    Text("\(p.featureCount) feats").font(.caption2).foregroundStyle(FFColor.textMuted)
                }
            }
        }
    }

    private func kv(_ key: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(key).font(.caption2.weight(.semibold)).foregroundStyle(FFColor.textMuted)
            Text(value).font(.caption).foregroundStyle(FFColor.textSecondary)
        }
    }

    private func configRow(_ key: String, _ value: String) -> some View {
        GridRow {
            Text(key).font(.caption).foregroundStyle(FFColor.textMuted)
            Text(value).font(.caption).foregroundStyle(FFColor.textPrimary)
        }
    }

    static func num(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: "%g", value)
    }
}
