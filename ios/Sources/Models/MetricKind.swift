import Foundation

/// The accuracy metric selected by the Comparison / History toggles and used by
/// the Performance charts. Lower is better for MAE/RMSE; higher for R².
enum MetricKind: String, CaseIterable, Identifiable, Sendable {
    case mae
    case rmse
    case r2

    var id: String { rawValue }

    var label: String {
        switch self {
        case .mae: return "MAE"
        case .rmse: return "RMSE"
        case .r2: return "R²"
        }
    }

    var higherIsBetter: Bool { self == .r2 }

    var hint: String {
        switch self {
        case .mae: return "Mean absolute error — lower is better"
        case .rmse: return "Root mean squared error — lower is better"
        case .r2: return "R² (coefficient of determination) — higher is better"
        }
    }

    func format(_ value: Double) -> String {
        String(format: self == .r2 ? "%.3f" : "%.2f", value)
    }

    /// Best value among a set (min for MAE/RMSE, max for R²).
    func best(of values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        return higherIsBetter ? values.max() : values.min()
    }
}
