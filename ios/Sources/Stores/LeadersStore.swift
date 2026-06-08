import Observation
import Foundation

/// Filter + sort state for Season Leaders. Pure client-side transforms over the
/// already-fetched rows (ports getFilteredPlayers / sortValue from app.js,
/// including the `*_err = pred − actual` sort keys and min-projected-points).
@MainActor
@Observable
final class LeadersStore {
    var position: Position?
    var week: Int?
    var team: String?
    var minPoints: Double?
    var sortKey: SortKey = .actual
    var ascending = false

    enum SortKey: String, CaseIterable, Identifiable {
        case actual, week
        case ridgePred, nnPred, attnNNPred, lgbmPred
        case ridgeErr, nnErr, attnErr, lgbmErr

        var id: String { rawValue }

        var label: String {
            switch self {
            case .actual: return "Actual"
            case .week: return "Week"
            case .ridgePred: return "Ridge"
            case .nnPred: return "NN"
            case .attnNNPred: return "Attn NN"
            case .lgbmPred: return "LGBM"
            case .ridgeErr: return "Ridge Err"
            case .nnErr: return "NN Err"
            case .attnErr: return "Attn Err"
            case .lgbmErr: return "LGBM Err"
            }
        }
    }

    func filteredSorted(_ players: [Player], search: String) -> [Player] {
        let filtered = players.filter { p in
            (position == nil || p.position == position?.rawValue)
                && (week == nil || p.week == week)
                && (team == nil || p.team == team)
                && (search.isEmpty || p.name.localizedCaseInsensitiveContains(search))
                && (minPoints == nil || (p.maxPrediction ?? -.greatestFiniteMagnitude) >= minPoints!)
        }
        return filtered.sorted(by: comparator)
    }

    private func comparator(_ a: Player, _ b: Player) -> Bool {
        // Push nil values to the bottom regardless of direction (matches app.js).
        switch (sortValue(a), sortValue(b)) {
        case (nil, _): return false
        case (_, nil): return true
        case let (x?, y?): return ascending ? x < y : x > y
        }
    }

    private func sortValue(_ p: Player) -> Double? {
        switch sortKey {
        case .actual: return p.actual
        case .week: return Double(p.week)
        case .ridgePred: return p.ridgePred
        case .nnPred: return p.nnPred
        case .attnNNPred: return p.attnNNPred
        case .lgbmPred: return p.lgbmPred
        case .ridgeErr: return Fmt.errDelta(p.ridgePred, p.actual)
        case .nnErr: return Fmt.errDelta(p.nnPred, p.actual)
        case .attnErr: return Fmt.errDelta(p.attnNNPred, p.actual)
        case .lgbmErr: return Fmt.errDelta(p.lgbmPred, p.actual)
        }
    }
}
