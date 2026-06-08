import Observation
import Foundation

/// Season Leaders data source. Snapshot-first: paint from the disk cache, fetch
/// `/api/snapshot` (all three scoring formats), and do all filtering/sorting
/// client-side (ports app.js). Falls back to live `/api/predictions` per scoring
/// when the snapshot is absent (404).
@MainActor
@Observable
final class SnapshotStore {
    private let api = APIClient.shared
    private let cache = SnapshotCache()
    private let decoder = JSONDecoder()

    var isLoading = false
    var errorMessage: String?
    var usingSnapshot = false
    var weeks: [Int] = []
    var teams: [String] = []
    var degradedPositions: [String] = []

    private var snapshot: SnapshotResponse?
    private var liveCache: [String: [Player]] = [:]

    var hasData: Bool { snapshot != nil || !liveCache.isEmpty }

    func players(_ scoring: ScoringFormat) -> [Player] {
        snapshot?.players(scoring) ?? liveCache[scoring.rawValue] ?? []
    }

    func hydrate(scoring: ScoringFormat) async {
        if !hasData { isLoading = true }
        if snapshot == nil, let cached = cache.load() { applySnapshot(cached) } // offline paint
        do {
            let data = try await api.rawData(.snapshot)
            let snap = try decoder.decode(SnapshotResponse.self, from: data)
            cache.save(data)
            usingSnapshot = true
            applySnapshot(snap)
            errorMessage = nil
        } catch let error as APIError where error.isNotFound {
            await loadLive(scoring)
        } catch {
            if !hasData { errorMessage = (error as? APIError)?.errorDescription ?? error.localizedDescription }
        }
        isLoading = false
    }

    /// Live-mode only: ensure the active scoring format's rows are loaded
    /// (snapshot mode already holds all three).
    func ensureLive(_ scoring: ScoringFormat) async {
        guard snapshot == nil, liveCache[scoring.rawValue] == nil else { return }
        await loadLive(scoring)
    }

    private func applySnapshot(_ snap: SnapshotResponse) {
        snapshot = snap
        weeks = snap.weeks
        teams = snap.teams
        degradedPositions = snap.degradedPositions
    }

    private func loadLive(_ scoring: ScoringFormat) async {
        usingSnapshot = false
        do {
            let resp = try await api.get(
                .predictions(position: "ALL", week: "ALL", search: "", sort: "actual", order: "desc", scoring: scoring),
                as: PredictionsResponse.self
            )
            liveCache[scoring.rawValue] = resp.players
            degradedPositions = resp.degradedPositions
            if weeks.isEmpty { weeks = Set(resp.players.map(\.week)).sorted() }
            if teams.isEmpty { teams = Set(resp.players.map(\.team).filter { !$0.isEmpty }).sorted() }
            errorMessage = nil
        } catch {
            if !hasData { errorMessage = (error as? APIError)?.errorDescription ?? error.localizedDescription }
        }
    }
}
