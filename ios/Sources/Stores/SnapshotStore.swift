import Observation
import Foundation

/// Season Leaders data source. Snapshot-first: paint from the disk cache, fetch
/// `/api/snapshot` (all three scoring formats), and do all filtering/sorting
/// client-side (ports app.js). Falls back to live `/api/predictions` per scoring
/// when the snapshot is absent (404).
@MainActor
@Observable
final class SnapshotStore {
    private let api: APIClient
    private let cache: SnapshotCache
    private let decoder = JSONDecoder()

    var isLoading = false
    var errorMessage: String?
    var usingSnapshot = false
    var weeks: [Int] = []
    var teams: [String] = []
    var degradedPositions: [String] = []

    private var snapshot: SnapshotResponse?
    private var liveCache: [String: [Player]] = [:]
    private var loadGeneration = 0
    private var liveCacheGeneration: Int?
    private var liveRequestSequence = 0
    private var latestLiveRequests: [String: Int] = [:]
    private var latestScoring: ScoringFormat = .ppr

    init(api: APIClient = .shared, cache: SnapshotCache = SnapshotCache()) {
        self.api = api
        self.cache = cache
    }

    var hasData: Bool { snapshot != nil || !liveCache.isEmpty }

    func players(_ scoring: ScoringFormat) -> [Player] {
        snapshot?.players(scoring) ?? liveCache[scoring.rawValue] ?? []
    }

    func hydrate(scoring: ScoringFormat) async {
        loadGeneration += 1
        let generation = loadGeneration
        latestScoring = scoring
        // Scoring changes must reach ensureLive while snapshot availability is
        // being resolved. Existing snapshot rows remain available for display.
        usingSnapshot = false
        defer { if generation == loadGeneration { isLoading = false } }
        if !hasData { isLoading = true }
        if !hasData, let cached = cache.load() { applySnapshot(cached) } // offline paint
        do {
            let data = try await api.rawData(.snapshot)
            guard generation == loadGeneration else { return }
            let snap = try decoder.decode(SnapshotResponse.self, from: data)
            cache.save(data)
            usingSnapshot = true
            applySnapshot(snap)
            errorMessage = nil
        } catch let error as APIError where error.isNotFound {
            guard generation == loadGeneration else { return }
            await loadLive(latestScoring, generation: generation)
        } catch {
            guard generation == loadGeneration else { return }
            if !hasData { errorMessage = (error as? APIError)?.errorDescription ?? error.localizedDescription }
        }
    }

    /// Live-mode only: ensure the active scoring format's rows are loaded
    /// (snapshot mode already holds all three).
    func ensureLive(_ scoring: ScoringFormat) async {
        latestScoring = scoring
        guard !usingSnapshot, liveCache[scoring.rawValue] == nil else { return }
        await loadLive(scoring, generation: loadGeneration)
    }

    private func applySnapshot(_ snap: SnapshotResponse) {
        snapshot = snap
        weeks = snap.weeks
        teams = snap.teams
        degradedPositions = snap.degradedPositions
    }

    private func loadLive(_ scoring: ScoringFormat, generation: Int) async {
        guard generation == loadGeneration, !usingSnapshot else { return }
        liveRequestSequence += 1
        let request = liveRequestSequence
        latestLiveRequests[scoring.rawValue] = request
        do {
            let resp = try await api.get(
                .predictions(position: "ALL", week: "ALL", search: "", sort: "actual", order: "desc", scoring: scoring),
                as: PredictionsResponse.self
            )
            guard generation == loadGeneration, !usingSnapshot,
                  latestLiveRequests[scoring.rawValue] == request else { return }
            // A successful live fallback supersedes the offline snapshot. Keep
            // the snapshot only when the fallback fails, so offline paint survives.
            snapshot = nil
            if liveCacheGeneration != generation {
                liveCache.removeAll()
                liveCacheGeneration = generation
            }
            liveCache[scoring.rawValue] = resp.players
            degradedPositions = resp.degradedPositions
            weeks = Set(resp.players.map(\.week)).sorted()
            teams = Set(resp.players.map(\.team).filter { !$0.isEmpty }).sorted()
            errorMessage = nil
        } catch {
            guard generation == loadGeneration, !usingSnapshot,
                  latestLiveRequests[scoring.rawValue] == request else { return }
            if !hasData { errorMessage = (error as? APIError)?.errorDescription ?? error.localizedDescription }
        }
    }
}
