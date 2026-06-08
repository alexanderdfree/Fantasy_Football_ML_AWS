import XCTest
@testable import FFPredictor

/// Decodes every model against real captured API payloads (Tests/Fixtures/*.json).
/// This is the one layer testable without a UI build — it catches snake_case,
/// nullable, and keyed-dictionary mistakes in the Codable layer.
final class DecodingTests: XCTestCase {
    private func data(_ name: String) throws -> Data {
        let url = try XCTUnwrap(
            Bundle(for: Self.self).url(forResource: name, withExtension: "json"),
            "missing fixture \(name).json"
        )
        return try Data(contentsOf: url)
    }

    private func decode<T: Decodable>(_ type: T.Type, _ name: String) throws -> T {
        try JSONDecoder().decode(T.self, from: data(name))
    }

    func testPredictions() throws {
        let response = try decode(PredictionsResponse.self, "predictions_qb_w1")
        XCTAssertFalse(response.players.isEmpty)
        XCTAssertEqual(response.scoring, "ppr")
        XCTAssertNotNil(response.players.first?.ridgePred)
        XCTAssertNotNil(response.players.first?.attnNNPred)
    }

    func testSnapshot() throws {
        let snapshot = try decode(SnapshotResponse.self, "snapshot")
        XCTAssertFalse(snapshot.players(.ppr).isEmpty)
        XCTAssertFalse(snapshot.players(.halfPPR).isEmpty)
        XCTAssertFalse(snapshot.weeks.isEmpty)
    }

    func testPlayerDetail() throws {
        let detail = try decode(PlayerDetail.self, "player")
        XCTAssertFalse(detail.weekly.isEmpty)
        XCTAssertNotNil(detail.seasonAvg)
        XCTAssertNotNil(detail.weekly.first?.attnNNPred)
    }

    func testBreakdown() throws {
        let breakdown = try decode(Breakdown.self, "breakdown")
        XCTAssertFalse(breakdown.components.isEmpty)
        XCTAssertNotNil(breakdown.components.first?.value(for: .attnNN))
    }

    func testMetrics() throws {
        let metrics = try decode(MetricsResponse.self, "metrics")
        XCTAssertNotNil(metrics.metrics(for: .ridge)?.overall?.mae)
        XCTAssertEqual(metrics.metrics(for: .attnNN)?.byPosition.isEmpty, false)
    }

    func testWeeklyAccuracy() throws {
        let weekly = try decode(WeeklyAccuracy.self, "weekly_accuracy")
        XCTAssertEqual(weekly.weeks.count, weekly.ridgeMAE.count)
        XCTAssertEqual(weekly.series(for: .lgbm).count, weekly.weeks.count)
    }

    func testPositionDetails() throws {
        let details = try decode(PositionDetailsResponse.self, "position_details")
        XCTAssertNotNil(details["QB"])
        XCTAssertFalse(details["QB"]?.targets.isEmpty ?? true)
        XCTAssertNotNil(details["QB"]?.targetMetrics?["total"])
        XCTAssertNotNil(details["QB"]?.targetMetrics?["passing_yards"]?.unit)
    }

    func testModelArchitecture() throws {
        let arch = try decode(ModelArchitecture.self, "model_architecture")
        XCTAssertNotNil(arch.positions["QB"])
        XCTAssertFalse(arch.overview.ensemble.isEmpty)
        XCTAssertFalse(arch.positions["QB"]?.features.isEmpty ?? true)
    }

    func testComparison() throws {
        let comparison = try decode(Comparison.self, "comparison")
        XCTAssertNotNil(comparison.cell(subset: "all", position: "QB", source: "ridge"))
        XCTAssertNotNil(comparison.cell(subset: "all", position: "QB", source: "nflcom"))
        XCTAssertNotNil(comparison.cell(subset: "top30", position: "RB", source: "lgbm"))
        XCTAssertNotNil(comparison.modelReliability(position: "QB", model: "ridge")?.sigma)
        XCTAssertNotNil(comparison.intervals)
        let expert = comparison.expertReliability?.positions["QB"]?["nflcom"]
        XCTAssertNotNil(expert?.perSeason?["2025"]?.sigma)
    }

    func testBenchmarkHistory() throws {
        let history = try decode(BenchmarkHistory.self, "benchmark_history")
        XCTAssertFalse(history.rows.isEmpty)
        XCTAssertFalse(history.targetLabels.isEmpty)
        XCTAssertFalse(history.targetUnits.isEmpty)
    }

    func testWikiIndex() throws {
        let entries = try decode([WikiIndexEntry].self, "wiki_index")
        XCTAssertFalse(entries.isEmpty)
        XCTAssertFalse(entries.first?.group.isEmpty ?? true)
    }

    func testUpcomingWarming() throws {
        // Offseason / not-yet-built returns the warming sentinel.
        let upcoming = try decode(UpcomingWeek.self, "upcoming_week")
        XCTAssertEqual(upcoming.status, "warming")
    }

    func testHealthTeamsWeeks() throws {
        XCTAssertEqual(try decode(Health.self, "health").status, "ok")
        XCTAssertFalse(try decode(TeamsResponse.self, "teams").teams.isEmpty)
        XCTAssertFalse(try decode(WeeksResponse.self, "weeks").weeks.isEmpty)
    }
}
