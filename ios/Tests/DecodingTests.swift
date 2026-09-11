import XCTest
@testable import FFPredictor

/// Decodes models against captured and locally generated API fixtures.
/// This is the one layer testable without a UI build — it catches snake_case,
/// nullable, and keyed-dictionary mistakes in the Codable layer.
final class DecodingTests: XCTestCase {
    private struct SharedClientFixture: Decodable {
        let architecture: ModelArchitecture
        let snapshot: SnapshotResponse
        let predictions: [String: PredictionsResponse]
        let player: [String: PlayerDetail]
    }
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

    func testSharedServerFixturesPreserveScoringAndMissingValues() throws {
        let fixture = try decode(SharedClientFixture.self, "client_contract")
        XCTAssertEqual(fixture.architecture.positions["QB"]?.backboneLayers, [7])
        XCTAssertEqual(fixture.architecture.positions["QB"]?.metadataSource, "bundle")
        XCTAssertEqual(fixture.architecture.positions["TE"]?.metadataSource, "config_fallback")
        for scoring in ScoringFormat.allCases {
            let players = fixture.snapshot.players(scoring)
            XCTAssertEqual(Set(players.map(\.position)), Set(Position.allCases.map(\.rawValue)))
            XCTAssertEqual(fixture.predictions[scoring.rawValue]?.players.count, 6)
            XCTAssertEqual(players.first(where: { $0.position == "K" })?.ridgePred, 0)
            XCTAssertNil(players.first(where: { $0.position == "TE" })?.ridgePred)
            XCTAssertEqual(fixture.player[scoring.rawValue]?.weekly.first?.actual,
                           players.first(where: { $0.position == "QB" })?.actual)
        }
        XCTAssertEqual(fixture.snapshot.players(.ppr).first?.actual, 30)
        XCTAssertEqual(fixture.snapshot.players(.standard).first?.actual, 10)
        XCTAssertEqual(fixture.snapshot.degradedPositions, ["TE"])
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
        let expert = comparison.expertReliability?.positions["QB"]?["nflcom"] ?? nil
        XCTAssertNotNil(expert?.perSeason?["2025"]?.sigma)
        let unavailableExpert = comparison.expertReliability?.positions["DST"]?["nflcom"] ?? nil
        XCTAssertNil(unavailableExpert)
        XCTAssertNil(comparison.sampleBasis)
        XCTAssertNil(comparison.coverage)
        XCTAssertTrue(comparison.sampleBasisDescription.contains("does not specify"))
        XCTAssertTrue(comparison.actualBasisDescription.contains("does not specify"))
        XCTAssertFalse(comparison.displayedSubsets.contains("weekly_reference_top24"))
    }

    func testCurrentComparisonContractAcrossAllPositions() throws {
        let comparison = try decode(Comparison.self, "comparison_current")
        XCTAssertEqual(comparison.sampleBasis, "shared_player_weeks")
        XCTAssertEqual(comparison.actualBasis, "shared_projected_components_v2")
        XCTAssertNotNil(comparison.excludedComponents?["DST"]?["points_allowed"])
        XCTAssertEqual(comparison.displayedSubsets, ["weekly_reference_top24", "all", "top30", "top12"])
        XCTAssertNil(comparison.intervals)
        XCTAssertNil(comparison.expertReliability)
        for position in Position.displayOrder {
            XCTAssertFalse(comparison.scoringComponents?[position.rawValue]?.isEmpty ?? true)
            for subset in comparison.displayedSubsets {
                let coverage = try XCTUnwrap(comparison.coverage?[subset]?[position.rawValue])
                let sources = comparison.sourceKeys(subset: subset, position: position.rawValue)
                XCTAssertTrue(sources.contains("espn"))
                for source in sources {
                    if let cell = comparison.cell(subset: subset, position: position.rawValue, source: source) {
                        XCTAssertEqual(cell.n, coverage.n, "Every source must use the shared sample")
                    }
                }
            }
        }
        XCTAssertEqual(comparison.cohortDefinitions?["weekly_reference_top24"],
                       "Top 24 per week by shared-component NFL.com/RotoWire mean; ESPN for K, RotoWire for DST")
        XCTAssertEqual(comparison.subsetTitle("top12"), "Season leaders · top 12")
        XCTAssertEqual(CmpSource.resolve("espn", comparison: comparison).label, "ESPN")
    }

    func testCurrentComparisonMissingCoverageAndZeroAreDistinct() throws {
        let comparison = try decode(Comparison.self, "comparison_current")
        XCTAssertEqual(comparison.cell(subset: "all", position: "K", source: "espn")?.mae, 0)
        XCTAssertNil(comparison.cell(subset: "all", position: "K", source: "nflcom"))
        XCTAssertNotNil(comparison.exclusionReason(subset: "all", position: "K", source: "nflcom"))
        let rb = try XCTUnwrap(comparison.coverage?["all"]?["RB"])
        XCTAssertEqual(rb.n, 5)
        XCTAssertEqual(rb.cohortN, 6)
        XCTAssertEqual(rb.sourceN?["espn"], 5)
        XCTAssertEqual(rb.sourceN?["ridge"], 6)
        let wr = try XCTUnwrap(comparison.coverage?["weekly_reference_top24"]?["WR"])
        XCTAssertEqual(wr.status, "partial")
        XCTAssertEqual(wr.missingReferenceWeeks, 1)
        XCTAssertTrue(wr.summary.contains("partial reference"))
        let dst = try XCTUnwrap(comparison.coverage?["weekly_reference_top24"]?["DST"])
        XCTAssertEqual(dst.status, "unavailable")
        XCTAssertEqual(dst.reason, "no_reference_for_evaluation_seasons")
        XCTAssertTrue(dst.summary.hasPrefix("Unavailable"))
        XCTAssertEqual(comparison.coverage?["all"]?["TE"]?.reason, "shared_actual_components_missing")
    }

    func testComparisonDoesNotInferNewSourceOrBasisMeaning() throws {
        var payload = try XCTUnwrap(JSONSerialization.jsonObject(with: data("comparison_current")) as? [String: Any])
        payload["sample_basis"] = "future_sample_basis"
        payload["actual_basis"] = "future_actual_basis"
        payload["subsets"] = ["all": ["QB": ["future_source": ["mae": 0.0, "n": 3]]]]
        payload.removeValue(forKey: "coverage")
        payload.removeValue(forKey: "excluded_sources")
        let comparison = try JSONDecoder().decode(Comparison.self, from: JSONSerialization.data(withJSONObject: payload))
        XCTAssertEqual(comparison.sourceKeys(subset: "all", position: "QB"), ["future_source"])
        XCTAssertTrue(comparison.sampleBasisDescription.contains("future_sample_basis"))
        XCTAssertTrue(comparison.actualBasisDescription.contains("future_actual_basis"))
        XCTAssertFalse(comparison.sampleBasisDescription.contains("same regular-season"))
        XCTAssertEqual(CmpSource.resolve("future_source", comparison: comparison).label, "future_source")
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
