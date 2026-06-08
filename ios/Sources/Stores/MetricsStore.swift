import Observation
import Foundation

/// Model Performance data: metrics + weekly accuracy + position details, fetched
/// concurrently. Each loads independently so one failure doesn't blank the tab.
@MainActor
@Observable
final class MetricsStore {
    private let api = APIClient.shared
    private var loadGeneration = 0

    var metrics: LoadState<MetricsResponse> = .idle
    var weekly: LoadState<WeeklyAccuracy> = .idle
    var positionDetails: LoadState<PositionDetailsResponse> = .idle

    func load(scoring: ScoringFormat) async {
        loadGeneration += 1
        let generation = loadGeneration
        metrics = .loading
        weekly = .loading
        positionDetails = .loading

        let api = self.api
        await withTaskGroup(of: MetricsLoadResult.self) { group in
            group.addTask {
                do {
                    return .metrics(try await api.get(.metrics(scoring), as: MetricsResponse.self))
                } catch {
                    return .metricsFailed(Self.message(error))
                }
            }
            group.addTask {
                do {
                    return .weekly(try await api.get(.weeklyAccuracy(scoring), as: WeeklyAccuracy.self))
                } catch {
                    return .weeklyFailed(Self.message(error))
                }
            }
            group.addTask {
                do {
                    return .positionDetails(try await api.get(.positionDetails(scoring), as: PositionDetailsResponse.self))
                } catch {
                    return .positionDetailsFailed(Self.message(error))
                }
            }

            for await result in group {
                guard generation == loadGeneration else { continue }
                switch result {
                case let .metrics(value): metrics = .loaded(value)
                case let .metricsFailed(message): metrics = .failed(message)
                case let .weekly(value): weekly = .loaded(value)
                case let .weeklyFailed(message): weekly = .failed(message)
                case let .positionDetails(value): positionDetails = .loaded(value)
                case let .positionDetailsFailed(message): positionDetails = .failed(message)
                }
            }
        }
    }

    nonisolated private static func message(_ error: Error) -> String {
        (error as? APIError)?.errorDescription ?? error.localizedDescription
    }

    private enum MetricsLoadResult: Sendable {
        case metrics(MetricsResponse)
        case metricsFailed(String)
        case weekly(WeeklyAccuracy)
        case weeklyFailed(String)
        case positionDetails(PositionDetailsResponse)
        case positionDetailsFailed(String)
    }
}
