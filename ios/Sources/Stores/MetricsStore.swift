import Observation
import Foundation

/// Model Performance data: metrics + weekly accuracy + position details, fetched
/// concurrently. Each loads independently so one failure doesn't blank the tab.
@MainActor
@Observable
final class MetricsStore {
    private let api = APIClient.shared

    var metrics: LoadState<MetricsResponse> = .idle
    var weekly: LoadState<WeeklyAccuracy> = .idle
    var positionDetails: LoadState<PositionDetailsResponse> = .idle

    func load(scoring: ScoringFormat) async {
        metrics = .loading
        weekly = .loading
        positionDetails = .loading

        async let m = api.get(.metrics(scoring), as: MetricsResponse.self)
        async let w = api.get(.weeklyAccuracy(scoring), as: WeeklyAccuracy.self)
        async let p = api.get(.positionDetails(scoring), as: PositionDetailsResponse.self)

        do { metrics = .loaded(try await m) } catch { metrics = .failed(message(error)) }
        do { weekly = .loaded(try await w) } catch { weekly = .failed(message(error)) }
        do { positionDetails = .loaded(try await p) } catch { positionDetails = .failed(message(error)) }
    }

    private func message(_ error: Error) -> String {
        (error as? APIError)?.errorDescription ?? error.localizedDescription
    }
}
