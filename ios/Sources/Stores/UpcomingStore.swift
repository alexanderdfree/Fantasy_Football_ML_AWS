import Observation
import Foundation

enum UpcomingScreenState {
    case loading
    case warming
    case offseason(String)
    case ready(UpcomingWeek)
    case failed(String)
}

@MainActor
@Observable
final class UpcomingStore {
    private let api: APIClient
    private var loadGeneration = 0
    private let decoder = JSONDecoder()
    var state: UpcomingScreenState = .loading

    init(api: APIClient = .shared) { self.api = api }

    func load() async {
        guard !Task.isCancelled else { return }
        loadGeneration += 1
        let generation = loadGeneration
        state = .loading
        do {
            let data = try await api.rawData(.upcomingWeek)
            guard generation == loadGeneration, !Task.isCancelled else { return }
            let week = try decoder.decode(UpcomingWeek.self, from: data)
            if week.available == false {
                state = .offseason(Self.reasonMessage(week.reason))
            } else if week.status == "warming" || week.scoring == nil {
                state = .warming
            } else {
                state = .ready(week)
            }
        } catch let error as APIError where error.isWarming {
            guard generation == loadGeneration, !Task.isCancelled else { return }
            state = .warming
        } catch {
            // SwiftUI cancels this task when its tab disappears. Keep loading
            // so the view's next appearance restarts it; cancellation is not a
            // failed network request. Genuine failures still offer manual retry.
            guard generation == loadGeneration, !Task.isCancelled else { return }
            state = .failed((error as? APIError)?.errorDescription ?? error.localizedDescription)
        }
    }

    private static func reasonMessage(_ reason: String?) -> String {
        switch reason {
        case "offseason": return "Live projections resume when the next slate is posted."
        case "no_slate": return "No games are scheduled for the upcoming week yet."
        case "no_roster": return "Rosters for the upcoming week aren't available yet."
        default: return "No upcoming games scheduled right now."
        }
    }
}
