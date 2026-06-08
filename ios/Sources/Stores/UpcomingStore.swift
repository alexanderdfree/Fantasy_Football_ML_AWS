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
    private let api = APIClient.shared
    private let decoder = JSONDecoder()
    var state: UpcomingScreenState = .loading

    func load() async {
        state = .loading
        do {
            let data = try await api.rawData(.upcomingWeek)
            let week = try decoder.decode(UpcomingWeek.self, from: data)
            if week.available == false {
                state = .offseason(Self.reasonMessage(week.reason))
            } else if week.status == "warming" || week.scoring == nil {
                state = .warming
            } else {
                state = .ready(week)
            }
        } catch let error as APIError where error.isWarming {
            state = .warming
        } catch {
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
