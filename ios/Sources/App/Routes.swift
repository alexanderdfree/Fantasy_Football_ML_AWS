import Foundation

/// Value route to a player's detail screen. `week` deep-links the breakdown
/// from Leaders; nil (Upcoming) just shows the season trend.
struct PlayerRoute: Hashable {
    let playerID: String
    var name: String? = nil
    var week: Int? = nil
}

/// Value route to a wiki document.
struct WikiRoute: Hashable {
    let slug: String
    var name: String? = nil
}
