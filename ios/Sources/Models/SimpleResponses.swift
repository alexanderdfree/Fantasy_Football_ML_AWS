import Foundation

struct WeeksResponse: Codable, Sendable {
    let weeks: [Int]
}

struct TeamsResponse: Codable, Sendable {
    let teams: [String]
}
