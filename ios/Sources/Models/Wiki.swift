import Foundation

/// `/api/wiki/index` entry.
struct WikiIndexEntry: Codable, Sendable, Identifiable, Hashable {
    let slug: String
    let name: String
    let group: String
    var id: String { slug }
}

/// `/api/wiki/<slug>` — server-rendered HTML body.
struct WikiDoc: Codable, Sendable {
    let slug: String
    let name: String
    let group: String
    let html: String
}
