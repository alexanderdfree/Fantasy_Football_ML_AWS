import Foundation

/// API base URL. Defaults to production (`fantasy.alexfree.me`) so a fresh
/// build works against live data with zero setup. Override for local dev by
/// setting `API_BASE_URL` in Resources/Info.plist to e.g.
/// `http://127.0.0.1:5050` (host root, no `/api`) and running the Flask app.
///
/// NOTE: the production host is the `fantasy.` subdomain — `alexfree.me` (apex)
/// is the owner's portfolio site and 404s on the API routes.
enum AppConfig {
    static let baseURL: URL = {
        if let s = Bundle.main.object(forInfoDictionaryKey: "API_BASE_URL") as? String,
           !s.isEmpty, let u = URL(string: s) {
            return u
        }
        return URL(string: "https://fantasy.alexfree.me")!
    }()
}
