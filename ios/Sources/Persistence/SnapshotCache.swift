import Foundation

/// Persists the last good `/api/snapshot` bytes to Application Support so a cold
/// launch (incl. offline) can paint instantly before the network resolves.
struct SnapshotCache {
    private var fileURL: URL? {
        let fm = FileManager.default
        guard let dir = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else { return nil }
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("last_snapshot.json")
    }

    func load() -> SnapshotResponse? {
        guard let url = fileURL, let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(SnapshotResponse.self, from: data)
    }

    /// Save the already-validated raw bytes (cheaper than re-encoding).
    func save(_ data: Data) {
        guard let url = fileURL else { return }
        try? data.write(to: url, options: .atomic)
    }
}
