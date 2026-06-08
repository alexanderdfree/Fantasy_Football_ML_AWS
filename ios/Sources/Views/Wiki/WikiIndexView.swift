import SwiftUI

/// Wiki sidebar → grouped doc list (pushes WikiDocView).
struct WikiIndexView: View {
    @State private var store = WikiStore()

    var body: some View {
        LoadStateView(state: store.index, retry: { Task { await store.loadIndex() } }) { entries in
            List {
                ForEach(grouped(entries), id: \.group) { section in
                    Section(section.group) {
                        ForEach(section.items) { item in
                            NavigationLink(value: WikiRoute(slug: item.slug, name: item.name)) {
                                Text(item.name).font(.subheadline)
                            }
                        }
                    }
                    .listRowBackground(FFColor.bgSecondary)
                }
            }
            .listStyle(.insetGrouped)
            .ffScreenBackground()
        }
        .navigationTitle("Docs")
        .navigationBarTitleDisplayMode(.inline)
        .task { await store.loadIndex() }
    }

    private func grouped(_ entries: [WikiIndexEntry]) -> [(group: String, items: [WikiIndexEntry])] {
        var order: [String] = []
        var map: [String: [WikiIndexEntry]] = [:]
        for entry in entries {
            if map[entry.group] == nil { order.append(entry.group) }
            map[entry.group, default: []].append(entry)
        }
        return order.map { ($0, map[$0] ?? []) }
    }
}
