import SwiftUI
import UIKit
import WebKit

/// Renders a server-rendered wiki doc. Intra-wiki `#wiki:slug` links swap the
/// content in place (re-fetch); external links open in the system browser.
struct WikiDocView: View {
    let slug: String
    var title: String?

    @State private var currentSlug: String
    @State private var state: LoadState<WikiDoc> = .idle

    init(slug: String, title: String? = nil) {
        self.slug = slug
        self.title = title
        _currentSlug = State(initialValue: slug)
    }

    var body: some View {
        Group {
            switch state {
            case .idle, .loading:
                ProgressView().tint(FFColor.accent).frame(maxWidth: .infinity, maxHeight: .infinity)
            case let .loaded(doc):
                WikiHTMLView(html: doc.html) { newSlug in
                    currentSlug = newSlug
                    Task { await load() }
                }
            case let .failed(message):
                EmptyStateView(icon: "doc.questionmark", title: "Couldn't load doc", message: message,
                               retry: { Task { await load() } })
            }
        }
        .background(FFColor.bgPrimary)
        .navigationTitle(state.value?.name ?? title ?? "Doc")
        .navigationBarTitleDisplayMode(.inline)
        .task { await load() }
    }

    private func load() async {
        state = .loading
        do {
            state = .loaded(try await APIClient.shared.get(.wikiDoc(slug: currentSlug), as: WikiDoc.self))
        } catch {
            state = .failed((error as? APIError)?.errorDescription ?? error.localizedDescription)
        }
    }
}

/// WKWebView wrapper: dark-themed, intercepts intra-wiki + external links.
struct WikiHTMLView: UIViewRepresentable {
    let html: String
    var onWikiLink: (String) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onWikiLink: onWikiLink) }

    func makeUIView(context: Context) -> WKWebView {
        let webView = WKWebView()
        webView.navigationDelegate = context.coordinator
        webView.scrollView.backgroundColor = .clear
        webView.backgroundColor = .clear
        webView.isOpaque = false
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        context.coordinator.onWikiLink = onWikiLink
        if context.coordinator.loadedHTML != html {
            context.coordinator.loadedHTML = html
            webView.loadHTMLString(Self.wrap(html), baseURL: AppConfig.baseURL)
        }
    }

    private static func wrap(_ body: String) -> String {
        """
        <!doctype html><html><head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        \(css)
        </head><body>\(body)</body></html>
        """
    }

    private static let css = """
    <style>
      body { font: -apple-system-body, -apple-system, system-ui, sans-serif; background: transparent;
             color: #e8eaed; padding: 4px 2px 32px; font-size: 15px; line-height: 1.6;
             -webkit-text-size-adjust: 100%; }
      a { color: #22c55e; text-decoration: none; }
      h1, h2, h3, h4 { color: #e8eaed; line-height: 1.3; }
      code, pre { font-family: ui-monospace, Menlo, monospace; background: #21242f; }
      code { padding: 1px 5px; border-radius: 5px; font-size: 0.88em; }
      pre { padding: 12px; border-radius: 8px; overflow: auto; border: 1px solid #2e3347; }
      pre code { padding: 0; background: transparent; }
      table { border-collapse: collapse; width: 100%; display: block; overflow-x: auto; font-size: 0.9em; }
      th, td { border: 1px solid #2e3347; padding: 6px 8px; text-align: left; }
      th { background: #1a1d27; }
      blockquote { border-left: 3px solid #2e3347; margin: 8px 0; padding: 2px 12px; color: #9aa0b0; }
      img { max-width: 100%; height: auto; }
      hr { border: none; border-top: 1px solid #2e3347; }
    </style>
    """

    final class Coordinator: NSObject, WKNavigationDelegate {
        var onWikiLink: (String) -> Void
        var loadedHTML: String?

        init(onWikiLink: @escaping (String) -> Void) { self.onWikiLink = onWikiLink }

        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationAction: WKNavigationAction,
            decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
        ) {
            guard let url = navigationAction.request.url else { return decisionHandler(.allow) }
            let string = url.absoluteString
            if let range = string.range(of: "#wiki:") {
                let rest = String(string[range.upperBound...])
                let slug = rest.split(separator: ":").first.map(String.init) ?? rest
                if !slug.isEmpty { onWikiLink(slug) }
                return decisionHandler(.cancel)
            }
            if navigationAction.navigationType == .linkActivated,
               url.scheme == "http" || url.scheme == "https" {
                UIApplication.shared.open(url)
                return decisionHandler(.cancel)
            }
            decisionHandler(.allow)
        }
    }
}
