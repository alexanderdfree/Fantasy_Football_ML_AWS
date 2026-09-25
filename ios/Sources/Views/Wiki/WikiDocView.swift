import SwiftUI
import UIKit
import WebKit

/// Renders a server-rendered wiki doc. Intra-wiki `#wiki:slug` links swap the
/// content in place (re-fetch); external links open in the system browser.
struct WikiDocView: View {
    let slug: String
    var title: String?

    @State private var currentSlug: String
    @State private var currentAnchor: String?
    @State private var loadGeneration = 0
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
                WikiHTMLView(html: doc.html, slug: doc.slug, anchor: currentAnchor) { newSlug, anchor in
                    currentAnchor = anchor
                    currentSlug = newSlug
                }
            case let .failed(message):
                EmptyStateView(icon: "doc.questionmark", title: "Couldn't load doc", message: message,
                               retry: { Task { await load(slug: currentSlug) } })
            }
        }
        .background(FFColor.bgPrimary)
        .navigationTitle(state.value?.name ?? title ?? "Doc")
        .navigationBarTitleDisplayMode(.inline)
        .task(id: currentSlug) { await load(slug: currentSlug) }
    }

    private func load(slug: String) async {
        guard !Task.isCancelled else { return }
        loadGeneration += 1
        let generation = loadGeneration
        state = .loading
        do {
            let doc = try await APIClient.shared.get(.wikiDoc(slug: slug), as: WikiDoc.self)
            guard generation == loadGeneration, !Task.isCancelled, slug == currentSlug else { return }
            state = .loaded(doc)
        } catch {
            guard generation == loadGeneration, !Task.isCancelled, slug == currentSlug else { return }
            state = .failed((error as? APIError)?.errorDescription ?? error.localizedDescription)
        }
    }
}

/// WKWebView wrapper: dark-themed, intercepts intra-wiki + external links.
struct WikiHTMLView: UIViewRepresentable {
    let html: String
    let slug: String
    var anchor: String?
    var onWikiLink: (String, String?) -> Void

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
        context.coordinator.slug = slug
        if context.coordinator.loadedHTML != html {
            context.coordinator.loadedHTML = html
            context.coordinator.pendingAnchor = anchor
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
        var onWikiLink: (String, String?) -> Void
        var loadedHTML: String?
        var slug: String?
        var pendingAnchor: String?

        init(onWikiLink: @escaping (String, String?) -> Void) { self.onWikiLink = onWikiLink }

        struct Destination: Equatable {
            let slug: String
            let anchor: String?
        }

        static func wikiDestination(_ url: URL) -> Destination? {
            guard let fragment = URLComponents(url: url, resolvingAgainstBaseURL: false)?.fragment,
                  fragment.hasPrefix("wiki:") else { return nil }
            let parts = fragment.dropFirst(5).split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
            guard let slug = parts.first, !slug.isEmpty else { return nil }
            return Destination(slug: String(slug), anchor: parts.count == 2 ? String(parts[1]) : nil)
        }

        static func isDocumentFragment(_ url: URL, baseURL: URL) -> Bool {
            guard url.fragment != nil,
                  var target = URLComponents(url: url, resolvingAgainstBaseURL: false),
                  var base = URLComponents(url: baseURL, resolvingAgainstBaseURL: false) else { return false }
            target.fragment = nil
            base.fragment = nil
            if target.path.isEmpty { target.path = "/" }
            if base.path.isEmpty { base.path = "/" }
            return target.url == base.url
        }

        private func scrollToAnchor(_ anchor: String, in webView: WKWebView) {
            // JSON encoding keeps quotes/Unicode in a Markdown heading ID as
            // data. Same-document links use native WebKit scrolling below.
            guard let encoded = try? JSONEncoder().encode(anchor),
                  let literal = String(data: encoded, encoding: .utf8) else { return }
            let script = anchor.isEmpty ? "window.scrollTo(0, 0)"
                : "document.getElementById(\(literal))?.scrollIntoView()"
            webView.evaluateJavaScript(script)
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            if let anchor = pendingAnchor {
                pendingAnchor = nil
                scrollToAnchor(anchor, in: webView)
            }
        }

        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationAction: WKNavigationAction,
            decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
        ) {
            guard let url = navigationAction.request.url else { return decisionHandler(.allow) }
            if Self.isDocumentFragment(url, baseURL: AppConfig.baseURL) {
                if let destination = Self.wikiDestination(url) {
                    if destination.slug == slug {
                        scrollToAnchor(destination.anchor ?? "", in: webView)
                    } else {
                        onWikiLink(destination.slug, destination.anchor)
                    }
                    return decisionHandler(.cancel)
                }
                // The server preserves #heading TOC links for client scrolling.
                return decisionHandler(.allow)
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
