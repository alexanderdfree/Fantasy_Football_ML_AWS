import XCTest
@testable import FFPredictor

final class WikiNavigationTests: XCTestCase {
    private typealias Coordinator = WikiHTMLView.Coordinator

    func testCanonicalCrossDocumentFragmentPreservesTheHeading() throws {
        let url = try XCTUnwrap(URL(string: "https://fantasy.alexfree.me/#wiki:architecture:3-decision-log"))
        XCTAssertEqual(Coordinator.wikiDestination(url), .init(slug: "architecture", anchor: "3-decision-log"))
        XCTAssertTrue(Coordinator.isDocumentFragment(url, baseURL: URL(string: "https://fantasy.alexfree.me")!))
    }

    func testDocumentOnlyLinkAndEncodedHeading() throws {
        let base = URL(string: "https://fantasy.alexfree.me/")!
        XCTAssertEqual(Coordinator.wikiDestination(URL(string: "#wiki:setup", relativeTo: base)!),
                       .init(slug: "setup", anchor: nil))
        XCTAssertEqual(Coordinator.wikiDestination(URL(string: "#wiki:setup:heading%20%22quoted%22:tail", relativeTo: base)!),
                       .init(slug: "setup", anchor: "heading \"quoted\":tail"))
        XCTAssertNil(Coordinator.wikiDestination(URL(string: "#wiki::heading", relativeTo: base)!))
    }

    func testPlainHeadingUsesCurrentDocumentButExternalURLsDoNot() {
        let base = URL(string: "https://fantasy.alexfree.me")!
        XCTAssertTrue(Coordinator.isDocumentFragment(URL(string: "https://fantasy.alexfree.me/#2-system-overview")!, baseURL: base))
        XCTAssertNil(Coordinator.wikiDestination(URL(string: "https://fantasy.alexfree.me/#2-system-overview")!))
        XCTAssertFalse(Coordinator.isDocumentFragment(URL(string: "https://github.com/example/repo#readme")!, baseURL: base))
        XCTAssertFalse(Coordinator.isDocumentFragment(URL(string: "https://fantasy.alexfree.me/support#contact")!, baseURL: base))
        XCTAssertFalse(Coordinator.isDocumentFragment(URL(string: "https://fantasy.alexfree.me/")!, baseURL: base))
    }
}
