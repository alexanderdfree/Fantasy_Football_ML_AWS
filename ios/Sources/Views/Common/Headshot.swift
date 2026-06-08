import SwiftUI

/// Circular player headshot via AsyncImage (cached by the shared URLCache).
/// Requests a small face-cropped variant (ports `sizedHeadshot`); DST is a team
/// unit with no photo, so it renders a shield monogram instead.
struct Headshot: View {
    let url: String
    var position: String? = nil
    var size: CGFloat = 36

    var body: some View {
        Group {
            if position == "DST" {
                Circle()
                    .fill(FFColor.bgSecondary)
                    .overlay(Image(systemName: "shield.lefthalf.filled").foregroundStyle(FFColor.textMuted))
            } else if url.isEmpty {
                Circle().fill(FFColor.bgSecondary)
            } else {
                AsyncImage(url: URL(string: Self.sized(url, Int(size * 3)))) { phase in
                    switch phase {
                    case let .success(image):
                        image.resizable().scaledToFill()
                    default:
                        Circle().fill(FFColor.bgSecondary)
                    }
                }
            }
        }
        .frame(width: size, height: size)
        .clipShape(Circle())
        .overlay(Circle().strokeBorder(FFColor.border))
        .accessibilityHidden(true)
    }

    /// Resize at the CDN (NFL Cloudinary transform / ESPN combiner) so we fetch
    /// a ~size variant instead of the full-resolution source.
    static func sized(_ url: String, _ px: Int) -> String {
        if url.contains("static.www.nfl.com/image/"), url.contains("/f_auto,q_auto/") {
            return url.replacingOccurrences(
                of: "/f_auto,q_auto/",
                with: "/f_auto,q_auto,w_\(px),h_\(px),c_fill,g_face/"
            )
        }
        if url.hasPrefix("https://a.espncdn.com"), let r = url.range(of: "/i/headshots/") {
            let path = String(url[r.lowerBound...])
            return "https://a.espncdn.com/combiner/i?img=\(path)&w=\(px)&h=\(px)"
        }
        return url
    }
}
