import SwiftUI

// Hex initializer so the whole palette can be defined in code (the app is
// dark-only, so an asset catalog with Any/Dark pairs would be redundant).
extension Color {
    init(hex: UInt, alpha: Double = 1) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255,
            opacity: alpha
        )
    }
}

/// Design tokens — a 1:1 port of the web frontend's CSS custom properties
/// (src/serving/static/css/style.css) so the iOS app matches the dashboard.
enum FFColor {
    // Backgrounds
    static let bgPrimary = Color(hex: 0x0F1117)
    static let bgSecondary = Color(hex: 0x1A1D27)
    static let bgCard = Color(hex: 0x21242F)
    static let bgHover = Color(hex: 0x282C3A)
    static let border = Color(hex: 0x2E3347)
    // Text
    static let textPrimary = Color(hex: 0xE8EAED)
    static let textSecondary = Color(hex: 0x9AA0B0)
    static let textMuted = Color(hex: 0x6B7280)
    // Semantic accents
    static let accent = Color(hex: 0x22C55E) // green
    static let accentDim = Color(hex: 0x22C55E, alpha: 0.15)
    static let accentSecondary = Color(hex: 0x3B82F6) // blue
    static let red = Color(hex: 0xEF4444)
    static let redDim = Color(hex: 0xEF4444, alpha: 0.15)
    static let yellow = Color(hex: 0xEAB308)
    static let yellowDim = Color(hex: 0xEAB308, alpha: 0.15)
    // Model series colors (match COLORS in app.js)
    static let modelRidge = Color(hex: 0x3B82F6)
    static let modelNN = Color(hex: 0x22C55E)
    static let modelAttnNN = Color(hex: 0xA855F7)
    static let modelLGBM = Color(hex: 0xF59E0B)
    static let seriesActual = Color(hex: 0xE8EAED)
}

enum FFRadius {
    static let sm: CGFloat = 8
    static let lg: CGFloat = 12
}

enum FFSpacing {
    static let xs: CGFloat = 4
    static let sm: CGFloat = 8
    static let md: CGFloat = 12
    static let lg: CGFloat = 16
    static let xl: CGFloat = 24
}
