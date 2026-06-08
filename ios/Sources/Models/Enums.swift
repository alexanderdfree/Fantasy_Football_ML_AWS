import SwiftUI

/// PPR / Half-PPR / Standard. `rawValue` matches the API `scoring` query param
/// and the per-format snapshot keys ("ppr" / "half_ppr" / "standard").
enum ScoringFormat: String, CaseIterable, Codable, Sendable, Identifiable {
    case ppr
    case halfPPR = "half_ppr"
    case standard

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .ppr: return "Full PPR"
        case .halfPPR: return "Half PPR"
        case .standard: return "Standard"
        }
    }

    var shortName: String {
        switch self {
        case .ppr: return "PPR"
        case .halfPPR: return "½ PPR"
        case .standard: return "Std"
        }
    }
}

/// The six NFL fantasy positions. `rawValue` matches the API `position` values.
enum Position: String, CaseIterable, Codable, Sendable, Identifiable {
    case qb = "QB"
    case rb = "RB"
    case wr = "WR"
    case te = "TE"
    case k = "K"
    case dst = "DST"

    var id: String { rawValue }

    /// Canonical display order used across the UI and the web frontend.
    static let displayOrder: [Position] = [.qb, .rb, .wr, .te, .k, .dst]

    /// Skill positions are the only ones with live upcoming-week projections.
    static let skill: Set<Position> = [.qb, .rb, .wr, .te]

    var fullName: String {
        switch self {
        case .qb: return "Quarterback"
        case .rb: return "Running Back"
        case .wr: return "Wide Receiver"
        case .te: return "Tight End"
        case .k: return "Kicker"
        case .dst: return "Defense/Special Teams"
        }
    }

    /// Position-badge foreground color (rendered on a 15%-opacity tint of itself).
    var color: Color {
        switch self {
        case .qb: return Color(hex: 0xF87171)
        case .rb: return Color(hex: 0x4ADE80)
        case .wr: return Color(hex: 0x60A5FA)
        case .te: return Color(hex: 0xFACC15)
        case .k: return Color(hex: 0xC084FC)
        case .dst: return Color(hex: 0xFB923C)
        }
    }
}

/// The four prediction models shown side by side. This type centralizes the
/// THREE key conventions the API uses for the same model:
///   - predictions / player:   ridge_pred / nn_pred / attn_nn_pred / lgbm_pred
///   - breakdown / comparison:  ridge / nn / attn_nn / lgbm   (== rawValue)
///   - metrics:                 "Ridge Regression" / "Neural Network" / …
///   - position_details:        ridge_mae / nn_mae / attn_nn_mae / lgbm_mae
enum PredictionModel: String, CaseIterable, Codable, Sendable, Identifiable {
    case ridge
    case nn
    case attnNN = "attn_nn"
    case lgbm

    var id: String { rawValue }

    var bareKey: String { rawValue } // ridge / nn / attn_nn / lgbm
    var predKey: String { rawValue + "_pred" } // ridge_pred … attn_nn_pred
    var maeKey: String { rawValue + "_mae" } // ridge_mae … attn_nn_mae

    var metricsDisplayName: String {
        switch self {
        case .ridge: return "Ridge Regression"
        case .nn: return "Neural Network"
        case .attnNN: return "Attention NN"
        case .lgbm: return "LightGBM"
        }
    }

    var shortLabel: String {
        switch self {
        case .ridge: return "Ridge"
        case .nn: return "NN"
        case .attnNN: return "Attn NN"
        case .lgbm: return "LGBM"
        }
    }

    var color: Color {
        switch self {
        case .ridge: return FFColor.modelRidge
        case .nn: return FFColor.modelNN
        case .attnNN: return FFColor.modelAttnNN
        case .lgbm: return FFColor.modelLGBM
        }
    }
}
