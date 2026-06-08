import SwiftUI

enum ChartStyle {
    static let modelColorScale: KeyValuePairs<String, Color> = [
        "Ridge": FFColor.modelRidge,
        "NN": FFColor.modelNN,
        "Attn NN": FFColor.modelAttnNN,
        "LGBM": FFColor.modelLGBM,
    ]
}
