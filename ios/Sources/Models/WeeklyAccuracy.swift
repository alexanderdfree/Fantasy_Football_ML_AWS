import Foundation

/// `/api/weekly_accuracy?scoring=` — per-week MAE per model (elements nullable).
struct WeeklyAccuracy: Codable, Sendable {
    let weeks: [Int]
    let ridgeMAE: [Double?]
    let nnMAE: [Double?]
    let attnNNMAE: [Double?]
    let lgbmMAE: [Double?]
    let scoring: String?

    func series(for model: PredictionModel) -> [Double?] {
        switch model {
        case .ridge: return ridgeMAE
        case .nn: return nnMAE
        case .attnNN: return attnNNMAE
        case .lgbm: return lgbmMAE
        }
    }

    enum CodingKeys: String, CodingKey {
        case weeks
        case ridgeMAE = "ridge_mae"
        case nnMAE = "nn_mae"
        case attnNNMAE = "attn_nn_mae"
        case lgbmMAE = "lgbm_mae"
        case scoring
    }
}
