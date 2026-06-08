import Foundation

/// A minimal recursive JSON value for decoding the loosely-typed corners of the
/// API (Comparison expert cells, reliability/interval blocks) defensively
/// without modeling every nested field.
enum JSONValue: Codable, Sendable, Hashable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self = .null
        } else if let d = try? c.decode(Double.self) {
            self = .number(d) // numbers before bool so 0/1 don't become Bool
        } else if let b = try? c.decode(Bool.self) {
            self = .bool(b)
        } else if let s = try? c.decode(String.self) {
            self = .string(s)
        } else if let a = try? c.decode([JSONValue].self) {
            self = .array(a)
        } else if let o = try? c.decode([String: JSONValue].self) {
            self = .object(o)
        } else {
            self = .null
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case let .string(s): try c.encode(s)
        case let .number(n): try c.encode(n)
        case let .bool(b): try c.encode(b)
        case let .object(o): try c.encode(o)
        case let .array(a): try c.encode(a)
        case .null: try c.encodeNil()
        }
    }

    var doubleValue: Double? {
        if case let .number(n) = self { return n }
        return nil
    }

    var stringValue: String? {
        if case let .string(s) = self { return s }
        return nil
    }

    var boolValue: Bool? {
        if case let .bool(b) = self { return b }
        return nil
    }

    subscript(_ key: String) -> JSONValue? {
        if case let .object(o) = self { return o[key] }
        return nil
    }
}
