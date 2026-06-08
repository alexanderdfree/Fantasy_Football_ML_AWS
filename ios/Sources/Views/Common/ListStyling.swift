import SwiftUI

extension View {
    /// Apply the app's dark canvas to a scrollable container (List/ScrollView/Form).
    func ffScreenBackground() -> some View {
        scrollContentBackground(.hidden)
            .background(FFColor.bgPrimary)
    }
}
