import SwiftUI

/// Centered icon + message for empty / offseason / warming / error states.
struct EmptyStateView: View {
    let icon: String
    let title: String
    var message: String? = nil
    var retry: (() -> Void)? = nil

    var body: some View {
        VStack(spacing: FFSpacing.md) {
            Image(systemName: icon)
                .font(.system(size: 44))
                .foregroundStyle(FFColor.textMuted)
            Text(title)
                .font(.headline)
                .foregroundStyle(FFColor.textPrimary)
                .multilineTextAlignment(.center)
            if let message {
                Text(message)
                    .font(.subheadline)
                    .foregroundStyle(FFColor.textSecondary)
                    .multilineTextAlignment(.center)
            }
            if let retry {
                Button("Try Again", action: retry)
                    .buttonStyle(.borderedProminent)
                    .tint(FFColor.accent)
                    .padding(.top, FFSpacing.xs)
            }
        }
        .padding(FFSpacing.xl)
        .frame(maxWidth: .infinity)
    }
}
