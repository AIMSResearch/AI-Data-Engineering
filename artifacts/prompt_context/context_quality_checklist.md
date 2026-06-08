# Context Quality Checklist

- Confirm the prompt template version is approved for the target model.
- Confirm retrieved chunks come from the expected corpus version.
- Confirm permission filtering was applied before context assembly.
- Confirm tool outputs are schema-valid before insertion into the prompt.
- Confirm token truncation did not remove policy or safety constraints.
- Confirm the prompt instance, retrieved context, and response can be reconstructed.
- Confirm benchmark conversations cover the main failure categories for this release.
