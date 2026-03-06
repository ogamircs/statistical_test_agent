# 2026-03-06 Chat Layout Design

## Goal
- Center the composer on the empty state.
- Move the composer to the bottom once a conversation starts.
- Add quick suggestion chips near the empty-state composer.
- Keep conversation history visible in a persistent left sidebar.

## Approach
- Use Chainlit configuration plus custom CSS/JS assets layered on the existing UI.
- Detect empty-vs-active conversation state from the rendered DOM and toggle layout classes.
- Inject suggestion chips that submit canned prompts into the existing composer.
- Expand and restyle the built-in sidebar instead of replacing the app shell.
