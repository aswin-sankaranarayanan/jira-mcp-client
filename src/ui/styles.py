"""UI styles for production-grade chat experience."""

APP_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    height: 100%;
    margin: 0;
    overflow: hidden !important;
}

[data-testid="stAppViewContainer"] > .main {
    height: 100dvh;
    min-height: 100vh;
    max-height: 100dvh;
    overflow: hidden !important;
    background:
        radial-gradient(circle at 10% 5%, #f9f7f1 0%, transparent 30%),
        radial-gradient(circle at 90% 10%, #e5f0ea 0%, transparent 28%),
        linear-gradient(180deg, #fcfbf8 0%, #f5f7f3 100%);
}

.main .block-container {
    height: 100%;
    max-height: 100%;
    max-width: 1280px;
    width: 100%;
    padding-top: 0;
    padding-bottom: 5.8rem;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    overflow: hidden !important;
}

.st-key-header_panel {
    position: sticky;
    top: 0;
    z-index: 35;
    background: transparent;
    backdrop-filter: none;
    padding-top: 0;
}

.st-key-header_panel h1 {
    margin-top: 0;
}

h1, h2, h3 {
    font-family: "Trebuchet MS", "Segoe UI", sans-serif;
    letter-spacing: 0.01em;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(46, 84, 63, 0.16);
    background: rgba(255, 255, 255, 0.72);
    border-radius: 16px;
    backdrop-filter: blur(2px);
    box-shadow: 0 12px 28px rgba(30, 48, 40, 0.08);
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 0.4rem 0.6rem;
}

.st-key-chat_panel {
    flex: 1 1 auto;
    min-height: 0;
}

.st-key-chat_panel [data-testid="stVerticalBlockBorderWrapper"] {
    height: 100%;
    overflow-y: auto;
    overscroll-behavior: contain;
}

.chat-note {
    color: #415146;
    font-size: 0.92rem;
    margin-top: 0.45rem;
}

[data-testid="stChatInput"] {
    position: fixed;
    left: 50%;
    bottom: 1.05rem;
    transform: translateX(-50%);
    width: min(1280px, calc(100vw - 2rem));
    z-index: 45;
    background: transparent;
    backdrop-filter: blur(3px);
    border-radius: 14px;
    padding-top: 0.2rem;
}

[data-testid="stChatInput"] textarea {
    max-height: 180px;
}

[data-testid="stChatInput"] > div {
    margin-bottom: 0;
}

[data-testid="stBottomBlockContainer"] {
    pointer-events: none;
}

[data-testid="stBottomBlockContainer"] [data-testid="stChatInput"] {
    pointer-events: auto;
    border-radius: 14px;
}

@media (max-width: 768px) {
    .main .block-container {
        padding-top: 0;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }

    .main .block-container {
        max-width: 100%;
        padding-bottom: 5.5rem;
    }

    [data-testid="stChatInput"] {
        width: calc(100vw - 1rem);
        bottom: 0.85rem;
    }
}
</style>
"""
