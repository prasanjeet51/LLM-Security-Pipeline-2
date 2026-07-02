"""Dark glassmorphism theme and CSS for the Gradio UI."""

import gradio as gr


def get_theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        body_background_fill="transparent",
        button_primary_background_fill="linear-gradient(90deg, #4f46e5, #7c3aed)",
        button_primary_background_fill_hover="linear-gradient(90deg, #4338ca, #6d28d9)",
        button_primary_text_color="white",
        block_background_fill="transparent",
    )


def get_css() -> str:
    return """
/* Dark gradient background */
body, .gradio-container {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    min-height: 100vh;
}

/* Glassmorphism cards */
.block, .panel, .form {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
}

/* Gradient text for h1 */
h1 {
    background: linear-gradient(90deg, #6366f1, #a855f7) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-weight: 800 !important;
    font-size: 2.5rem !important;
}

/* Button glow hover */
button.primary {
    background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
    border: none !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.6) !important;
}

/* Input focus glow */
textarea:focus, input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.3) !important;
    outline: none !important;
}

/* Shimmer animation */
@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}

/* Animated gradient border on .generating */
@keyframes borderGlow {
    0%, 100% { border-color: rgba(99, 102, 241, 0.6); }
    50% { border-color: rgba(168, 85, 247, 0.8); }
}
.generating {
    background: linear-gradient(
        90deg,
        rgba(99, 102, 241, 0.1) 25%,
        rgba(168, 85, 247, 0.2) 50%,
        rgba(99, 102, 241, 0.1) 75%
    ) !important;
    background-size: 1000px 100% !important;
    border: 1px solid rgba(99, 102, 241, 0.6) !important;
    animation:
        shimmer 2s infinite linear,
        borderGlow 2s infinite ease-in-out !important;
}

/* Result reveal animation */
@keyframes slideUpFadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-reveal {
    animation: slideUpFadeIn 0.4s ease-out forwards !important;
}

/* Custom thin scrollbar — 6px, indigo thumb */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.05); }
::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4f46e5; }

/* Readable text */
label, .label-wrap, p, span {
    color: rgba(255, 255, 255, 0.85) !important;
}

/* Tab button visibility */
button[role="tab"] {
    color: rgba(255, 255, 255, 0.65) !important;
    background: transparent !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}
button[role="tab"][aria-selected="true"],
button[role="tab"].selected {
    color: white !important;
    border-bottom: 2px solid #6366f1 !important;
    font-weight: 700 !important;
}
button:not(.primary) { color: rgba(255, 255, 255, 0.85) !important; }
textarea, input[type="text"] { color: rgba(255, 255, 255, 0.92) !important; }

/* Decision card variants */
.decision-allow {
    background: linear-gradient(135deg, #065f46, #059669) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 28px !important;
    text-align: center !important;
}
.decision-block {
    background: linear-gradient(135deg, #7f1d1d, #dc2626) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 28px !important;
    text-align: center !important;
}
.decision-review {
    background: linear-gradient(135deg, #78350f, #d97706) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 28px !important;
    text-align: center !important;
}

/* Pipeline stage trace */
.stage-step {
    padding: 6px 12px !important;
    margin: 3px 0 !important;
    border-left: 3px solid #6366f1 !important;
    color: rgba(255, 255, 255, 0.8) !important;
    font-size: 0.85rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}
"""
