# Hugging Face Deployment Strategy

Project 01 now has two separate application layers:

| Layer | Use |
|---|---|
| Python/Gradio | Local development, Python inference, engineering demonstration, and compute-backed hosting where eligible |
| Static/Transformers.js | Free public Hugging Face Space with real browser-based Transformer inference |

For the user's current free-plan goal, deploy **only the contents of `web/`** to the Hugging Face Space.

See `STATIC_SPACE_DEPLOYMENT.md` for configuration, automation, manual deployment, and troubleshooting.
