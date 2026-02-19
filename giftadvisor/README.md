# Gift Advisor

A standalone gift recommendation website that helps people find the perfect gift for specific occasions. Built using patterns from the havanora-shopify chat companion as a baseline.

## Features

- **Occasion-focused**: Birthday, anniversary, wedding, holiday, graduation, baby shower, housewarming, thank you
- **Conversational AI**: Asks about the recipient, budget, and preferences to suggest personalized gifts
- **Amazon products**: SerpAPI Amazon engine returns native Amazon URLs
- **Product carousel**: Image, title, brief description, price; horizontal scroll
- **Streaming responses**: Real-time typing effect for a natural chat experience
- **Standalone**: No Shopify dependency—runs as its own web app

## Setup

1. **Create a virtual environment** (recommended):

   ```bash
   cd giftadvisor
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set your API keys** (in `.env` or environment):

   ```bash
   OPENAI_API_KEY=sk-your-key-here
   SERPAPI_API_KEY=your-serpapi-key   # For Amazon products (serpapi.com, engine=amazon)
   ```

  Optional: `OPENAI_MODEL` (default: `gpt-4.1-mini`), `AMAZON_AFFILIATE_TAG` (default: bestgift0514-20)

4. **Run the app**:

   ```bash
   python app.py
   ```

5. Open **http://localhost:5001** in your browser.

## Project Structure

```
giftadvisor/
├── app.py              # Flask app, routes
├── gift_advisor.py     # AI logic, streaming, gift-focused prompts
├── requirements.txt
├── README.md
└── static/
    ├── index.html      # Main page
    ├── styles.css      # Styling
    └── app.js          # Frontend logic
```

## API

- `POST /gift_advisor` — Chat endpoint for gift recommendations
  - Body: `{ "message": "...", "occasion": "...", "history": [...], "stream": true }`
  - Returns: SSE stream (delta events + final payload) or JSON

## Relationship to havanora-shopify

This app reuses:

- Flask + CORS setup
- OpenAI streaming pattern (adapted for Chat Completions)
- Message history and context handling
- SSE event format (`delta`, `final`, `done`)
- Chat UI layout (messages, composer, typing indicator)

Changes from the chat companion:

- **Purpose**: Gift recommendations instead of general companionship
- **Occasion selection**: Chips for common gift occasions
- **System prompt**: Gift advisor persona, asks about recipient/budget
- **No telemetry/threads**: Simplified for standalone use (can be added later)
- **No Shopify**: Works without any e-commerce platform
