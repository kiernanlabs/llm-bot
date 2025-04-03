# Multi-Model Company Research App

A Streamlit application that uses multiple AI models (OpenAI GPT-4, Anthropic Claude, and Google Gemini) to research and compare company information based on user queries.

## Features

- Query-based company research using multiple AI models
- Comparative analysis of results from different models
- Interactive table display with hover details
- Clean and modern user interface

## Setup

1. Clone the repository:
```bash
git clone https://github.com/kiernanlabs/llm-bot.git
cd llm-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys:
   - Create a `.streamlit` directory if it doesn't exist:
     ```bash
     mkdir -p .streamlit
     ```
   - Copy the example secrets file to create your own:
     ```bash
     cp .streamlit/secrets.toml.example .streamlit/secrets.toml
     ```
   - Edit `.streamlit/secrets.toml` and replace the placeholder values with your actual API keys:
     ```toml
     OPENAI_API_KEY = "your_actual_openai_api_key"
     ANTHROPIC_API_KEY = "your_actual_anthropic_api_key"
     GOOGLE_API_KEY = "your_actual_google_api_key"
     ```
   - **IMPORTANT**: Never commit your actual API keys to the repository. The `.streamlit/secrets.toml` file is already in the `.gitignore` file.

4. Run the app:
```bash
streamlit run streamlit_app.py
```

## Usage

1. Enter your research query in the text input field
2. Click "Research Companies"
3. View the comparative results from different AI models
4. Hover over the 💡 icon to see detailed commentary about each company

## API Keys Required

- OpenAI API key (for GPT-4)
- Anthropic API key (for Claude)
- Google AI API key (for Gemini)

## Deployment

When deploying to Streamlit Cloud:
1. Add your API keys in the Streamlit Cloud dashboard under "Secrets"
2. The keys should be added in the same format as the secrets.toml file
3. Never share your API keys or commit them to the repository

## Security Notes

- API keys are sensitive information. Never share them or commit them to version control.
- The `.streamlit/secrets.toml` file is included in `.gitignore` to prevent accidental commits.
- When deploying to Streamlit Cloud, use the secure secrets management in the dashboard.

## License

Apache License 2.0
