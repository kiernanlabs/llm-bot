import streamlit as st
import pandas as pd
import json
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

# Initialize API clients using Streamlit secrets
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Set page config
st.set_page_config(
    page_title="LLM Company Research",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 Multi-Model Company Research")
st.markdown("""
This app uses multiple AI models to research companies based on your query.
Results will be displayed in a comparative table showing rankings from each model.
""")

# Query input
query = st.text_input(
    "Enter your research query:",
    placeholder="e.g., Top AI companies in healthcare"
)

# Function to format the prompt
def get_prompt(query):
    return f"""Based on the following query, provide a list of companies in JSON format.
    Each company should have: Rank (1-10), Company Name, URL, and Commentary on Differentiators.
    Query: {query}
    
    IMPORTANT: Respond ONLY with a JSON array. Do not include any other text, markdown, or formatting.
    The response must be a valid JSON array of objects with these exact fields:
    [
        {{
            "Rank": number,
            "Company Name": "string",
            "URL": "string",
            "Commentary on Differentiators": "string"
        }}
    ]
    
    Example response format:
    [
        {{
            "Rank": 1,
            "Company Name": "Example Corp",
            "URL": "https://example.com",
            "Commentary on Differentiators": "Leading innovator in the field"
        }}
    ]
    """

# Function to call OpenAI
def call_openai(prompt):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return None

# Function to call Anthropic
def call_anthropic(prompt):
    try:
        response = anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        # Extract the text content and ensure it's properly formatted
        content = response.content[0].text.strip()
        # Try to find JSON content within the response
        try:
            # First try direct JSON parsing
            return json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                st.error("Could not find valid JSON in Anthropic response")
                return None
    except Exception as e:
        st.error(f"Anthropic API Error: {str(e)}")
        return None

# Function to call Google
def call_google(prompt):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Google API Error: {str(e)}")
        return None

# Submit button
if st.button("Research Companies"):
    if query:
        with st.spinner("Researching companies using multiple AI models..."):
            # Create progress indicators for each model
            progress_containers = {
                "GPT-4": st.empty(),
                "Claude": st.empty(),
                "Gemini": st.empty()
            }
            
            # Get results from each model with progress updates
            progress_containers["GPT-4"].info("Querying GPT-4...")
            openai_results = call_openai(get_prompt(query))
            progress_containers["GPT-4"].success("GPT-4 results received") if openai_results else progress_containers["GPT-4"].error("GPT-4 query failed")
            
            progress_containers["Claude"].info("Querying Claude...")
            anthropic_results = call_anthropic(get_prompt(query))
            progress_containers["Claude"].success("Claude results received") if anthropic_results else progress_containers["Claude"].error("Claude query failed")
            
            progress_containers["Gemini"].info("Querying Gemini...")
            google_results = call_google(get_prompt(query))
            progress_containers["Gemini"].success("Gemini results received") if google_results else progress_containers["Gemini"].error("Gemini query failed")
            
            # Create DataFrames for each model
            dfs = {
                "GPT-4": pd.DataFrame([openai_results] if isinstance(openai_results, dict) else (openai_results or [])),
                "Claude": pd.DataFrame([anthropic_results] if isinstance(anthropic_results, dict) else (anthropic_results or [])),
                "Gemini": pd.DataFrame([google_results] if isinstance(google_results, dict) else (google_results or []))
            }
            
            # Display results in a table
            st.markdown("### Results")
            
            # Create a container for the table
            table_container = st.container()
            
            # Create the table header
            cols = st.columns(len(dfs))
            for col, (model, _) in zip(cols, dfs.items()):
                col.markdown(f"**{model}**")
            
            # Display the results
            max_rows = max(len(df) for df in dfs.values() if not df.empty)
            for row in range(max_rows):
                cols = st.columns(len(dfs))
                for col, (model, df) in zip(cols, dfs.items()):
                    if not df.empty and row < len(df):
                        company_data = df.iloc[row]
                        col.markdown(
                            f"**{company_data['Rank']}. {company_data['Company Name']}**\n\n"
                            f"[{company_data['URL']}]({company_data['URL']})\n\n"
                            f"<div title='{company_data['Commentary on Differentiators']}'>"
                            f"💡 Hover for details</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        col.markdown("")
    else:
        st.warning("Please enter a research query.")
