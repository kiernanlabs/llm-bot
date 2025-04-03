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
    
    Format the response as a JSON array of objects with these exact fields:
    [
        {{
            "Rank": number,
            "Company Name": "string",
            "URL": "string",
            "Commentary on Differentiators": "string"
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
        return json.loads(response.content[0].text)
    except Exception as e:
        st.error(f"Anthropic API Error: {str(e)}")
        return None

# Function to call Google
def call_google(prompt):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Google API Error: {str(e)}")
        return None

# Submit button
if st.button("Research Companies"):
    if query:
        with st.spinner("Researching companies using multiple AI models..."):
            # Get results from each model
            openai_results = call_openai(get_prompt(query))
            anthropic_results = call_anthropic(get_prompt(query))
            google_results = call_google(get_prompt(query))
            
            # Create DataFrames for each model
            dfs = {
                "GPT-4": pd.DataFrame(openai_results) if openai_results else pd.DataFrame(),
                "Claude": pd.DataFrame(anthropic_results) if anthropic_results else pd.DataFrame(),
                "Gemini": pd.DataFrame(google_results) if google_results else pd.DataFrame()
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
