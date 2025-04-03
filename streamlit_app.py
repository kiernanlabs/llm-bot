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
    return f"""Based on the following query, provide a list of 10 companies.
    For each company, include the rank (1-10), company name, URL, and commentary on what differentiates this company.
    
    Query: {query}
    """

# Helper function to add JSON formatting instructions for non-OpenAI models
def add_json_instructions(base_prompt):
    return base_prompt + """
    
    IMPORTANT: You must respond ONLY with a valid JSON object in the following format:
    {
        "companies": [
            {
                "Rank": 1,
                "Company Name": "Example Corp",
                "URL": "https://example.com",
                "Commentary on Differentiators": "Example commentary"
            },
            ...and so on for all 10 companies...
        ]
    }
    
    Your entire response must be valid JSON that can be parsed with json.loads(). 
    Do not include any text, explanations, or markdown outside the JSON structure.
    Do not include backticks or code blocks (```).
    """

# Function to call OpenAI
def call_openai(prompt):
    with st.expander("OpenAI Debug Output", expanded=False):
        try:
            st.write("Debug: Sending request to OpenAI using responses API...")
            st.write(f"Debug: Using model: gpt-4o")
            
            # Define the JSON schema for the companies list
            companies_schema = {
                "type": "object",
                "properties": {
                    "companies": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Rank": {"type": "integer", "minimum": 1, "maximum": 10},
                                "Company Name": {"type": "string"},
                                "URL": {"type": "string"},
                                "Commentary on Differentiators": {"type": "string"}
                            },
                            "required": ["Rank", "Company Name", "URL", "Commentary on Differentiators"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["companies"],
                "additionalProperties": False
            }
            
            # Use the responses API with schema validation
            response = openai_client.responses.create(
                model="gpt-4o",
                input=[{"role": "user", "content": prompt}],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "company_research",
                        "schema": companies_schema,
                        "strict": True
                    }
                }
            )
            
            st.write("Debug: Received response from OpenAI")
            
            # Extract and parse the structured JSON response
            parsed_content = json.loads(response.output_text)
            
            # Show successful response for debugging
            st.write("Debug: Successfully parsed JSON response")
            return parsed_content
            
        except json.JSONDecodeError as e:
            st.error(f"JSON parsing error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"OpenAI API Error: {str(e)}")
            st.write(f"Debug: Exception details: {str(e)}")
            return None

# Function to call Anthropic
def call_anthropic(prompt):
    try:
        # Add JSON formatting instructions for Anthropic
        formatted_prompt = add_json_instructions(prompt)
        
        response = anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            messages=[{"role": "user", "content": formatted_prompt}]
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
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                st.error("Could not find valid JSON in Anthropic response")
                st.markdown("**Raw Response:**")
                st.code(content)
                return None
    except Exception as e:
        st.error(f"Anthropic API Error: {str(e)}")
        return None

# Function to call Google
def call_google(prompt):
    try:
        # Add JSON formatting instructions for Google
        formatted_prompt = add_json_instructions(prompt)
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(formatted_prompt)
        
        if not response or not response.text:
            st.error("Google returned empty response")
            return None
            
        content = response.text.strip()
        
        # Try to parse the JSON response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                st.error("Could not find valid JSON in Google response")
                st.markdown("**Raw Response:**")
                st.code(content)
                return None
    except Exception as e:
        st.error(f"Google API Error: {str(e)}")
        return None

# Submit button
if st.button("Research Companies"):
    if query:
        with st.spinner("Researching companies using multiple AI models..."):
            # Create a progress section
            st.markdown("### Progress")
            progress_cols = st.columns(3)
            
            # Get results from each model with progress updates
            with progress_cols[0]:
                st.info("Querying GPT-4...")
                openai_results = call_openai(get_prompt(query))
                if openai_results:
                    st.success("GPT-4 results received")
                else:
                    st.error("GPT-4 query failed")
            
            with progress_cols[1]:
                st.info("Querying Claude...")
                anthropic_results = call_anthropic(get_prompt(query))
                if anthropic_results:
                    st.success("Claude results received")
                else:
                    st.error("Claude query failed")
            
            with progress_cols[2]:
                st.info("Querying Gemini...")
                google_results = call_google(get_prompt(query))
                if google_results:
                    st.success("Gemini results received")
                else:
                    st.error("Gemini query failed")
            
            # Create DataFrames for each model
            dfs = {}
            
            if openai_results and "companies" in openai_results:
                dfs["GPT-4"] = pd.DataFrame(openai_results["companies"])
            elif openai_results:
                dfs["GPT-4"] = pd.DataFrame(openai_results)
            else:
                dfs["GPT-4"] = pd.DataFrame([])
                
            dfs["Claude"] = pd.DataFrame([anthropic_results] if isinstance(anthropic_results, dict) else (anthropic_results or []))
            dfs["Gemini"] = pd.DataFrame([google_results] if isinstance(google_results, dict) else (google_results or []))
            
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
                            f"[{company_data['URL']}]({company_data['URL']})"
                        )
                        with col.expander("View Details"):
                            st.markdown(f"**Commentary on Differentiators:**")
                            st.markdown(company_data['Commentary on Differentiators'])
                    else:
                        col.markdown("")
            
            # Display raw results for debugging
            with st.expander("View Raw Results"):
                st.markdown("### Raw Results")
                
                # Display raw results for each model
                for model, results in [
                    ("GPT-4", openai_results),
                    ("Claude", anthropic_results),
                    ("Gemini", google_results)
                ]:
                    st.markdown(f"**{model} Raw Response:**")
                    if results:
                        st.json(results)
                    else:
                        st.error(f"No results from {model}")
    else:
        st.warning("Please enter a research query.")
