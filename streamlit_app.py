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

# Model selection
with st.expander("Model Settings", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("OpenAI")
        openai_model = st.text_input("OpenAI Model", value="gpt-4o", key="openai_model")
    
    with col2:
        st.subheader("Anthropic")
        anthropic_model = st.text_input("Anthropic Model", value="claude-3-7-sonnet-20250219", key="anthropic_model")
    
    with col3:
        st.subheader("Google")
        google_model = st.text_input("Google Model", value="gemini-2.0-flash", key="google_model")

# Display current models in info box
st.info(f"Currently using: OpenAI: {openai_model} | Anthropic: {anthropic_model} | Google: {google_model}")

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
            st.write("Debug: Sending request to OpenAI...")
            st.write(f"Debug: Using model: {openai_model}")
            
            # Define the JSON schema for the companies list
            companies_schema = {
                "type": "object",
                "properties": {
                    "companies": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Rank": {"type": "integer"},
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
                model=openai_model,
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
    with st.expander("Claude Debug Output", expanded=False):
        try:
            # Add JSON formatting instructions for Anthropic
            formatted_prompt = add_json_instructions(prompt)
            
            st.write("Debug: Sending request to Anthropic...")
            st.write(f"Debug: Using model: {anthropic_model}")
            
            response = anthropic_client.messages.create(
                model=anthropic_model,
                max_tokens=1000,
                messages=[{"role": "user", "content": formatted_prompt}]
            )
            
            st.write("Debug: Received response from Anthropic")
            
            # Extract the text content and ensure it's properly formatted
            content = response.content[0].text.strip()
            st.write(f"Debug: Raw response content: {content[:500]}{'...' if len(content) > 500 else ''}")
            
            # Try to find JSON content within the response
            try:
                # First try direct JSON parsing
                parsed_content = json.loads(content)
                st.write("Debug: Successfully parsed JSON response")
                return parsed_content
            except json.JSONDecodeError as e:
                st.write(f"Debug: JSON parsing failed with error: {str(e)}")
                # If that fails, try to extract JSON from the text
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    st.write(f"Debug: Found JSON in text using regex: {json_str[:200]}{'...' if len(json_str) > 200 else ''}")
                    return json.loads(json_str)
                else:
                    st.error("Could not find valid JSON in Anthropic response")
                    st.markdown("**Raw Response:**")
                    st.code(content)
                    return None
        except Exception as e:
            st.error(f"Anthropic API Error: {str(e)}")
            st.write(f"Debug: Exception details: {str(e)}")
            return None

# Function to call Google
def call_google(prompt):
    with st.expander("Gemini Debug Output", expanded=False):
        try:
            # Add JSON formatting instructions for Google
            formatted_prompt = add_json_instructions(prompt)
            
            st.write("Debug: Sending request to Google Gemini...")
            st.write(f"Debug: Using model: {google_model}")
            
            model = genai.GenerativeModel(google_model)
            response = model.generate_content(formatted_prompt)
            
            st.write("Debug: Received response from Google")
            
            if not response or not response.text:
                st.error("Google returned empty response")
                return None
                
            content = response.text.strip()
            st.write(f"Debug: Raw response content: {content[:500]}{'...' if len(content) > 500 else ''}")
            
            # Try to parse the JSON response
            try:
                parsed_content = json.loads(content)
                st.write("Debug: Successfully parsed JSON response")
                return parsed_content
            except json.JSONDecodeError as e:
                st.write(f"Debug: JSON parsing failed with error: {str(e)}")
                # If that fails, try to extract JSON from the text
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    st.write(f"Debug: Found JSON in text using regex: {json_str[:200]}{'...' if len(json_str) > 200 else ''}")
                    return json.loads(json_str)
                else:
                    st.error("Could not find valid JSON in Google response")
                    st.markdown("**Raw Response:**")
                    st.code(content)
                    return None
        except Exception as e:
            st.error(f"Google API Error: {str(e)}")
            st.write(f"Debug: Exception details: {str(e)}")
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
            
            # Process OpenAI results
            if openai_results and "companies" in openai_results:
                dfs["GPT-4"] = pd.DataFrame(openai_results["companies"])
            elif openai_results:
                dfs["GPT-4"] = pd.DataFrame(openai_results)
            else:
                dfs["GPT-4"] = pd.DataFrame([])
            
            # Process Anthropic results
            if anthropic_results and "companies" in anthropic_results:
                dfs["Claude"] = pd.DataFrame(anthropic_results["companies"])
            elif isinstance(anthropic_results, list):
                dfs["Claude"] = pd.DataFrame(anthropic_results)
            elif anthropic_results:
                dfs["Claude"] = pd.DataFrame([anthropic_results])
            else:
                dfs["Claude"] = pd.DataFrame([])
            
            # Process Google results
            if google_results and "companies" in google_results:
                dfs["Gemini"] = pd.DataFrame(google_results["companies"])
            elif isinstance(google_results, list):
                dfs["Gemini"] = pd.DataFrame(google_results)
            elif google_results:
                dfs["Gemini"] = pd.DataFrame([google_results])
            else:
                dfs["Gemini"] = pd.DataFrame([])
            
            # Normalize column names (handle case differences)
            for model, df in dfs.items():
                if not df.empty:
                    # Normalize column names to handle case variations
                    column_map = {}
                    for col in df.columns:
                        if col.lower() == "rank":
                            column_map[col] = "Rank"
                        elif col.lower() in ["company", "company name", "name"]:
                            column_map[col] = "Company Name"
                        elif col.lower() in ["url", "website", "link"]:
                            column_map[col] = "URL"
                        elif col.lower() in ["commentary", "differentiators", "commentary on differentiators"]:
                            column_map[col] = "Commentary on Differentiators"
                    
                    # Rename columns if needed
                    if column_map:
                        df.rename(columns=column_map, inplace=True)
            
            # Display results in a table
            st.markdown("### Results")
            
            # Create a container for the table
            table_container = st.container()
            
            # Create the table header
            cols = st.columns(len(dfs))
            for col, (model, _) in zip(cols, dfs.items()):
                col.markdown(f"**{model}**")
            
            # Display the results
            max_rows = max((len(df) for df in dfs.values() if not df.empty), default=0)
            for row in range(max_rows):
                cols = st.columns(len(dfs))
                for col, (model, df) in zip(cols, dfs.items()):
                    if not df.empty and row < len(df):
                        company_data = df.iloc[row]
                        
                        # Extract data with fallbacks
                        rank = company_data.get("Rank", row + 1) if hasattr(company_data, "get") else row + 1
                        company_name = company_data.get("Company Name", "Unknown") if hasattr(company_data, "get") else "Unknown"
                        url = company_data.get("URL", "#") if hasattr(company_data, "get") else "#"
                        commentary = company_data.get("Commentary on Differentiators", "") if hasattr(company_data, "get") else ""
                        
                        col.markdown(
                            f"**{rank}. {company_name}**\n\n"
                            f"[{url}]({url})"
                        )
                        with col.expander("View Details"):
                            st.markdown(f"**Commentary on Differentiators:**")
                            st.markdown(commentary)
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
