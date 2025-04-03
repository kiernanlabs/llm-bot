import streamlit as st
import pandas as pd
import json
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import urllib.parse
import re
import concurrent.futures
import time

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
            content = response.output_text
            
            # Show response preview
            st.write("### Response Preview (first 500 chars):")
            st.text(f"{content[:500]}{'...' if len(content) > 500 else ''}")
            
            # Show full response in a collapsible section using st.checkbox + conditional display
            show_full = st.checkbox("Show Full Response", key="openai_full")
            if show_full:
                st.write("### Full Raw Response:")
                st.code(content)
            
            parsed_content = json.loads(content)
            
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
            
            # Show response preview
            st.write("### Response Preview (first 500 chars):")
            st.text(f"{content[:500]}{'...' if len(content) > 500 else ''}")
            
            # Show full response in a collapsible section using st.checkbox + conditional display
            show_full = st.checkbox("Show Full Response", key="claude_full")
            if show_full:
                st.write("### Full Raw Response:")
                st.code(content)
            
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
                    st.write(f"Debug: Found JSON in text using regex (preview): {json_str[:200]}{'...' if len(json_str) > 200 else ''}")
                    
                    # Show extracted JSON
                    show_extracted = st.checkbox("Show Extracted JSON", key="claude_json")
                    if show_extracted:
                        st.write("### Full Extracted JSON:")
                        st.code(json_str)
                    
                    return json.loads(json_str)
                else:
                    st.error("Could not find valid JSON in Anthropic response")
                    return None
            except Exception as e:
                st.error(f"Anthropic API Error: {str(e)}")
                st.write(f"Debug: Exception details: {str(e)}")
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
            
            # Show response preview
            st.write("### Response Preview (first 500 chars):")
            st.text(f"{content[:500]}{'...' if len(content) > 500 else ''}")
            
            # Show full response in a collapsible section using st.checkbox + conditional display
            show_full = st.checkbox("Show Full Response", key="gemini_full")
            if show_full:
                st.write("### Full Raw Response:")
                st.code(content)
            
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
                    st.write(f"Debug: Found JSON in text using regex (preview): {json_str[:200]}{'...' if len(json_str) > 200 else ''}")
                    
                    # Show extracted JSON
                    show_extracted = st.checkbox("Show Extracted JSON", key="gemini_json")
                    if show_extracted:
                        st.write("### Full Extracted JSON:")
                        st.code(json_str)
                    
                    return json.loads(json_str)
                else:
                    st.error("Could not find valid JSON in Google response")
                    return None
            except Exception as e:
                st.error(f"Google API Error: {str(e)}")
                st.write(f"Debug: Exception details: {str(e)}")
                return None
        except Exception as e:
            st.error(f"Google API Error: {str(e)}")
            st.write(f"Debug: Exception details: {str(e)}")
            return None

# Function to normalize URLs (extract just the domain)
def normalize_url(url):
    try:
        # Parse the URL
        parsed_url = urllib.parse.urlparse(url)
        
        # Extract the domain (netloc)
        domain = parsed_url.netloc
        
        # Remove 'www.' prefix if present
        domain = re.sub(r'^www\.', '', domain)
        
        # If we couldn't extract a domain, return the original URL
        if not domain:
            return url
            
        return domain
    except:
        # If there's any error in parsing, return the original URL
        return url

# Submit button
if st.button("Research Companies"):
    if query:
        with st.spinner("Researching companies using multiple AI models..."):
            # Create a progress section
            st.markdown("### Progress")
            progress_container = st.container()
            
            # Set number of runs per model
            runs_per_model = 5
            
            # Initialize results containers
            all_results_list = []
            
            # Run each model multiple times
            for model_name in ["OpenAI", "Claude", "Gemini"]:
                # Create a progress status for this model
                model_progress = progress_container.empty()
                model_progress.info(f"{model_name}: Starting {runs_per_model} runs...")
                
                # Run the model multiple times
                for run_num in range(1, runs_per_model + 1):
                    run_progress = progress_container.empty()
                    run_progress.info(f"{model_name} - Run {run_num}/{runs_per_model}: Querying...")
                    
                    try:
                        # Call the appropriate model
                        if model_name == "OpenAI":
                            result = call_openai(get_prompt(query))
                        elif model_name == "Claude":
                            result = call_anthropic(get_prompt(query))
                        else:  # Gemini
                            result = call_google(get_prompt(query))
                        
                        # Check if we got a valid result
                        if result:
                            run_progress.success(f"{model_name} - Run {run_num}/{runs_per_model}: Complete")
                            
                            # Process the result based on its format
                            df = pd.DataFrame()
                            
                            if model_name == "OpenAI" and "companies" in result:
                                df = pd.DataFrame(result["companies"])
                            elif "companies" in result:
                                df = pd.DataFrame(result["companies"])
                            elif isinstance(result, list):
                                df = pd.DataFrame(result)
                            elif result:
                                df = pd.DataFrame([result])
                            
                            # Add source metadata to the results
                            if not df.empty:
                                # Normalize column names
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
                                
                                # Add source information
                                df["Source Model"] = model_name
                                df["Run Number"] = run_num
                                
                                # Add to all results
                                all_results_list.append(df)
                        else:
                            run_progress.error(f"{model_name} - Run {run_num}/{runs_per_model}: Failed")
                    except Exception as e:
                        run_progress.error(f"{model_name} - Run {run_num}/{runs_per_model}: Error - {str(e)[:50]}...")
                
                # Update overall model progress
                model_progress.success(f"{model_name}: All runs completed")
            
            # Combine all results
            if all_results_list:
                all_results = pd.concat(all_results_list, ignore_index=True)
                
                # Add normalized URL for grouping
                all_results["Normalized URL"] = all_results["URL"].apply(normalize_url)
                
                # Display run summary
                st.markdown("### Run Summary")
                summary_cols = st.columns(3)
                
                total_runs = len(all_results_list)
                summary_cols[0].metric("Total Runs", f"{total_runs}")
                summary_cols[1].metric("Unique Companies", f"{all_results['Normalized URL'].nunique()}")
                summary_cols[2].metric("Data Points", f"{len(all_results)}")
                
                # Group by URL to get stats
                url_stats = {}
                total_runs = runs_per_model * 3  # 3 models
                
                for norm_url in all_results["Normalized URL"].unique():
                    # Get all entries for this normalized URL
                    url_entries = all_results[all_results["Normalized URL"] == norm_url]
                    
                    # Calculate metrics across all runs
                    runs_appeared_in = len(url_entries)
                    appearance_pct = (runs_appeared_in / total_runs) * 100
                    avg_rank = url_entries["Rank"].mean()
                    
                    # Calculate visibility score (10 points for #1, 9 points for #2, etc.)
                    visibility_score = 0
                    for _, row in url_entries.iterrows():
                        rank = row["Rank"]
                        # Award points based on rank (max 10 points for rank 1)
                        if 1 <= rank <= 10:
                            visibility_score += (11 - rank)
                    
                    # Get company name (use most common one if different across runs)
                    company_name = url_entries["Company Name"].mode()[0]
                    
                    # Get the most common actual URL for display
                    display_url = url_entries["URL"].mode()[0]
                    
                    # Store the commentaries from each model and run
                    commentaries = {}
                    for _, row in url_entries.iterrows():
                        model_run = f"{row['Source Model']} (Run {row['Run Number']})"
                        commentaries[model_run] = row["Commentary on Differentiators"]
                    
                    # Store the stats
                    url_stats[norm_url] = {
                        "Company Name": company_name,
                        "URL": display_url,
                        "Normalized URL": norm_url,
                        "Runs Appeared": runs_appeared_in,
                        "Appearance %": appearance_pct,
                        "Average Rank": avg_rank,
                        "Visibility Score": visibility_score,
                        "Commentaries": commentaries
                    }
                
                # Convert to dataframe and sort
                stats_df = pd.DataFrame(url_stats.values())
                stats_df = stats_df.sort_values(by=["Visibility Score", "Appearance %", "Average Rank"], 
                                               ascending=[False, False, True])
                
                # Display the consolidated results
                st.markdown("### Consolidated Results")
                
                # Display the table
                for _, row in stats_df.iterrows():
                    cols = st.columns([3, 1, 1, 1, 2])
                    
                    # Column 1: Company Name and URL
                    cols[0].markdown(f"**{row['Company Name']}**")
                    cols[0].markdown(f"[{row['URL']}]({row['URL']})")
                    
                    # Column 2: Appearance %
                    cols[1].metric("Appearance", f"{row['Appearance %']:.0f}%")
                    
                    # Column 3: Average Rank
                    cols[2].metric("Avg Rank", f"{row['Average Rank']:.1f}")
                    
                    # Column 4: Visibility Score
                    cols[3].metric("Visibility", f"{row['Visibility Score']:.0f}")
                    
                    # Column 5: Commentaries Expander
                    with cols[4].expander("View Differentiators"):
                        for model_run, commentary in row["Commentaries"].items():
                            st.markdown(f"**{model_run}:**")
                            st.markdown(commentary)
                            st.markdown("---")
            else:
                st.error("All model runs failed to return valid results. Please try again.")
                st.stop()
            
            # Display raw data for debugging
            with st.expander("View Raw Data"):
                st.markdown("### URL Normalization")
                if not all_results.empty:
                    url_mapping = all_results[["URL", "Normalized URL"]].drop_duplicates()
                    st.dataframe(url_mapping)
                
                st.markdown("### Complete Dataset")
                if not all_results.empty:
                    st.dataframe(all_results)
    else:
        st.warning("Please enter a research query.")
