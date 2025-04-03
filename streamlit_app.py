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
from datetime import datetime

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

# Initialize debug log container in session state if not present
if 'debug_logs' not in st.session_state:
    st.session_state.debug_logs = []

# Initialize results storage in session state
if 'all_results' not in st.session_state:
    st.session_state.all_results = None
if 'has_run_query' not in st.session_state:
    st.session_state.has_run_query = False

# Function to add a log entry
def add_debug_log(model, run_id, message, level="INFO", data=None):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "model": model,
        "run_id": run_id,
        "level": level,
        "message": message,
        "data": data
    }
    st.session_state.debug_logs.append(log_entry)
    return log_entry

# Model selection
with st.expander("Model Settings", expanded=False):
    # Add a setting for number of runs per model
    runs_per_model = st.number_input("Number of Runs per Model", min_value=1, max_value=10, value=5, step=1, help="How many times to run each model")
    
    # Store in session state for persistence
    if 'runs_per_model' not in st.session_state:
        st.session_state.runs_per_model = runs_per_model
    else:
        st.session_state.runs_per_model = runs_per_model
    
    st.markdown("---")
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
def call_openai(prompt, run_id=None):
    model_name = "OpenAI"
    add_debug_log(model_name, run_id, f"Starting request to {model_name} with model: {openai_model}")
    
    try:
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
        
        add_debug_log(model_name, run_id, "Response received")
        
        # Extract and parse the structured JSON response
        content = response.output_text
        
        # Log the response content
        add_debug_log(model_name, run_id, "Response preview", data=content[:500] + ('...' if len(content) > 500 else ''))
        add_debug_log(model_name, run_id, "Full response", level="DEBUG", data=content)
        
        parsed_content = json.loads(content)
        add_debug_log(model_name, run_id, "Successfully parsed JSON response")
        
        return parsed_content
        
    except json.JSONDecodeError as e:
        add_debug_log(model_name, run_id, f"JSON parsing error: {str(e)}", level="ERROR")
        return None
    except Exception as e:
        add_debug_log(model_name, run_id, f"API Error: {str(e)}", level="ERROR")
        return None

# Function to call Anthropic
def call_anthropic(prompt, run_id=None):
    model_name = "Claude"
    add_debug_log(model_name, run_id, f"Starting request to {model_name} with model: {anthropic_model}")
    
    try:
        # Import re module in this scope to ensure it's available
        import re
        
        # Add JSON formatting instructions for Anthropic
        formatted_prompt = add_json_instructions(prompt)
        
        # Make the JSON instructions even more explicit for Claude
        formatted_prompt += """
        
        IMPORTANT FORMATTING RULES:
        1. Do not use single quotes (') for JSON strings, always use double quotes (")
        2. Make sure all strings are properly terminated with a closing quote
        3. Escape any double quotes within strings with a backslash (\")
        4. Do not include any text, markdown formatting, or code block markers
        5. The response must be a single, valid JSON object
        """
        
        response = anthropic_client.messages.create(
            model=anthropic_model,
            max_tokens=4000,  # Increase max tokens to avoid truncation
            messages=[{"role": "user", "content": formatted_prompt}]
        )
        
        add_debug_log(model_name, run_id, "Response received")
        
        # Extract the text content and ensure it's properly formatted
        content = response.content[0].text.strip()
        
        # Log the response content
        add_debug_log(model_name, run_id, "Response preview", data=content[:500] + ('...' if len(content) > 500 else ''))
        add_debug_log(model_name, run_id, "Full response", level="DEBUG", data=content)
        
        # Clean up the content to help with JSON parsing
        def clean_json_string(json_str):
            # Remove any potential markdown code block syntax
            json_str = re.sub(r'^```json', '', json_str)
            json_str = re.sub(r'^```', '', json_str)
            json_str = re.sub(r'```$', '', json_str)
            
            # Remove any text before the first curly brace or after the last curly brace
            first_brace = json_str.find('{')
            last_brace = json_str.rfind('}')
            
            if first_brace != -1 and last_brace != -1:
                json_str = json_str[first_brace:last_brace+1]
            
            return json_str.strip()
        
        # Check for truncated JSON response
        def fix_truncated_json(json_str):
            # Look for incomplete JSON structure (missing closing braces)
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            
            # If we have unclosed braces, attempt to fix
            if open_braces > close_braces:
                add_debug_log(model_name, run_id, f"Detected truncated JSON: {open_braces} opening braces vs {close_braces} closing braces", level="WARNING")
                
                # Check for incomplete company entry
                if json_str.endswith('"Commentary on Differentiators": "'):
                    json_str += '"}]}'
                elif '"Commentary on Differentiators": "' in json_str and not json_str.endswith('}'):
                    # Find the last quote that might be missing a closing
                    last_quote_pos = json_str.rfind('"')
                    if last_quote_pos > 0 and json_str[last_quote_pos-1] != '\\':
                        json_str = json_str[:last_quote_pos+1] + '"}]}'
                else:
                    # Add the necessary closing braces
                    for _ in range(open_braces - close_braces):
                        if json_str.rstrip().endswith('"') or json_str.rstrip().endswith(','):
                            # If ending with quote or comma, assume we need to close an object
                            json_str += '"}]}'
                            break
                        else:
                            json_str += '}'
            
            return json_str
        
        # Function to ask the model to fix its own JSON
        def ask_model_to_fix_json(original_content, error_message):
            add_debug_log(model_name, run_id, "Asking model to fix its own JSON", level="INFO")
            
            fix_prompt = f"""
            You previously provided a JSON response that could not be parsed due to the following error:
            {error_message}
            
            Here is your original response:
            {original_content}
            
            Please fix the JSON formatting issues and provide a valid, parsable JSON object. 
            Only respond with the corrected JSON. No explanation, no markdown, no code blocks.
            Make sure the response is complete and all brackets are closed properly.
            """
            
            try:
                # Send request to fix the JSON
                fix_response = anthropic_client.messages.create(
                    model=anthropic_model,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": fix_prompt}]
                )
                
                fixed_content = fix_response.content[0].text.strip()
                add_debug_log(model_name, run_id, "Model provided fixed JSON", level="INFO")
                add_debug_log(model_name, run_id, "Fixed JSON response", level="DEBUG", data=fixed_content)
                
                # Clean the fixed content
                return clean_json_string(fixed_content)
            except Exception as fix_error:
                add_debug_log(model_name, run_id, f"Error asking model to fix JSON: {str(fix_error)}", level="ERROR")
                return None
        
        # Clean the content
        cleaned_content = clean_json_string(content)
        # Fix truncated JSON if needed
        cleaned_content = fix_truncated_json(cleaned_content)
        add_debug_log(model_name, run_id, "Cleaned JSON", level="DEBUG", data=cleaned_content)
        
        # Try to find JSON content within the response
        try:
            # First try direct JSON parsing
            parsed_content = json.loads(cleaned_content)
            add_debug_log(model_name, run_id, "Successfully parsed JSON response")
            return parsed_content
        except json.JSONDecodeError as e:
            add_debug_log(model_name, run_id, f"JSON parsing failed: {str(e)}", level="WARNING")
            
            # Log more detailed error information
            error_context = cleaned_content
            if hasattr(e, 'pos'):
                start = max(0, e.pos - 50)
                end = min(len(cleaned_content), e.pos + 50)
                error_context = f"...{cleaned_content[start:e.pos]}>>>ERROR HERE>>>{cleaned_content[e.pos:end]}..."
            
            add_debug_log(model_name, run_id, f"JSON error context", level="ERROR", data=error_context)
            
            # Ask the model to fix its own JSON
            fixed_json = ask_model_to_fix_json(content, str(e))
            if fixed_json:
                try:
                    # Try parsing the AI-fixed JSON
                    parsed_content = json.loads(fixed_json)
                    add_debug_log(model_name, run_id, "Successfully parsed AI-fixed JSON", level="INFO")
                    return parsed_content
                except json.JSONDecodeError as fix_e:
                    add_debug_log(model_name, run_id, f"AI-fixed JSON still has parsing issues: {str(fix_e)}", level="WARNING")
            
            # Try more aggressive JSON fixing approaches
            try:
                # Try with a JSON repair library if available
                try:
                    import json5
                    add_debug_log(model_name, run_id, "Attempting to parse with json5", level="DEBUG")
                    parsed_content = json5.loads(cleaned_content)
                    add_debug_log(model_name, run_id, "Successfully parsed with json5", level="INFO")
                    return parsed_content
                except ImportError:
                    add_debug_log(model_name, run_id, "json5 library not available", level="DEBUG")
                
                # Manual JSON fixing for common issues
                fixed_content = cleaned_content
                
                # Fix unterminated strings by looking for quotes without matching pairs
                quote_positions = [i for i, char in enumerate(fixed_content) if char == '"' and (i == 0 or fixed_content[i-1] != '\\')]
                if len(quote_positions) % 2 == 1:  # Odd number of quotes means unterminated string
                    last_quote = quote_positions[-1]
                    fixed_content = fixed_content[:last_quote+1] + '"' + fixed_content[last_quote+1:]
                    add_debug_log(model_name, run_id, "Fixed unterminated string", level="DEBUG", data=fixed_content)
                
                # Try parsing the fixed content
                parsed_content = json.loads(fixed_content)
                add_debug_log(model_name, run_id, "Successfully parsed after fixing", level="INFO")
                return parsed_content
            except Exception as fix_error:
                add_debug_log(model_name, run_id, f"JSON fixing failed: {str(fix_error)}", level="ERROR")
            
            # If that fails, try to extract JSON from the text
            json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                add_debug_log(model_name, run_id, "Found JSON in text using regex")
                add_debug_log(model_name, run_id, "Extracted JSON", level="DEBUG", data=json_str)
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Last resort: try to create a minimal valid JSON with the company data
                    add_debug_log(model_name, run_id, "Attempting to extract company data pattern", level="DEBUG")
                    companies = []
                    # Look for patterns like "Rank": 1, "Company Name": "Example"
                    company_pattern = r'"Rank"\s*:\s*(\d+).*?"Company Name"\s*:\s*"([^"]+)".*?"URL"\s*:\s*"([^"]+)".*?"Commentary[^"]*"\s*:\s*"([^"]+)"'
                    matches = re.finditer(company_pattern, content, re.DOTALL)
                    
                    for match in matches:
                        companies.append({
                            "Rank": int(match.group(1)),
                            "Company Name": match.group(2),
                            "URL": match.group(3),
                            "Commentary on Differentiators": match.group(4)
                        })
                    
                    if companies:
                        result = {"companies": companies}
                        add_debug_log(model_name, run_id, f"Created minimal JSON with {len(companies)} companies", level="INFO")
                        return result
                    
                    add_debug_log(model_name, run_id, "Could not extract company data", level="ERROR")
                    return None
            else:
                add_debug_log(model_name, run_id, "Could not find valid JSON in response", level="ERROR")
                return None
    except Exception as e:
        add_debug_log(model_name, run_id, f"API Error: {str(e)}", level="ERROR")
        return None

# Function to call Google
def call_google(prompt, run_id=None):
    model_name = "Gemini"
    add_debug_log(model_name, run_id, f"Starting request to {model_name} with model: {google_model}")
    
    try:
        # Add JSON formatting instructions for Google
        formatted_prompt = add_json_instructions(prompt)
        
        model = genai.GenerativeModel(google_model)
        response = model.generate_content(formatted_prompt)
        
        add_debug_log(model_name, run_id, "Response received")
        
        if not response or not response.text:
            add_debug_log(model_name, run_id, "Empty response received", level="ERROR")
            return None
        
        content = response.text.strip()
        
        # Log the response content
        add_debug_log(model_name, run_id, "Response preview", data=content[:500] + ('...' if len(content) > 500 else ''))
        add_debug_log(model_name, run_id, "Full response", level="DEBUG", data=content)
        
        # Try to parse the JSON response
        try:
            parsed_content = json.loads(content)
            add_debug_log(model_name, run_id, "Successfully parsed JSON response")
            return parsed_content
        except json.JSONDecodeError as e:
            add_debug_log(model_name, run_id, f"JSON parsing failed: {str(e)}", level="WARNING")
            # If that fails, try to extract JSON from the text
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                add_debug_log(model_name, run_id, "Found JSON in text using regex")
                add_debug_log(model_name, run_id, "Extracted JSON", level="DEBUG", data=json_str)
                
                return json.loads(json_str)
            else:
                add_debug_log(model_name, run_id, "Could not find valid JSON in response", level="ERROR")
                return None
    except Exception as e:
        add_debug_log(model_name, run_id, f"API Error: {str(e)}", level="ERROR")
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
        # Set flag to indicate query is running
        st.session_state.has_run_query = True
        
        with st.spinner("Researching companies using multiple AI models..."):
            # Create layout for different sections
            progress_container = st.container()
            results_container = st.container()
            
            # Set number of runs per model
            runs_per_model = st.session_state.runs_per_model
            total_runs = runs_per_model * 3  # 3 models (OpenAI, Claude, Gemini)
            
            # Initialize results containers
            all_results_list = []
            
            # Create tabs for combined view and debug logs
            tab1, tab2 = st.tabs(["Research Results", "Debug Logs"])
            
            with tab1:
                # Create progress bars in the progress tab
                st.markdown("### Processing Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize counters
                completed_runs = 0
                
                # Create a compact view for all models
                for model_name in ["OpenAI", "Claude", "Gemini"]:
                    with st.expander(f"#### {model_name} Runs", expanded=False):
                        # Create placeholders for each run's status
                        run_statuses = {}
                        for run_num in range(1, runs_per_model + 1):
                            run_statuses[run_num] = st.empty()
                            run_statuses[run_num].info(f"Run {run_num}/{runs_per_model}: Waiting...")
                        
                        # Run the model multiple times
                        for run_num in range(1, runs_per_model + 1):
                            run_id = f"{model_name}_run{run_num}"
                            
                            # Update status
                            status_text.text(f"Processing: {model_name} - Run {run_num}/{runs_per_model} ({completed_runs+1}/{total_runs} total)")
                            run_statuses[run_num].info(f"Run {run_num}/{runs_per_model}: In Progress...")
                            
                            # Execute the query without displaying debug output
                            try:
                                # Call the appropriate model
                                if model_name == "OpenAI":
                                    result = call_openai(get_prompt(query), run_id)
                                elif model_name == "Claude":
                                    result = call_anthropic(get_prompt(query), run_id)
                                else:  # Gemini
                                    result = call_google(get_prompt(query), run_id)
                                
                                # Process the result if valid
                                if result:
                                    run_statuses[run_num].success(f"Run {run_num}/{runs_per_model}: Complete ✅")
                                    
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
                                    run_statuses[run_num].error(f"Run {run_num}/{runs_per_model}: Failed ❌ (No valid results)")
                            except Exception as e:
                                run_statuses[run_num].error(f"Run {run_num}/{runs_per_model}: Error ❌ ({str(e)[:50]}...)")
                                add_debug_log(model_name, run_id, f"Exception during processing: {str(e)}", level="ERROR")
                            
                            # Update progress
                            completed_runs += 1
                            progress_bar.progress(completed_runs / total_runs)
                
                # Clear the primary progress indicators when done
                status_text.text(f"✅ Processing complete! {completed_runs}/{total_runs} runs finished")
                
                # Add a separator between progress and results
                st.markdown("---")
                
                # After all processing, continue with results in the same tab
                if all_results_list:
                    # Combine all results
                    all_results = pd.concat(all_results_list, ignore_index=True)
                    
                    # Add normalized URL for grouping
                    all_results["Normalized URL"] = all_results["URL"].apply(normalize_url)
                    
                    # Store in session state for persistence
                    st.session_state.all_results = all_results
                    
                    # Display run summary
                    st.markdown("### Run Summary")
                    summary_cols = st.columns(3)
                    
                    total_companies = all_results["Normalized URL"].nunique()
                    summary_cols[0].metric("Completed Runs", f"{completed_runs}/{total_runs}")
                    summary_cols[1].metric("Unique Companies", f"{total_companies}")
                    summary_cols[2].metric("Data Points", f"{len(all_results)}")
                    
                    # Add download button at the top of the results tab
                    st.markdown("### Download Data")
                    csv = all_results.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Full Dataset as CSV",
                        data=csv,
                        file_name=f"company_research_{query[:20].replace(' ', '_').lower() if query else 'results'}.csv",
                        mime="text/csv",
                        help="Download the complete raw dataset for all runs and models",
                        key="download_button_initial"
                    )
                    
                    # Add model filtering
                    st.markdown("### Filter Results")
                    selected_models = st.multiselect(
                        "Filter by Model",
                        options=["OpenAI", "Claude", "Gemini", "All"],
                        default=["All"],
                        key="results_model_filter_initial"
                    )
                    
                    # Filter results based on selected models
                    filtered_results = all_results
                    if "All" not in selected_models:
                        filtered_results = all_results[all_results["Source Model"].isin(selected_models)]
                    
                    # Calculate the number of filtered runs
                    filtered_runs = len(filtered_results["Source Model"].unique()) * runs_per_model
                    if filtered_runs == 0:  # Avoid division by zero
                        filtered_runs = 1
                    
                    # Group by URL to get stats
                    url_stats = {}
                    
                    for norm_url in filtered_results["Normalized URL"].unique():
                        # Get all entries for this normalized URL
                        url_entries = filtered_results[filtered_results["Normalized URL"] == norm_url]
                        
                        # Calculate metrics across all runs
                        runs_appeared_in = len(url_entries)
                        appearance_pct = (runs_appeared_in / filtered_runs) * 100
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
                    if not stats_df.empty:
                        stats_df = stats_df.sort_values(by=["Visibility Score", "Appearance %", "Average Rank"], 
                                                       ascending=[False, False, True])
                        
                        # Display the consolidated results
                        filter_text = ", ".join(selected_models) if "All" not in selected_models else "All Models"
                        st.markdown(f"### Consolidated Results ({filter_text})")
                        
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
                        
                        # Display raw data for debugging
                        with st.expander("View Raw Data"):
                            st.markdown("### URL Normalization")
                            url_mapping = filtered_results[["URL", "Normalized URL"]].drop_duplicates()
                            st.dataframe(url_mapping)
                            
                            st.markdown("### Complete Dataset")
                            st.dataframe(filtered_results)
                    else:
                        st.warning("No results match the selected filters. Please select different models.")
                else:
                    st.error("All model runs failed to return valid results. Please try again.")
                    st.stop()
            
            # Update the debug logs tab
            with tab2:
                st.markdown("### Debug Logs")
                
                # Add log filtering options
                log_filter_cols = st.columns(3)
                with log_filter_cols[0]:
                    selected_models = st.multiselect(
                        "Filter by Model",
                        options=["OpenAI", "Claude", "Gemini", "All"],
                        default=["All"],
                        key="debug_model_filter_initial"
                    )
                
                with log_filter_cols[1]:
                    selected_levels = st.multiselect(
                        "Filter by Level",
                        options=["INFO", "WARNING", "ERROR", "DEBUG", "All"],
                        default=["All"],
                        key="debug_level_filter_initial"
                    )
                
                with log_filter_cols[2]:
                    clear_logs = st.button("Clear Logs", key="clear_logs_initial")
                    if clear_logs:
                        st.session_state.debug_logs = []
                        st.experimental_rerun()
                
                # Apply filters
                filtered_logs = st.session_state.debug_logs
                if "All" not in selected_models:
                    filtered_logs = [log for log in filtered_logs if log["model"] in selected_models]
                
                if "All" not in selected_levels:
                    filtered_logs = [log for log in filtered_logs if log["level"] in selected_levels]
                
                # Display logs in a table
                if filtered_logs:
                    log_table = []
                    for log in filtered_logs:
                        row = {
                            "Time": log["timestamp"],
                            "Model": log["model"],
                            "Run": log["run_id"],
                            "Level": log["level"],
                            "Message": log["message"]
                        }
                        log_table.append(row)
                    
                    st.dataframe(log_table)
                    
                    # Always show log content
                    st.markdown("### Log Content")
                    for i, log in enumerate(filtered_logs):
                        if log.get("data"):
                            with st.expander(f"{log['timestamp']} - {log['model']} - {log['message']}"):
                                st.code(log["data"])
                else:
                    st.info("No logs matching the selected filters.")
    else:
        st.warning("Please enter a research query.")
        
# Display results if they exist in session state (for when filters change)
if st.session_state.has_run_query and st.session_state.all_results is not None:
    # Create tabs layout again (needed since we're outside the button click)
    tab1, tab2 = st.tabs(["Research Results", "Debug Logs"])
    
    with tab1:
        # Add processing complete message
        st.success(f"Processing complete! Using {st.session_state.runs_per_model} runs per model.")
        
        # Access the stored results
        all_results = st.session_state.all_results
        
        # Display run summary
        st.markdown("### Run Summary")
        summary_cols = st.columns(3)
        
        runs_per_model = st.session_state.runs_per_model
        total_runs = runs_per_model * len(all_results["Source Model"].unique())
        total_companies = all_results["Normalized URL"].nunique()
        completed_runs = len(all_results) // 10  # Estimate based on typical results
        
        summary_cols[0].metric("Completed Runs", f"{completed_runs}/{total_runs}")
        summary_cols[1].metric("Unique Companies", f"{total_companies}")
        summary_cols[2].metric("Data Points", f"{len(all_results)}")
        
        # Add download button at the top of the results tab
        st.markdown("### Download Data")
        csv = all_results.to_csv(index=False)
        st.download_button(
            label="💾 Download Full Dataset as CSV",
            data=csv,
            file_name=f"company_research_{query[:20].replace(' ', '_').lower() if query else 'results'}.csv",
            mime="text/csv",
            help="Download the complete raw dataset for all runs and models",
            key="download_button_filtered"
        )
        
        # Add model filtering
        st.markdown("### Filter Results")
        selected_models = st.multiselect(
            "Filter by Model",
            options=["OpenAI", "Claude", "Gemini", "All"],
            default=["All"],
            key="results_model_filter_persistent"
        )
        
        # Filter results based on selected models
        filtered_results = all_results
        if "All" not in selected_models:
            filtered_results = all_results[all_results["Source Model"].isin(selected_models)]
        
        # Calculate the number of filtered runs
        filtered_runs = len(filtered_results["Source Model"].unique()) * runs_per_model
        if filtered_runs == 0:  # Avoid division by zero
            filtered_runs = 1
        
        # Group by URL to get stats
        url_stats = {}
        
        for norm_url in filtered_results["Normalized URL"].unique():
            # Get all entries for this normalized URL
            url_entries = filtered_results[filtered_results["Normalized URL"] == norm_url]
            
            # Calculate metrics across all runs
            runs_appeared_in = len(url_entries)
            appearance_pct = (runs_appeared_in / filtered_runs) * 100
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
        if not stats_df.empty:
            stats_df = stats_df.sort_values(by=["Visibility Score", "Appearance %", "Average Rank"], 
                                           ascending=[False, False, True])
            
            # Display the consolidated results
            filter_text = ", ".join(selected_models) if "All" not in selected_models else "All Models"
            st.markdown(f"### Consolidated Results ({filter_text})")
            
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
            
            # Display raw data for debugging
            with st.expander("View Raw Data"):
                st.markdown("### URL Normalization")
                url_mapping = filtered_results[["URL", "Normalized URL"]].drop_duplicates()
                st.dataframe(url_mapping)
                
                st.markdown("### Complete Dataset")
                st.dataframe(filtered_results)
        else:
            st.warning("No results match the selected filters. Please select different models.")
    
    # Show debug logs tab content
    with tab2:
        st.markdown("### Debug Logs")
        
        # Add log filtering options
        log_filter_cols = st.columns(3)
        with log_filter_cols[0]:
            selected_models = st.multiselect(
                "Filter by Model",
                options=["OpenAI", "Claude", "Gemini", "All"],
                default=["All"],
                key="debug_model_filter_persistent"
            )
        
        with log_filter_cols[1]:
            selected_levels = st.multiselect(
                "Filter by Level",
                options=["INFO", "WARNING", "ERROR", "DEBUG", "All"],
                default=["All"],
                key="debug_level_filter_persistent"
            )
        
        with log_filter_cols[2]:
            clear_logs = st.button("Clear Logs", key="clear_logs_persistent")
            if clear_logs:
                st.session_state.debug_logs = []
                st.experimental_rerun()
        
        # Apply filters
        filtered_logs = st.session_state.debug_logs
        if "All" not in selected_models:
            filtered_logs = [log for log in filtered_logs if log["model"] in selected_models]
        
        if "All" not in selected_levels:
            filtered_logs = [log for log in filtered_logs if log["level"] in selected_levels]
        
        # Display logs in a table
        if filtered_logs:
            log_table = []
            for log in filtered_logs:
                row = {
                    "Time": log["timestamp"],
                    "Model": log["model"],
                    "Run": log["run_id"],
                    "Level": log["level"],
                    "Message": log["message"]
                }
                log_table.append(row)
            
            st.dataframe(log_table)
            
            # Always show log content
            st.markdown("### Log Content")
            for i, log in enumerate(filtered_logs):
                if log.get("data"):
                    with st.expander(f"{log['timestamp']} - {log['model']} - {log['message']}"):
                        st.code(log["data"])
        else:
            st.info("No logs matching the selected filters.")
