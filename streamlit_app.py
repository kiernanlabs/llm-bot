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
        # Add JSON formatting instructions for Anthropic
        formatted_prompt = add_json_instructions(prompt)
        
        response = anthropic_client.messages.create(
            model=anthropic_model,
            max_tokens=1000,
            messages=[{"role": "user", "content": formatted_prompt}]
        )
        
        add_debug_log(model_name, run_id, "Response received")
        
        # Extract the text content and ensure it's properly formatted
        content = response.content[0].text.strip()
        
        # Log the response content
        add_debug_log(model_name, run_id, "Response preview", data=content[:500] + ('...' if len(content) > 500 else ''))
        add_debug_log(model_name, run_id, "Full response", level="DEBUG", data=content)
        
        # Try to find JSON content within the response
        try:
            # First try direct JSON parsing
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
        with st.spinner("Researching companies using multiple AI models..."):
            # Create layout for different sections
            progress_container = st.container()
            results_container = st.container()
            
            # Set number of runs per model
            runs_per_model = 5
            total_runs = runs_per_model * 3  # 3 models (OpenAI, Claude, Gemini)
            
            # Initialize results containers
            all_results_list = []
            
            # Create tabs for progress, results, and debug logs
            tab1, tab2, tab3 = st.tabs(["Progress", "Results", "Debug Logs"])
            
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
            
            # Update the debug logs tab with all collected logs
            with tab3:
                st.markdown("### Debug Logs")
                
                # Add log filtering options
                log_filter_cols = st.columns(3)
                with log_filter_cols[0]:
                    selected_models = st.multiselect(
                        "Filter by Model",
                        options=["OpenAI", "Claude", "Gemini", "All"],
                        default=["All"]
                    )
                
                with log_filter_cols[1]:
                    selected_levels = st.multiselect(
                        "Filter by Level",
                        options=["INFO", "WARNING", "ERROR", "DEBUG", "All"],
                        default=["All"]
                    )
                
                with log_filter_cols[2]:
                    clear_logs = st.button("Clear Logs")
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
            
            # After all processing, show results in the results tab
            if all_results_list:
                # Combine all results
                all_results = pd.concat(all_results_list, ignore_index=True)
                
                # Add normalized URL for grouping
                all_results["Normalized URL"] = all_results["URL"].apply(normalize_url)
                
                with tab2:
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
                        file_name=f"company_research_{query[:20].replace(' ', '_').lower()}.csv",
                        mime="text/csv",
                        help="Download the complete raw dataset for all runs and models"
                    )
                    
                    # Add model filtering
                    st.markdown("### Filter Results")
                    selected_models = st.multiselect(
                        "Filter by Model",
                        options=["OpenAI", "Claude", "Gemini", "All"],
                        default=["All"],
                        key="results_model_filter"
                    )
                    
                    # Filter results based on selected models
                    filtered_results = all_results
                    if "All" not in selected_models:
                        filtered_results = all_results[all_results["Source Model"].isin(selected_models)]
                    
                    # Calculate the number of filtered runs
                    filtered_runs = len(filtered_results["Source Model"].unique()) * runs_per_model
                    
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
                with tab2:
                    st.error("All model runs failed to return valid results. Please try again.")
                    st.stop()
    else:
        st.warning("Please enter a research query.")
