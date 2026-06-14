import os
import json
import datetime
import requests
import pandas as pd
from typing import List, Dict, Any
from openai import OpenAI

# Initialize the OpenAI client (or your preferred LLM provider)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Configuration for your Cloud Run URL
CLOUD_RUN_URL = os.environ.get("CLOUD_RUN_URL", "https://your-model-app-uc.a.run.app")

def parse_and_forecast_sales(user_prompt: str, historical_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Agent tool function that parses conversational sales requests, construct feature 
    payloads with proper time features, and submits them to Cloud Run.
    """
    
    # 1. Ask the LLM to extract data into the Pydantic schema structure
    # We pass it a system prompt instructing it how to extract or infer variables.
    system_prompt = f"""
    You are a data-formatting assistant. The user wants to run a sales forecast.
    Extract or infer the feature values from the user's prompt. 
    
    Today's date context is: {datetime.date.today().strftime('%Y-%m-%d')}
    
    If fields like lag features or rolling means are not explicitly provided by the user, 
    look for them in the 'historical_context' or default them to reasonable historical 
    averages based on similar entries.
    
    Return a JSON object matching this schema structure:
    {{
        "features": [
            {{
                "county": "string",
                "bottles": int,
                "liters": float,
                "transactions": int,
                "year": int, "month": int, "day": int, "dayofweek": int, "dayofyear": int, "weekofyear": int,
                "is_weekend": int, "is_month_start": int, "is_month_end": int, "is_quarter_start": int, "is_quarter_end": int,
                "y_lag_1": float, "y_lag_7": float, "y_lag_14": float, "y_lag_30": float,
                "y_rolling_mean_7": float, "y_rolling_std_7": float, 
                "y_rolling_mean_14": float, "y_rolling_std_14": float,
                "y_rolling_mean_30": float, "y_rolling_std_30": float
            }}
        ]
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Prompt: {user_prompt}\nHistorical Context: {json.dumps(historical_context or {})}"}
        ]
    )
    
    # 2. Extract JSON string and validate payload structure
    try:
        payload = json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"Failed to parse text to proper payload schema: {str(e)}"}

    # 3. Hit the FastAPI application hosted on Cloud Run
    try:
        api_endpoint = f"{CLOUD_RUN_URL.rstrip('/')}/predict"
        headers = {"Content-Type": "application/json"}
        
        api_response = requests.post(api_endpoint, json=payload, headers=headers, timeout=30)
        api_response.raise_for_status()
        
        forecast_results = api_response.json()
        
        # Combine input details with forecast results for the final interpretation phase
        return {
            "status": "success",
            "inputs_sent": payload["features"],
            "forecast_outputs": forecast_results["forecast"]
        }
        
    except requests.exceptions.RequestException as e:
        return {"error": f"Cloud Run execution failed: {str(e)}"}


def sales_assistant_agent(user_query: str, history: Dict[str, Any] = None) -> str:
    """
    Main entrypoint for the Sales Assistant. It coordinates the execution 
    of parsing requests, getting model responses, and narrating predictions.
    """
    print(f"🕵️‍♂️ Sales Assistant processing request: '{user_query}'...")
    
    # Step 1: Execute the backend data processing tool
    execution_result = parse_and_forecast_sales(user_query, historical_context=history)
    
    if "error" in execution_result:
        return f"Sorry, I ran into an error pulling that forecast data together: {execution_result['error']}"
        
    # Step 2: Use an LLM to read raw numeric arrays and draft a human explanation
    interpretation_prompt = f"""
    You are an expert Sales Assistant Agent. Review the raw feature parameters sent 
    to our Machine Learning model and the resulting prediction output.
    Explain the results back to the executive/sales manager with practical business insights, 
    callouts about any spikes, or reflections on historical trends.
    
    Model Inputs:
    {json.dumps(execution_result['inputs_sent'], indent=2)}
    
    Model Predictions (Forecast Array):
    {execution_result['forecast_outputs']}
    
    Provide a well-formatted overview containing summaries, recommendations, or warnings 
    regarding stock levels based on these forecasts. Do not show raw JSON to the user.
    """
    
    narrative_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You present machine learning outputs clearly and engagingly to sales staff."},
            {"role": "user", "content": interpretation_prompt}
        ]
    )
    
    return narrative_response.choices[0].message.content