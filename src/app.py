import os
from typing import Annotated, Any, Dict, List
from typing_extensions import TypedDict

from fastapi import FastAPI, HTTPException, status
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# 1. FastAPI Initialization & Configuration
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Liquor Sales Forecasting Agent API",
    description="Extracts date/county from raw query, then runs a 30-day parallel forecast.",
    version="3.0.0",
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_FALLBACK_KEY_IF_NEEDED")
CLOUD_RUN_URL = os.getenv(
    "CLOUD_RUN_URL",
    "https://licor-forecaster-1098523474218.us-central1.run.app/predict",
)

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)


# -----------------------------------------------------------------------------
# 2. Pydantic Schemas & Graph State
# -----------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    user_intent: str = Field(
        ...,
        example="Give me a 30 day outlook for SIOUX starting on 2026-06-20",
        description="The raw natural language query from the user.",
    )


class ForecastResponse(BaseModel):
    success: bool
    final_message: str
    target_date: str = None
    county: str = None
    forecast_results: List[Dict[str, Any]] = []
    error: str = None


# 1. Define the Shared Agent State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    raw_user_intent: str
    feature_payload: Dict[str, Any]
    forecast_results: List[float]
    error_message: str


# Initialize OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,  # Note: Keep this secure in production!
)


# 2. Node: Extract and construct features using LLM structured outputs
def extract_features_node(state: AgentState):
    user_query = state["raw_user_intent"]

    # Simulated output matching the Pydantic schema structure
    mock_payload = {
        "features": [
            {
                "county": "POLK",
                "bottles": 500,
                "liters": 375.0,
                "transactions": 40,
                "year": 2026,
                "month": 6,
                "day": 20,
                "dayofweek": 5,
                "dayofyear": 171,
                "weekofyear": 25,
                "is_weekend": 1,
                "is_month_start": 0,
                "is_month_end": 0,
                "is_quarter_start": 0,
                "is_quarter_end": 0,
                "y_lag_1": 150.5,
                "y_lag_7": 142.0,
                "y_lag_14": 138.2,
                "y_lag_30": 160.0,
                "y_rolling_mean_7": 145.2,
                "y_rolling_std_7": 12.1,
                "y_rolling_mean_14": 141.8,
                "y_rolling_std_14": 14.5,
                "y_rolling_mean_30": 151.0,
                "y_rolling_std_30": 18.2,
            }
        ]
    }
    return {"feature_payload": mock_payload, "error_message": ""}


# 3. Node: Validate data before wasting an API call
def validate_payload_node(state: AgentState):
    payload = state.get("feature_payload", {}) or {}
    features = payload.get("features", [{}])[0]

    if features.get("month", 0) < 1 or features.get("month", 0) > 12:
        return {"error_message": "Invalid month value parsed."}
    return {"error_message": ""}


# 4. Node: Query Cloud Run Endpoint
def call_cloud_run_node(state: AgentState):
    if state.get("error_message"):
        return {}

    cloud_run_url = "https://licor-forecaster-1098523474218.us-central1.run.app/predict"
    try:
        response = requests.post(
            cloud_run_url, json=state["feature_payload"], timeout=30
        )
        response.raise_for_status()
        return {"forecast_results": response.json()["forecast"]}
    except Exception as e:
        return {"error_message": f"Cloud Run API Error: {str(e)}"}


# 5. Node: Translate arrays to human-readable insights
def narrate_results_node(state: AgentState):
    if state.get("error_message"):
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Execution halted: {state['error_message']}",
                }
            ]
        }

    forecast = state["forecast_results"]

    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful retail sales expert summarizing forecasts.",
                },
                {
                    "role": "user",
                    "content": f"The model projected sales at {forecast[0]}. Summarize this briefly for the manager.",
                },
            ],
        )
        summary = response.choices[0].message.content
    except Exception as e:
        summary = f"I calculated the forecast value as {forecast[0]}, but I encountered an error creating your natural language summary: {str(e)}"

    return {"messages": [{"role": "assistant", "content": summary}]}


# 6. Conditional Routing Logic
def routing_decision(state: AgentState):
    if state.get("error_message"):
        return "extract_features"
    return "call_cloud_run"


# 7. Compile the Graph Correctly
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("extract_features", extract_features_node)
workflow.add_node("validate_payload", validate_payload_node)
workflow.add_node("call_cloud_run", call_cloud_run_node)
workflow.add_node("narrate_results", narrate_results_node)

# Set up clean transitions without duplicates
workflow.add_edge(START, "extract_features")
workflow.add_edge("extract_features", "validate_payload")

# Dynamic routing only leaves from validate_payload
workflow.add_conditional_edges(
    "validate_payload",
    routing_decision,
    {"extract_features": "extract_features", "call_cloud_run": "call_cloud_run"},
)

# Linear transitions for remaining nodes
workflow.add_edge("call_cloud_run", "narrate_results")
workflow.add_edge("narrate_results", END)

sales_agent = workflow.compile()


# -----------------------------------------------------------------------------
# 7. API Endpoints
# -----------------------------------------------------------------------------
@app.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    initial_state = {
        "messages": [],
        "raw_user_intent": request.user_intent,
        "target_date": "",
        "county": "",
        "forecast_results": [],
        "error_message": "",
    }

    try:
        final_state = sales_agent.invoke(initial_state)

        if final_state.get("error_message"):
            return ForecastResponse(
                success=False,
                final_message="Agent pipeline failed.",
                error=final_state["error_message"],
            )

        last_message = (
            final_state["messages"][-1].content
            if final_state.get("messages")
            else "No summary available."
        )

        return ForecastResponse(
            success=True,
            final_message=last_message,
            target_date=final_state.get("target_date"),
            county=final_state.get("county"),
            forecast_results=final_state.get("forecast_results", []),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal system error: {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
