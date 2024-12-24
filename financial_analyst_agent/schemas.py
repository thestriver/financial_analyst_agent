from pydantic import BaseModel, Field
from typing import Optional, List

class FinancialAnalysisInput(BaseModel):
    ticker_symbols: List[str] = Field(..., description="List of stock ticker symbols to analyze")
    analysis_type: str = Field("brief", description="Type of analysis: 'brief', 'moderate', or 'comprehensive'")
    time_period: Optional[str] = Field("1y", description="Time period for historical data")
    specific_metrics: Optional[List[str]] = Field(None, description="Specific financial metrics to focus on")

class InputSchema(BaseModel):
    analysis_input: FinancialAnalysisInput
    max_news_sources: Optional[int] = Field(5, description="Maximum number of news sources to analyze")