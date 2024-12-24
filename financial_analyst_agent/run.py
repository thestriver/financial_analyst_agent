#!/usr/bin/env python
from dotenv import load_dotenv
import os
import yfinance as yf
from datetime import datetime

load_dotenv()

print("OpenAI Key loaded:", bool(os.getenv("OPENAI_API_KEY")))
print("Serper Key loaded:", bool(os.getenv("SERPER_API_KEY")))

from naptha_sdk.schemas import AgentRunInput
from naptha_sdk.utils import get_logger
from financial_analyst_agent.schemas import InputSchema, FinancialAnalysisInput
from typing import Dict, Any, List

from crewai import Agent, Task, Crew, Process
from textwrap import dedent
from langchain_openai import ChatOpenAI
from crewai_tools import SerperDevTool, WebsiteSearchTool

logger = get_logger(__name__)

class FinancialAnalyzer:
    def __init__(self, module_run: AgentRunInput):
        self.module_run = module_run
        self.llm_config = module_run.agent_deployment.agent_config.llm_config
        self.setup_tools()
        self.setup_agents()

    def setup_tools(self):
        """Initialize analysis tools"""
        self.search_tool = SerperDevTool()
        self.web_tool = WebsiteSearchTool()

    def get_financial_data(self, symbols: List[str], period: str) -> Dict:
        """Fetch financial data using yfinance"""
        data = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            data[symbol] = {
                'info': ticker.info,
                'income_stmt': ticker.income_stmt,
                'balance_sheet': ticker.balance_sheet,
                'calendar': ticker.calendar,
                'history': ticker.history(period=period)
            }
        return data

    def setup_agents(self):
        """Initialize the financial analysis agents with CrewAI"""
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        
        self.data_analyst = Agent(
            role="Financial Data Analyst",
            goal="Analyze financial statements and metrics",
            backstory="""Expert financial analyst specializing in quantitative analysis 
                        of company financial statements and market data.""",
            tools=[],
            llm=llm,
            verbose=True
        )

        self.market_researcher = Agent(
            role="Market Research Analyst",
            goal="Research market trends and company news",
            backstory="""Market research specialist focused on analyzing industry trends, 
                        company news, and market sentiment.""",
            tools=[self.search_tool, self.web_tool],
            llm=llm,
            verbose=True
        )

        self.report_compiler = Agent(
            role="Financial Report Compiler",
            goal="Synthesize financial analysis and market research",
            backstory="""Financial report specialist skilled at combining quantitative 
                        and qualitative analysis into actionable insights.""",
            tools=[],
            llm=llm,
            verbose=True
        )

    def create_tasks(self, financial_data: Dict, analysis_input: FinancialAnalysisInput) -> List[Task]:
        """Create analysis tasks for the crew"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        task_1 = Task(
            description=f"""
                Analyze financial data for {', '.join(analysis_input.ticker_symbols)}:
                1. Review key financial metrics
                2. Analyze trends in financial statements
                3. Identify significant changes or anomalies
                4. Calculate important financial ratios
                Depth: {analysis_input.analysis_type}
            """,
            expected_output="Detailed financial analysis report with key metrics and trends",
            agent=self.data_analyst
        )

        task_2 = Task(
            description=f"""
                Research market conditions and news for {', '.join(analysis_input.ticker_symbols)}:
                1. Analyze recent news and developments
                2. Identify market trends
                3. Research competitor activities
                4. Gather analyst opinions and forecasts
            """,
            expected_output="Comprehensive market research report with news analysis and trends",
            agent=self.market_researcher,
            context=[task_1]
        )

        task_3 = Task(
            description="""
                Compile comprehensive financial report:
                1. Synthesize financial analysis and market research
                2. Identify key insights and recommendations
                3. Create executive summary
                4. Highlight risks and opportunities
            """,
            expected_output="Final consolidated report with insights and recommendations",
            agent=self.report_compiler,
            context=[task_1, task_2]
        )

        return [task_1, task_2, task_3]

    def analyze(self, analysis_input: FinancialAnalysisInput) -> Dict[str, Any]:
        """Execute the financial analysis workflow"""
        financial_data = self.get_financial_data(
            analysis_input.ticker_symbols,
            analysis_input.time_period
        )
        
        tasks = self.create_tasks(financial_data, analysis_input)
        
        crew = Crew(
            agents=[self.data_analyst, self.market_researcher, self.report_compiler],
            tasks=tasks,
            verbose=True,
            process=Process.sequential
        )

        result = crew.kickoff()
        
        try:
            return {
                "final_report": result.raw,
                "total_tokens_used": result.token_usage.total_tokens if result.token_usage else 0
            }
        except Exception as e:
            logger.error(f"Failed to process analysis results: {str(e)}")
            return {
                "final_report": "Failed to generate report",
                "total_tokens_used": 0
            }

def run(module_run: AgentRunInput):
    """Main entry point for the financial analyzer agent"""
    analyzer = FinancialAnalyzer(module_run)
    
    if isinstance(module_run.inputs, dict):
        input_params = InputSchema(**module_run.inputs)
    else:
        input_params = module_run.inputs
        
    analysis_results = analyzer.analyze(input_params.analysis_input)
    return analysis_results

if __name__ == "__main__":
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import load_agent_deployments

    naptha = Naptha()

    # Test input parameters
    input_params = InputSchema(
        analysis_input=FinancialAnalysisInput(
            ticker_symbols=["GOOG", "AMZN"],
            analysis_type="brief",
            time_period="1y",
            specific_metrics=["PE", "Revenue Growth", "Profit Margins"]
        ),
        max_news_sources=1
    )

    # Load agent deployment configuration
    agent_deployments = load_agent_deployments(
        "financial_analyst_agent/configs/agent_deployments.json", 
        load_persona_data=False, 
        load_persona_schema=False
    )

    # Create agent run input
    agent_run = AgentRunInput(
        inputs=input_params,
        agent_deployment=agent_deployments[0],
        consumer_id=naptha.user.id,
    )

    # Execute the agent
    response = run(agent_run)
    print("Analysis Results:", response)