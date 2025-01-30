import json
import os

from dotenv import load_dotenv
from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langgraph.graph import START, END, StateGraph

from assistant.configuration import Configuration, SearchAPI
from assistant.utils import deduplicate_and_format_sources, tavily_search, format_sources, perplexity_search
from assistant.state import SummaryState, SummaryStateInput, SummaryStateOutput
from assistant.prompts import query_writer_instructions, summarizer_instructions, reflection_instructions


def llm_model(conf: Configuration, temperature: int = 0, json_mode: bool = False):
    load_dotenv()

    provider = conf.model_provider
    if provider == "ollama":
        return ChatOllama(model=conf.model_name, temperature=temperature, format="json" if json_mode else None)
    if provider == "openai":
        return ChatOpenAI(
            model=conf.model_name, temperature=temperature, model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {},
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    if provider == "azure-ai":
        return AzureAIChatCompletionsModel(
            model_name=conf.model_name, temperature=temperature, model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {},
            endpoint=os.getenv("AZURE_AI_ENDPOINT"),
            credential=os.getenv("AZURE_AI_CREDENTIAL"),
            api_version="2023-05-15",
        )
    if provider == "openrouter":
        return ChatOpenAI(
            model=conf.model_name, temperature=temperature, model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {},
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL"),
            extra_body={"provider": {"sort": os.getenv("OPENROUTER_PROVIDER_SORT")}},
        )


# Nodes   
def generate_query(state: SummaryState, config: RunnableConfig):
    """ Generate a query for web search """
    configurable = Configuration.from_runnable_config(config)
    
    # Format the prompt
    query_writer_instructions_formatted = query_writer_instructions.format(research_topic=state.research_topic, language=configurable.language)

    # Generate a query
    llm_json_mode = llm_model(configurable, json_mode=True)
    result = llm_json_mode.invoke(
        [SystemMessage(content=query_writer_instructions_formatted),
        HumanMessage(content=f"Generate a query for web search:")]
    )   
    query = json.loads(result.content)
    
    return {"search_query": query['query']}

def web_research(state: SummaryState, config: RunnableConfig):
    """ Gather information from the web """
    
    # Configure 
    configurable = Configuration.from_runnable_config(config)

    # Handle both cases for search_api:
    # 1. When selected in Studio UI -> returns a string (e.g. "tavily")
    # 2. When using default -> returns an Enum (e.g. SearchAPI.TAVILY)
    if isinstance(configurable.search_api, str):
        search_api = configurable.search_api
    else:
        search_api = configurable.search_api.value

    # Search the web
    if search_api == "tavily":
        search_results = tavily_search(state.search_query, include_raw_content=True, max_results=configurable.max_search_results)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=configurable.max_tokens_per_source, include_raw_content=True)
    elif search_api == "perplexity":
        search_results = perplexity_search(state.search_query, state.research_loop_count)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=configurable.max_tokens_per_source, include_raw_content=False)
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")
        
    return {"sources_gathered": [format_sources(search_results)], "research_loop_count": state.research_loop_count + 1, "web_research_results": [search_str]}

def summarize_sources(state: SummaryState, config: RunnableConfig):
    """ Summarize the gathered sources """
    
    # Existing summary
    existing_summary = state.running_summary

    # Most recent web research
    most_recent_web_research = state.web_research_results[-1]

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"Extend the existing summary: {existing_summary}\n\n"
            f"Include new search results: {most_recent_web_research} "
            f"That addresses the following topic: {state.research_topic}"
        )
    else:
        human_message_content = (
            f"Generate a summary of these search results: {most_recent_web_research} "
            f"That addresses the following topic: {state.research_topic}"
        )

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)
    llm = llm_model(configurable)
    result = llm.invoke(
        [SystemMessage(content=summarizer_instructions.format(additional_requirements=state.additional_requirements, language=configurable.language)),
        HumanMessage(content=human_message_content)]
    )
    print(result)

    running_summary = result.content

    # TODO: This is a hack to remove the <think> tags w/ Deepseek models 
    # It appears very challenging to prompt them out of the responses 
    while "<think>" in running_summary and "</think>" in running_summary:
        start = running_summary.find("<think>")
        end = running_summary.find("</think>") + len("</think>")
        running_summary = running_summary[:start] + running_summary[end:]

    return {"running_summary": running_summary}

def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """ Reflect on the summary and generate a follow-up query """

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = llm_model(configurable, json_mode=True)
    result = llm_json_mode.invoke(
        [SystemMessage(content=reflection_instructions.format(research_topic=state.research_topic, additional_requirements=state.additional_requirements, language=configurable.language)),
        HumanMessage(content=f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {state.running_summary}")]
    )   
    follow_up_query = json.loads(result.content)

    # Get the follow-up query
    query = follow_up_query.get('follow_up_query')

    # JSON mode can fail in some cases
    if not query:

        # Fallback to a placeholder query
        return {"search_query": f"Tell me more about {state.research_topic}"}

    # Update search query with follow-up query
    return {"search_query": follow_up_query['follow_up_query']}

def finalize_summary(state: SummaryState):
    """ Finalize the summary """
    
    # Format all accumulated sources into a single bulleted list
    all_sources = "\n".join(source for source in state.sources_gathered)
    state.running_summary = f"## Summary\n\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    return {"running_summary": state.running_summary}

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "web_research"]:
    """ Route the research based on the follow-up query """

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configurable.max_web_research_loops:
        return "web_research"
    else:
        return "finalize_summary" 
    
# Add nodes and edges 
builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

# Add edges
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "web_research")
builder.add_edge("web_research", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("finalize_summary", END)

graph = builder.compile()
