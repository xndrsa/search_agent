import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler 
from googlesearch import search


# Constants
MAX_ITERATIONS = 5      
MAX_SEARCH_RESULTS = 3  
GROQ_API = os.getenv("GROQ_API")


PROMPT_TEMPLATES = {
    "Product Search": """You are a product analyst. Analyze this product: {query}
    
    TASK:
    1. Extract product details from markdown content
    2. Identify: full product name, category, price, and URL
    3. Format response precisely
    
    Available information from search:
    {search_results}
    
    INSTRUCTIONS:
    - IMPORTANT: Only use prices that appear directly in the search results
    - If no price is found, use "Price not found" instead of guessing
    - Use official retailer URL when available
    - Format response exactly as shown below
    - No hallucination or guessing of information
    - If information is not found, use "Not found" 
    
    OUTPUT FORMAT:
    [Product Name]<||>[Category]<||>[Price]<||>[URL]""",
    
    "Location Info": """You are a geographic analyst. Analyze this query: {query}
    
    Your task:
    1. Determine the type of location (country, state, city, etc.)
    2. Search for accurate population and area data
    3. Format response precisely
    
    Available information from search:
    {search_results}
    
    Rules:
    - For US locations, specify if it's a state or city
    - Population should be the most recent available
    - Area should include the unit (km² or sq mi)
    - Be accurate with administrative divisions""",
    
    "Company Details": """You are a business analyst. Analyze this query: {query}
    
    Your task:
    1. Research company information
    2. Find company website and details
    3. Format response precisely
    
    Available information from search:
    {search_results}
    
    Rules:
    - Look for company website first
    - Extract industry from company description or website
    - Include headquarters if found
    - If information is not available, use "Not found"
    - NO HALLUCINATION - only use found information
    
    OUTPUT FORMAT:
    [Company Name]<||>[Industry]<||>[Revenue/Status]<||>[Location/Website]""",
    
    "Custom": """You are an information analyst. Analyze this query: {query}
    
    Your task:
    1. Search for relevant information
    2. Extract specific details matching the requested format
    3. Format response precisely
    
    Available information from search:
    {search_results}
    
    Rules:
    - Focus on accuracy and relevance
    - Include source URLs when available
    - Follow the exact format requested"""
}

def create_formatted_prompt(query: str, template_option: str, format_input: str) -> str:
    """Create a formatted prompt based on the selected template"""
    base_prompt = PROMPT_TEMPLATES.get(template_option, PROMPT_TEMPLATES["Custom"])
    formatted_prompt = base_prompt.format(
        query=query,
        search_results="{search_results}" 
    )
    return f"{formatted_prompt}\nRequired output format: {format_input}"

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

def format_to_markdown(search_results: str) -> str:
    """Convert search results to markdown format"""
    markdown_lines = []
    current_source = None
    
    for line in search_results.split('\n'):
        line = line.strip()
        if line.startswith('Source:'):
            current_source = line.replace('Source:', '').strip()

            domain = current_source.split('/')[2] if '/' in current_source else current_source
            markdown_lines.append(f"\n### [{domain}]({current_source})")
        elif current_source and line:

            if 'RYOBI' in line or 'PCL' in line or '$' in line:
                markdown_lines.append(f"- {line}")
    
    return "\n".join(markdown_lines)

def google_search_tool(query: str, num: int = MAX_SEARCH_RESULTS) -> str:
    """Enhanced Google search with markdown formatting and content extraction"""
    try:
        if "company" in query.lower() or "llc" in query.lower():
            company_name = query.replace("llc", "").strip()
            company_query = f"{company_name} company website OR {company_name}.com"
            specific_query = f"{company_name} company (industry OR revenue OR headquarters)"
            
            results = list(search(company_query, num=2, stop=2, pause=2.0, lang='en'))
            results.extend(list(search(specific_query, num=2, stop=2, pause=2.0, lang='en')))
        else:
            results = list(search(query, num=num, stop=num, pause=2.0, lang='en'))


        formatted_results = []
        for url in results:
            try:
                domain = url.split('/')[2]
                info = []
                
                if domain.endswith('.com') or domain.endswith('.org'):
                    info.append(f"Company Website: {domain}")
                
                formatted_results.append(f"""
                Source: {url}
                Website: {domain}
                Info: {' '.join(info)}
                """)
            except Exception as e:
                continue

        markdown_content = format_to_markdown("\n".join(formatted_results))
        return markdown_content
    except Exception as e:
        return f"Error in Google Search: {str(e)}"


#status_placeholder = st.empty()
google_tool = Tool(
    name="Google Search",
    func=google_search_tool,
    description="Busca información en Google."
)

st.title("🔎Web Search")

#def update_status(message):
    #"""Update status message with timestamp"""
    #status_placeholder.info(f"{time.strftime('%H:%M:%S')} - {message}")

with st.sidebar:
    st.header("Response Format Configuration")
    
    template_option = st.selectbox(
        "Choose a template or create custom format",
        ["Custom", "Product Search", "Location Info", "Company Details"]
    )
    
    templates = {
        "Product Search": {
            "format": "product_name<||>category<||>price<||>url",
            "example": "DW6SS is a dishwasher<||>Appliances<||>$299.99<||>https://example.com",
            "description": "For searching product information"
        },
        "Location Info": {
            "format": "location_name<||>location_type<||>Country<||>population<||>area",
            "example": "Texas<||>US State<||>United States of America<||>30.5 million<||>695,662 km²",
            "description": "For geographic information. Location type can be: Country, US State, City, etc."
        },
        "Company Details": {
            "format": "company<||>industry<||>revenue<||>headquarters",
            "example": "Apple<||>Technology<||>$365.8B<||>Cupertino, CA",
            "description": "For company information"
        }
    }
    
    if template_option == "Custom":
        format_input = st.text_input(
            "Enter your custom format (use <||> as separator):",
            "field1<||>field2<||>field3"
        )
        example_input = st.text_input(
            "Enter an example of expected output:",
            "value1<||>value2<||>value3"
        )
    else:
        format_input = templates[template_option]["format"]
        example_input = templates[template_option]["example"]
        st.info(templates[template_option]["description"])
    
    st.divider()
    st.caption("Format Guide:")
    st.code(f"Format: {format_input}\nExample: {example_input}")

    #update_status("Ready to search!")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

def create_search_agent(tools, llm):
    return initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        handle_parsing_errors=True,
        max_iterations=MAX_ITERATIONS,
        early_stopping_method='generate'
    )

if prompt := st.chat_input(placeholder="Enter Your Question Here"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=GROQ_API, model_name="gemma2-9b-it", streaming=True)
    tools = [arxiv, wiki, google_tool]

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            formatted_prompt = create_formatted_prompt(
                prompt,
                template_option,
                format_input
            )

            search_agent = create_search_agent(tools, llm)
            try:
                response = search_agent.run([{"role": "user", "content":formatted_prompt}], callbacks=[st_cb])
            except Exception as e:
                if "maximum iterations" in str(e).lower():
                    response = "Search stopped due to too many iterations. Please try a more specific query."
                else:
                    raise e
            
            if "<||>" not in response:
                fields = format_input.split("<||>")
                response = "<||>".join([response] + ["N/A"] * (len(fields) - 1))
            
            st.session_state.messages.append({'role': 'assistant', "content": response})
            st.write(response)
            

            #update_status("Search completed successfully!")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            #update_status("Search failed!")

