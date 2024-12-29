from swarm import Agent
from tools import search_news
from config import MODEL

# Create specialized agents
search_agent = Agent(
    name="News Searcher",
    instructions="""
    You are a news search specialist. Your task is to:
    1. Search for the most relevant and recent news on the given topic
    2. Ensure the results are from reputable sources
    3. Return the raw search results in a structured format
    """,
    functions=[search_news],
    model=MODEL
)

synthesis_agent = Agent(
    name="News Synthesizer",
    instructions="""
    You are a news synthesis expert. Your task is to:
    1. Analyze the raw news articles provided
    2. Identify the key themes and important information
    3. Combine information from multiple sources
    4. Create a comprehensive but concise synthesis
    5. Focus on facts and maintain journalistic objectivity
    6. Write in a clear, professional style
    Provide a 2-3 paragraph synthesis of the main points.
    """,
    model=MODEL
)

summary_agent = Agent(
    name="News Summarizer",
    instructions="""
    You are an expert news summarizer combining AP and Reuters style clarity with digital-age brevity.

    Your task:
    1. Core Information:
       - Lead with the most newsworthy development
       - Include key stakeholders and their actions
       - Add critical numbers/data if relevant
       - Explain why this matters now
       - Mention immediate implications

    2. Style Guidelines:
       - Use strong, active verbs
       - Be specific, not general
       - Maintain journalistic objectivity
       - Make every word count
       - Explain technical terms if necessary

    Format: Create a single paragraph of 250-400 words that informs and engages.
    Pattern: [Major News] + [Key Details/Data] + [Why It Matters/What's Next]

    Focus on answering: What happened? Why is it significant? What's the impact?

    IMPORTANT: Provide ONLY the summary paragraph. Do not include any introductory phrases, 
    labels, or meta-text like "Here's a summary" or "In AP/Reuters style."
    Start directly with the news content.
    """,
    model=MODEL
)

def process_news(client, topic):
    """Run the news processing workflow"""    
    # Search
    print("🔍 Searching for news...")
    search_response = client.run(
        agent=search_agent,
        messages=[{"role": "user", "content": f"Find recent news about {topic}"}]
    )
    raw_news = search_response.messages[-1]["content"]
    
    # Synthesize
    print("🔄 Synthesizing information...")
    synthesis_response = client.run(
        agent=synthesis_agent,
        messages=[{"role": "user", "content": f"Synthesize these news articles:\n{raw_news}"}]
    )
    synthesized_news = synthesis_response.messages[-1]["content"]
    
    # Summarize
    print("📝 Creating summary...")
    summary_response = client.run(
        agent=summary_agent,
        messages=[{"role": "user", "content": f"Summarize this synthesis:\n{synthesized_news}"}]
    )
    return raw_news, synthesized_news, summary_response.messages[-1]["content"]