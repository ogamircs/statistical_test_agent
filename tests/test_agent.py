"""Test the LangChain A/B Testing Agent"""

import os
from dotenv import load_dotenv

load_dotenv()

from agent import ABTestingAgent

def test_agent():
    print("Testing LangChain A/B Testing Agent...")
    print("=" * 60)

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return False

    print(f"API Key found: {api_key[:10]}...{api_key[-4:]}")

    # Initialize agent
    print("\n1. Initializing agent...")
    agent = ABTestingAgent()
    print("   Agent initialized successfully!")

    # Test loading CSV
    print("\n2. Testing CSV loading...")
    response = agent.run("Load the CSV file at path: sample_ab_data.csv")
    print(f"   Response preview: {response[:200]}...")

    # Test setting column mapping
    print("\n3. Testing column mapping...")
    response = agent.run(
        "Set the column mapping: group column is 'experiment_group', "
        "effect value is 'effect_value', segment is 'customer_segment'"
    )
    print(f"   Response preview: {response[:200]}...")

    # Test setting group labels
    print("\n4. Testing group labels...")
    response = agent.run("The treatment label is 'treatment' and control label is 'control'")
    print(f"   Response preview: {response[:200]}...")

    # Test running analysis
    print("\n5. Testing full analysis...")
    response = agent.run("Run a full A/B test analysis for all segments")
    print(f"   Response preview: {response[:500]}...")

    # Test data query
    print("\n6. Testing data query...")
    response = agent.run("What is the average effect value for Premium customers?")
    print(f"   Response preview: {response[:200]}...")

    print("\n" + "=" * 60)
    print("LangChain Agent: ALL TESTS PASSED!")
    return True

if __name__ == "__main__":
    test_agent()
