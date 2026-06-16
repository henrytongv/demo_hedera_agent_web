from flask import Flask, request, jsonify, send_from_directory
import random
import string

import asyncio
import os

from dotenv import load_dotenv
from hedera_agent_kit.langchain.toolkit import HederaLangchainToolkit
from hedera_agent_kit.plugins import core_account_query_plugin
from hedera_agent_kit.shared.configuration import Configuration, Context, AgentMode
from hiero_sdk_python import Client, Network, AccountId, PrivateKey
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic

from coincap_hedera_plugin import conincap_h_plugin

load_dotenv()

app = Flask(__name__, static_folder='.')

# Hedera client setup (Testnet by default)
account_id = AccountId.from_string(os.getenv("ACCOUNT_ID"))
private_key = PrivateKey.from_string(os.getenv("PRIVATE_KEY"))
client = Client(Network(network="testnet"))
client.set_operator(account_id, private_key)

# Prepare Hedera toolkit
hedera_toolkit = HederaLangchainToolkit(
    client=client,
    configuration=Configuration(
        tools=[],  # Empty = load all tools from plugins
        plugins=[
            core_account_query_plugin,
            conincap_h_plugin  # <---- add the plugin here
        ],
        context=Context(
            mode=AgentMode.AUTONOMOUS,
            account_id=str(account_id),
        ),
    ),
)

tools = hedera_toolkit.get_tools()

# llm from Antrophic
llm = ChatAnthropic(
    model="claude-haiku-4-5",
)    

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=MemorySaver(),
    system_prompt="You are a helpful assistant with access to Hedera blockchain tools and plugin tools",
)

@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
async def chat():
    """Handle chat requests and return random responses"""
    data = request.get_json()
    user_message = data.get('message', '')
    
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config={"configurable": {"thread_id": "1"}},
    )

    final_message_content = response["messages"][-1].content
    print("\n--- Agent Response ---")
    print(final_message_content)    
    
    return jsonify({'response': final_message_content})

if __name__ == '__main__':
    app.run(debug=True, port=5000) # add host='0.0.0.0' to listen from all addresses
