# 🤖 LangChain ReAct Shopping Assistant Agent

A lightweight, beginner-friendly implementation of a **ReAct (Reason + Act)** AI agent using **LangChain**, **Groq** (`qwen/qwen3.6-27b`), and custom **Python Tools**.

---

## 📌 Overview

This project demonstrates how to build a decision-making AI agent using the **ReAct pattern**. Rather than answering questions directly from memorized training data or guessing numbers, the agent follows a loop of reasoning:

1. **Reason**: Analyzes the user's question and decides which action to take.
2. **Act**: Requests execution of a specific tool with parameter values.
3. **Observe**: Receives output from the tool and repeats until it has enough information to form a final answer.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- A [Groq API Key](https://console.groq.com/)

### Installation

1. **Navigate to the project directory**:
   ```bash
   cd "Langchain-ReAct"
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   # or using standard pip:
   pip install langchain langchain-groq python-dotenv langsmith
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

### Running the Agent

```bash
python main.py
```

---

## 🛠️ How Tools Work (Custom Python Functions vs External APIs)

### 💡 The Core Concept

When starting with LangChain, it is easy to think of a **"Tool"** as an external service or cloud API (like **Tavily** for web searching, Google Maps API, or a database connection).

However, in LangChain:
> **A Tool is simply ANY Python function that you decorate with `@tool` and share with the LLM.**

Whether that tool calls an external web API (like Tavily), queries a SQL database, or just runs 2 lines of local math in Python (`price * 0.75`), **to the LLM, they are all treated identically**.

### 🔄 How Tool Calling Works Under the Hood

Crucial rule to understand: **The LLM (Groq / Qwen / ChatGPT) CANNOT execute code on your machine directly.**

Instead, tool calling is a 5-step communication exchange between your machine and Groq's servers:

```
┌────────────────────────────────┐                 1. Send User Question + Tool Schemas
│                                │ ────────────────────────────────────────────────────────► ┌────────────────────────────────┐
│                                │                                                           │          Groq LLM              │
│                                │ ◄──────────────────────────────────────────────────────── │      (Reasoning Engine)        │
│          Python Script         │                 2. LLM responds: "Please run tool         └────────────────────────────────┘
│         (Your Machine)         │                    `get_product_price(product='keyboard')`"
│                                │
│                                │ ───► 3. Your computer runs local Python function `get_product_price('keyboard')` -> returns 75.0
│                                │
│                                │                 4. Send Tool Result (75.0) back in a `ToolMessage`
│                                │ ────────────────────────────────────────────────────────► ┌────────────────────────────────┐
│                                │                                                           │          Groq LLM              │
│                                │ ◄──────────────────────────────────────────────────────── │                                │
│                                │                 5. LLM decides next step or gives final   └────────────────────────────────┘
└────────────────────────────────┘                    text answer
```

1. **Converting Python to JSON Schemas**: When you decorate a function with `@tool`, LangChain reads the function's **name**, **docstring** (description), and **type annotations**. It converts them into a JSON schema format:
   ```json
   {
     "name": "get_product_price",
     "description": "Look up the price of a product in the catalog.",
     "parameters": { "product": { "type": "string" } }
   }
   ```
2. **Sending Tools to LLM**: `llm.bind_tools(tools)` sends these JSON schemas to Groq alongside your question.
3. **LLM Decision**: The LLM reads the description *"Look up the price of a product"* and realizes: *"To answer 'price of a keyboard', I need to call `get_product_price` with `product='keyboard'`!"*
4. **Local Execution**: The LLM responds with a JSON object asking your Python script to run that function. Your script executes `get_product_price("keyboard")` locally and gets `75.0`.
5. **Feedback Loop**: Your script packages `75.0` into a `ToolMessage` and sends it back to the LLM. The LLM receives the price and proceeds to the next reasoning step.

---

## 📖 Line-by-Line & Function-by-Function Breakdown of `main.py`

Below is a complete, detailed walkthrough of every section and line in [`main.py`](file:///d:/AI%20Engineering/Langchain%20Projects/Langchain-ReAct/main.py):

### 1. Setup & Imports (Lines 1–11)

```python
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool 
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 5
MODEL = "qwen/qwen3.6-27b"
```

- **`load_dotenv()`**: Reads `.env` and loads your `GROQ_API_KEY` into Python's environment variables so the Groq client can authenticate automatically.
- **`init_chat_model`**: LangChain's universal factory function to initialize chat models across providers (e.g. `groq`, `openai`, `anthropic`).
- **`@tool`**: A decorator that transforms standard Python functions into LangChain-compatible tools.
- **Message Classes**:
  - `SystemMessage`: Sets rules, context, and behavior instructions for the LLM.
  - `HumanMessage`: Represents user input.
  - `ToolMessage`: Holds the output returned by a tool execution to pass back to the LLM.
- **`@traceable`**: Optional LangSmith decorator to log and visually trace the agent's execution trajectory.
- **`MAX_ITERATIONS = 5`**: Safety limit on the loop to prevent infinite tool calling.
- **`MODEL = "qwen/qwen3.6-27b"`**: Specifies the exact Groq-hosted model used as the agent's reasoning core.

---

### 2. Defining Custom Tools (Lines 13–32)

#### Tool 1: `get_product_price`
```python
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f" >> Executing get_product_price(product='{product}')")
    prices = {
        "laptop": 1299.99,
        "mouse": 24.99,
        "keyboard": 75.00,
        "monitor": 299.99,
    }
    return prices.get(product.lower(), 0.0)
```
- **Docstring (`"""Look up the price..."""`)**: Crucial for tool calling. The LLM reads this text description to know what the tool is used for.
- **Function Logic**: Looks up a product name in a local dictionary and returns its price as a float.

#### Tool 2: `apply_discount`
```python
@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"   >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 15, "gold": 25}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price*(1-discount/100), 2)
```
- **Docstring**: Explains available discount tiers (`bronze`, `silver`, `gold`).
- **Function Logic**: Takes the original price and discount tier, calculates the final price, and rounds it to 2 decimal places.

---

### 3. Initializing Agent & Tools (Lines 36–42)

```python
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"groq:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
```

- **`tools_dict`**: Creates a dictionary mapping tool names to functions (e.g., `{"get_product_price": get_product_price, "apply_discount": apply_discount}`). This allows looking up tools by string name when the LLM requests a tool call.
- **`init_chat_model(f"groq:{MODEL}", temperature=0)`**: Initializes the Groq LLM with `temperature=0` to ensure deterministic, focused outputs.
- **`llm.bind_tools(tools)`**: Converts the tool functions to JSON schemas and binds them to the model object.

---

### 4. System Prompt & Conversation History (Lines 47–66)

```python
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one." 
            )
        ),
        HumanMessage(content=question),
    ]
```

- **`SystemMessage`**: Enforces strict operational rules. It explicitly instructs the LLM not to guess prices or do math internally, ensuring it delegates computation to the tools.
- **`HumanMessage`**: Appends the user query (e.g., *"What is the price of a keyboard with a gold discount?"*) to the message list.

---

### 5. The ReAct Loop (Lines 68–96)

```python
    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
```

1. **`llm_with_tools.invoke(messages)`**: Sends the entire conversation history to the LLM.
2. **`ai_message.tool_calls`**: Checks if the LLM returned any requested tool calls.
3. **If `not tool_calls`**: The LLM determined it has all necessary data and generated a final text answer. The function prints and returns `ai_message.content`.
4. **If `tool_calls` exist**:
   - Extracts the tool name (`tool_name`), arguments (`tool_args`), and call identifier (`tool_call_id`).
   - Retrieves the tool function from `tools_dict` and runs it (`tool_to_use.invoke(tool_args)`).
   - Appends both `ai_message` (the LLM's request) and `ToolMessage(content=str(observation), tool_call_id=tool_call_id)` (the result) to `messages`.
5. **Repeat**: The loop starts the next iteration, sending updated history to the LLM.

---

### 6. Script Execution Entrypoint (Lines 107–111)

```python
if __name__ == "__main__":
    print("Running Agent")
    print()
    result = run_agent("What is the price of a keyboard with a gold discount?")
    print(result)
```

---

## 📊 Sample Execution Log

```text
Running Agent

Question: What is the price of a keyboard with a gold discount?

============================================================

--- Iteration 1 ---
  [Tool Selected] get_product_price with args: {'product': 'keyboard'}
 >> Executing get_product_price(product='keyboard')
  [Tool Result] 75.0

--- Iteration 2 ---
  [Tool Selected] apply_discount with args: {'discount_tier': 'gold', 'price': 75.0}
   >> Executing apply_discount(price=75.0, discount_tier='gold')
  [Tool Result] 56.25

--- Iteration 3 ---

Final Answer: The price of a keyboard with a gold discount is $56.25.
```

---

## ⚙️ Key Concepts Summary

| Term | Description |
| :--- | :--- |
| `init_chat_model` | Standard LangChain helper to initialize models (e.g. `groq:qwen/qwen3.6-27b`). |
| `@tool` | Decorator converting custom Python functions into LLM-callable tools. |
| `bind_tools` | Attaches tool schemas to the model so the LLM knows how to trigger them. |
| `ToolMessage` | A message sent back to the LLM containing the output of a tool execution. |
| `ReAct Loop` | The iterative pattern of **Prompt -> LLM decision -> Tool execution -> Observation -> Prompt**. |