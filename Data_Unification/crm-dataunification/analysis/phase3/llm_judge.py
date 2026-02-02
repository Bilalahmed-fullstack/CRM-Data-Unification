import requests
import json

def describe_column(column_name: str) -> str:
    """
    Uses Gemma 3 1B via Ollama to generate a description for a column name.
    """

    prompt = f"""
    You are a data catalog system.

    Task:
    Given a database column name, produce a short, normalized description
    that clearly states the semantic meaning of the column.

    Rules:
    - One sentence only
    - No assumptions about business logic
    - No examples
    - No adjectives like "likely", "probably"
    - Use neutral, technical language
    - Do not mention table names
    - Do not mention data types explicitly

    Column name: {column_name}

    Return only the description sentence.
    """


    response = requests.post(
        "http://localhost:11434/api/generate",
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "model": "gemma3:1b",
            "prompt": prompt,
            "stream": False
        })
    )

    response.raise_for_status()
    result = response.json()

    return result["response"].strip()


print(describe_column("user_created_at"))
print(describe_column("total_order_amount"))
print(describe_column("is_active"))
