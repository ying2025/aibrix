import json
from typing import List, Any

def load_workload(input_path: str) -> List[Any]:
    try:
        if input_path.endswith(".jsonl"):
            with open(input_path, "r") as file:
                load_struct = [json.loads(line) for line in file]
        else:
            with open(input_path, "r") as file:
                load_struct = json.load(file)
        return load_struct
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {input_path}: {str(e)}")
    except IOError as e:
        raise ValueError(f"Error reading file {input_path}: {str(e)}")
    
    
# Function to wrap the prompt into OpenAI's chat completion message format.
def wrap_prompt_as_chat_message(prompt: str):
    """
    Wrap the prompt into OpenAI's chat completion message format.

    :param prompt: The user prompt to be converted.
    :return: A list containing chat completion messages.
    """
    user_message = {"role": "user", "content": prompt}
    return [user_message]