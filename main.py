from llm_utils_tinker import completion


def main():
    weather_tool_def = {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }

    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    msgs = [{"role": "user", "content": "서울 날씨 알려줘"}]

    print(f"User Query: {msgs[0]['content']}")
    response = completion(model_name, msgs, tools=[weather_tool_def])
    print(f"\nFinal Response: {response.to_dict()}")


if __name__ == "__main__":
    main()
