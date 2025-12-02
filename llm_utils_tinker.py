import json
import re
import uuid
import time
import tinker
from tinker import types
from tinker_cookbook.tokenizer_utils import get_tokenizer
from typing import List, Dict, Optional, Any, Union

# ==============================================================================
# 1. Mock Classes for OpenAI Compatibility
# ==============================================================================
# (이전과 동일하므로 생략하지 않고 그대로 유지)

service_client = tinker.ServiceClient()


class MockFunction:
    """Represents a function call within a tool call."""

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class MockToolCall:
    """Represents a tool call object compatible with OpenAI structure."""

    def __init__(self, id: str, func_name: str, func_args: str):
        self.id = id
        self.type = "function"
        self.function = MockFunction(func_name, func_args)


class MockMessage:
    """Represents a chat completion message."""

    def __init__(
        self,
        role: str,
        content: Optional[str],
        tool_calls: Optional[List[MockToolCall]] = None,
    ):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class MockChoice:
    """Represents a single choice in the completion response."""

    def __init__(self, message: MockMessage, finish_reason: str = "stop"):
        self.message = message
        self.finish_reason = finish_reason


class MockUsage:
    """Represents token usage statistics."""

    def __init__(
        self, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class MockResponse:
    """Represents the final response object compatible with OpenAI API."""

    def __init__(self, choices: List[MockChoice], usage: MockUsage, model: str):
        self.id = f"chatcmpl-{uuid.uuid4()}"
        self.object = "chat.completion"
        self.created = int(time.time())
        self.model = model
        self.choices = choices
        self.usage = usage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "choices": [
                {
                    "message": {
                        "role": c.message.role,
                        "content": c.message.content,
                        "tool_calls": (
                            [
                                {
                                    "id": tc.id,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                    "type": "function",
                                }
                                for tc in c.message.tool_calls
                            ]
                            if c.message.tool_calls
                            else None
                        ),
                    },
                    "finish_reason": c.finish_reason,
                }
                for c in self.choices
            ],
            "usage": self.usage.__dict__,
        }


# ==============================================================================
# 2. Helper Functions (Template & Parsing)
# ==============================================================================
# (이전과 동일)


def apply_qwen_chat_template(
    messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None
) -> str:
    prompt = ""
    if tools:
        prompt += "<|im_start|>system\n"
        if messages and messages[0]["role"] == "system":
            prompt += messages[0]["content"] + "\n\n"

        prompt += (
            "# Tools\n"
            "You are a helpful assistant. You have access to the following functions. "
            "Use them to answer the user's question if necessary.\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        )
        for tool in tools:
            prompt += "\n" + json.dumps(tool, ensure_ascii=False)
        prompt += "\n</tools>\n\n"
        prompt += (
            "To call a function, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'
        )
    else:
        if messages and messages[0]["role"] == "system":
            prompt += f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg.get("content", "") or ""

        if role == "system" and i == 0:
            continue

        if role == "user" or (role == "system" and i > 0):
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        elif role == "assistant":
            prompt += f"<|im_start|>{role}\n{content}"
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", tc)
                    name = fn.get("name")
                    args = fn.get("arguments")
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    prompt += f'\n<tool_call>\n{{"name": "{name}", "arguments": {args}}}\n</tool_call>'
            prompt += "<|im_end|>\n"

        elif role == "tool":
            prev_role = messages[i - 1]["role"] if i > 0 else None
            next_role = messages[i + 1]["role"] if i < len(messages) - 1 else None

            if prev_role != "tool":
                prompt += "<|im_start|>user"
            prompt += f"\n<tool_response>\n{content}\n</tool_response>"
            if next_role != "tool":
                prompt += "<|im_end|>\n"

    prompt += "<|im_start|>assistant\n"
    return prompt


def parse_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    pattern = r"<tool_call>\n(.*?)\n</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    tool_calls = []
    for json_str in matches:
        try:
            tool_calls.append(json.loads(json_str))
        except json.JSONDecodeError:
            continue
    return tool_calls


# ==============================================================================
# 3. Tool Definitions (Example Registry)
# ==============================================================================
# (이전과 동일)


def get_current_weather(location: str, unit: str = "celsius") -> str:
    return json.dumps(
        {
            "location": location,
            "temperature": "22",
            "unit": unit,
            "description": "Sunny",
        },
        ensure_ascii=False,
    )


AVAILABLE_FUNCTIONS = {"get_current_weather": get_current_weather}


# ==============================================================================
# 4. Main Completion Function
# ==============================================================================


def completion(
    model: str,
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> MockResponse:
    """
    Generates a chat completion using the Tinker library for sampling
    and Hugging Face Transformers for tokenization.
    """

    # 1. Initialize Clients
    sampling_client = service_client.create_sampling_client(base_model=model)
    tokenizer = get_tokenizer(model)

    # 2. Extract Parameters
    max_tokens = kwargs.get("max_tokens", 1024)
    temperature = kwargs.get("temperature", 0.5)
    top_p = kwargs.get("top_p", 0.9)
    top_k = kwargs.get("top_k", 50)
    stop = kwargs.get("stop", ["<|im_end|>"])

    # 3. Prepare Loop Variables
    current_messages = [m.copy() for m in messages]
    max_turns = 5

    total_prompt_tokens = 0
    total_completion_tokens = 0

    final_content = ""
    finish_reason = "stop"

    all_accumulated_tool_calls: List[MockToolCall] = []

    # 4. Inference Loop
    for _ in range(max_turns):
        # Generate Prompt
        prompt_text = apply_qwen_chat_template(current_messages, tools=tools)
        input_ids = tokenizer.encode(prompt_text)
        total_prompt_tokens += len(input_ids)

        # Execute Sampling
        prompt_input = types.ModelInput.from_ints(input_ids)
        params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )

        future = sampling_client.sample(
            prompt=prompt_input, sampling_params=params, num_samples=1
        )
        result = future.result()
        generated_text = tokenizer.decode(result.sequences[0].tokens)
        total_completion_tokens += len(result.sequences[0].tokens)

        # Process Output
        tool_calls_data = parse_tool_calls_from_text(generated_text)
        content_part = (
            generated_text.split("<tool_call>")[0].replace("<|im_end|>", "").strip()
        )
        final_content = content_part

        if tool_calls_data:
            current_turn_calls = []
            for tc in tool_calls_data:
                args_val = tc.get("arguments")

                # [수정됨] 한글 깨짐 방지를 위해 ensure_ascii=False 추가
                args_str = (
                    json.dumps(args_val, ensure_ascii=False)
                    if isinstance(args_val, dict)
                    else str(args_val)
                )

                mock_call = MockToolCall(
                    id=f"call_{uuid.uuid4()}",
                    func_name=tc.get("name"),
                    func_args=args_str,
                )
                current_turn_calls.append(mock_call)

            all_accumulated_tool_calls.extend(current_turn_calls)

            current_messages.append(
                {
                    "role": "assistant",
                    "content": content_part,
                    "tool_calls": [{"function": tc} for tc in tool_calls_data],
                }
            )

            for tc in tool_calls_data:
                func_name = tc.get("name")
                args = tc.get("arguments")

                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                if func_name in AVAILABLE_FUNCTIONS:
                    try:
                        # 실행 결과 생성 시에도 한글 처리가 필요할 수 있습니다.
                        # (Mock Function들은 이미 처리되어 있다고 가정)
                        res = AVAILABLE_FUNCTIONS[func_name](**args)
                    except Exception as e:
                        res = f"Error executing {func_name}: {str(e)}"

                    current_messages.append(
                        {"role": "tool", "name": func_name, "content": res}
                    )
                else:
                    current_messages.append(
                        {
                            "role": "tool",
                            "content": f"Error: Function {func_name} not found",
                        }
                    )
            continue

        else:
            finish_reason = "stop"
            break

    else:
        finish_reason = "length"

    # 5. Construct Response
    final_tool_calls = (
        all_accumulated_tool_calls if all_accumulated_tool_calls else None
    )

    message = MockMessage(
        role="assistant", content=final_content, tool_calls=final_tool_calls
    )

    choice = MockChoice(message=message, finish_reason=finish_reason)
    usage = MockUsage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_prompt_tokens + total_completion_tokens,
    )

    return MockResponse(choices=[choice], usage=usage, model=model)
