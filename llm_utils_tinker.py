import json
import re
import uuid
import time
import tinker
from tinker import types
from typing import List, Dict, Optional, Any, Union


# ==========================================
# 1. OpenAI Style Mock Classes (generate 함수 호환용)
# ==========================================
class MockFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments  # JSON string


class MockToolCall:
    def __init__(self, id: str, func_name: str, func_args: str):
        self.id = id
        self.type = "function"
        self.function = MockFunction(func_name, func_args)


class MockMessage:
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
    def __init__(self, message: MockMessage, finish_reason: str = "stop"):
        self.message = message
        self.finish_reason = finish_reason


class MockUsage:
    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class MockResponse:
    def __init__(self, choices: List[MockChoice], usage: MockUsage, model: str):
        self.id = f"chatcmpl-{uuid.uuid4()}"
        self.object = "chat.completion"
        self.created = int(time.time())
        self.model = model
        self.choices = choices
        self.usage = usage

    def to_dict(self):
        # generate 함수의 raw_data 저장을 위해 필요할 수 있음
        return {
            "id": self.id,
            "choices": [
                {
                    "message": {
                        "role": c.message.role,
                        "content": c.message.content,
                        # tool_calls 직렬화 로직은 필요시 추가
                    },
                    "finish_reason": c.finish_reason,
                }
                for c in self.choices
            ],
            "usage": self.usage.__dict__,
        }


# ==========================================
# 2. Helper Logic (Template & Parsing)
# ==========================================
def apply_qwen_chat_template(
    messages: List[Dict], tools: Optional[List[Dict]] = None
) -> str:
    # (이전과 동일한 로직, 생략 없이 사용한다고 가정)
    prompt = ""
    if tools:
        prompt += "<|im_start|>system\n"
        if messages and messages[0]["role"] == "system":
            prompt += messages[0]["content"] + "\n\n"
        prompt += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        prompt += "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        for tool in tools:
            prompt += "\n" + json.dumps(tool, ensure_ascii=False)
        prompt += "\n</tools>\n\n"
        prompt += 'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'
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
            if "tool_calls" in msg and msg["tool_calls"]:
                # 단순화된 처리
                prompt += "\n"  # tool calls formatting 생략
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


def parse_tool_calls_from_text(text: str) -> List[Dict]:
    pattern = r"<tool_call>\n(.*?)\n</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    tool_calls = []
    for json_str in matches:
        try:
            tool_calls.append(json.loads(json_str))
        except json.JSONDecodeError:
            continue
    return tool_calls


# 툴 정의 (예시)
AVAILABLE_FUNCTIONS = {
    "get_current_weather": lambda location, unit="celsius": json.dumps(
        {"location": location, "temp": 22, "unit": unit}
    )
}


# ==========================================
# 3. Completion Function (generate 호환용)
# ==========================================
def completion(
    model: str,
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict]] = None,
    tool_choice: Optional[str] = None,
    **kwargs,
) -> MockResponse:

    # 1. 클라이언트 초기화 (요청하신 방식)
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model)
    training_client = service_client.create_lora_training_client(base_model=model)
    tokenizer = training_client.get_tokenizer()

    # 2. kwargs에서 파라미터 추출 (generate에서 넘겨주는 값 처리)
    max_tokens = kwargs.get("max_tokens", 1024)
    temperature = kwargs.get("temperature", 0.7)
    top_p = kwargs.get("top_p", 0.9)
    top_k = kwargs.get("top_k", 50)
    stop = kwargs.get("stop", ["<|im_end|>"])

    # messages 복사
    current_messages = [
        m.copy() for m in messages
    ]  # Deep copy might be safer but shallow is usually ok for simple dicts
    max_turns = 5

    total_prompt_tokens = 0
    total_completion_tokens = 0

    final_content = ""
    final_tool_calls = None
    finish_reason = "stop"

    # 3. Tool Execution Loop
    for _ in range(max_turns):
        prompt_text = apply_qwen_chat_template(current_messages, tools=tools)

        # 토큰 수 계산
        input_ids = tokenizer.encode(prompt_text)
        total_prompt_tokens += len(input_ids)

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

        # Tool Call 파싱
        tool_calls_data = parse_tool_calls_from_text(generated_text)

        # [수정 1] 현재 턴의 결과로 변수를 미리 업데이트합니다.
        # 이렇게 하면 max_turns에 걸리거나 루프가 멈췄을 때 마지막 상태를 보존할 수 있습니다.
        content_part = (
            generated_text.split("<tool_call>")[0].replace("<|im_end|>", "").strip()
        )
        final_content = content_part

        # 툴 호출 데이터가 있으면 포맷에 맞춰 저장, 없으면 None
        if tool_calls_data:
            # generate 함수 등 외부에서 사용할 수 있도록 MockToolCall 형태로 변환하거나 raw list 유지
            # 여기서는 raw list를 유지하다가 MockResponse에서 처리하거나,
            # 간단히 리스트가 존재함만 표시합니다.
            final_tool_calls = [
                MockToolCall(
                    id=f"call_{uuid.uuid4()}",
                    func_name=tc.get("name"),
                    func_args=(
                        json.dumps(tc.get("arguments"))
                        if isinstance(tc.get("arguments"), dict)
                        else str(tc.get("arguments"))
                    ),
                )
                for tc in tool_calls_data
            ]
        else:
            final_tool_calls = None

        # --- 분기 처리 ---

        if tool_calls_data:
            # 툴 호출이 발생함 -> 히스토리 저장 및 실행 -> 루프 계속(continue)

            # 1. Assistant 메시지 추가 (Tool Call 포함)
            current_messages.append(
                {
                    "role": "assistant",
                    "content": content_part,
                    "tool_calls": [{"function": tc} for tc in tool_calls_data],
                }
            )

            # 2. 툴 실제 실행
            for tc in tool_calls_data:
                func_name = tc.get("name")
                args = tc.get("arguments")

                # MockResponse 반환값을 위해 함수 인자 파싱 (dict -> str -> dict 변환 방지용)
                if isinstance(args, str):
                    try:
                        args_dict = json.loads(args)
                    except:
                        args_dict = {}
                else:
                    args_dict = args

                if func_name in AVAILABLE_FUNCTIONS:
                    try:
                        res = AVAILABLE_FUNCTIONS[func_name](**args_dict)
                    except Exception as e:
                        res = str(e)

                    # 3. 결과 히스토리 추가
                    current_messages.append(
                        {"role": "tool", "name": func_name, "content": res}
                    )
                else:
                    current_messages.append(
                        {"role": "tool", "content": f"Error: {func_name} not found"}
                    )

            # 툴 실행 결과를 가지고 다시 추론하기 위해 continue
            continue

        else:
            # 툴 호출 없음 -> 최종 답변 완료 -> 루프 종료(break)
            finish_reason = "stop"
            break

    # for 루프가 break 없이 끝난 경우 (max_turns 초과)
    else:
        finish_reason = "length"
        # 이때 final_content와 final_tool_calls에는 마지막 턴의 시도 내용이 들어있음

    # 4. 결과 객체 생성
    msg_obj = MockMessage(
        role="assistant", content=final_content, tool_calls=final_tool_calls
    )

    choice_obj = MockChoice(message=msg_obj, finish_reason=finish_reason)
    usage_obj = MockUsage(
        total_prompt_tokens,
        total_completion_tokens,
        total_prompt_tokens + total_completion_tokens,
    )

    return MockResponse(choices=[choice_obj], usage=usage_obj, model=model)
