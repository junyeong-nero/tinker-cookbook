import tinker
import logging
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.colors import Normalize
from matplotlib import cm
from tinker import types
from tinker_cookbook.tokenizer_utils import get_tokenizer
from typing import List, Dict, Optional, Any, Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TinkerExperiment")

service_client = tinker.ServiceClient()


def inference(
    client,
    tokenizer,
    prompt: str,
    max_tokens=1024,
    temperature=1,
):
    logger.info(
        f"Starting inference with temperature={temperature}, max_tokens={max_tokens}"
    )

    input_ids = tokenizer.encode(prompt)
    logger.debug(f"Encoded prompt length: {len(input_ids)}")

    prompt_input = types.ModelInput.from_ints(input_ids)
    params = types.SamplingParams(max_tokens=max_tokens, temperature=temperature)

    response = client.sample(
        prompt=prompt_input,
        sampling_params=params,
        num_samples=1,
    ).result()

    tokens = response.sequences[0].tokens
    # logprobs = response.sequences[0].logprobs # Unused in original code return
    text = tokenizer.decode(response.sequences[0].tokens)

    logger.info(f"Inference completed. Generated sequence length: {len(tokens)}")
    return response, text


def get_logprob(client, tokenizer, prompt):
    # logger.debug("Fetching logprobs for prompt...")
    # (너무 빈번하게 호출될 경우 주석 처리하거나 DEBUG 레벨 유지)

    sample_response = client.sample(
        prompt=types.ModelInput.from_ints(tokenizer.encode(prompt)),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=20,
    ).result()

    return sample_response.topk_prompt_logprobs[1:]


def get_reverse_KL(student, teacher):
    student = {token: logprob for token, logprob in student}
    teacher = {token: logprob for token, logprob in teacher}

    token_subset = set(student.keys())
    diff = []

    for token in token_subset:
        prob_student = math.exp(student.get(token, -100))
        # teacher에 해당 토큰이 없으면 매우 작은 값(-100)으로 대체
        diff.append(
            prob_student * (student.get(token, -100) - teacher.get(token, -100))
        )

    return sum(diff)


def generate(student_model, teacher_model, prompt):
    logger.info("Initializing generation and comparison process")
    logger.info(f"Student Model: {student_model}")
    logger.info(f"Teacher Model: {teacher_model}")

    tokenizer = get_tokenizer(student_model)
    student_client = service_client.create_sampling_client(base_model=student_model)
    teacher_client = service_client.create_sampling_client(base_model=teacher_model)

    # 1. Student 모델로 텍스트 생성
    result_student, text_student = inference(
        student_client, tokenizer, prompt, max_tokens=4096
    )

    full_text = prompt + text_student
    logger.info("Calculating logprobs for both models on the generated text...")

    # 2. Logprob 계산
    logprob_student = get_logprob(student_client, tokenizer, full_text)
    logprob_teacher = get_logprob(teacher_client, tokenizer, full_text)

    tokens = tokenizer.encode(full_text)[1:]

    # 로깅: 길이 확인 (디버깅용 중요 정보)
    logger.info(f"Token count (encoded): {len(tokens)}")
    logger.info(f"Logprob count (Student): {len(logprob_student)}")
    logger.info(f"Logprob count (Teacher): {len(logprob_teacher)}")

    if not (len(tokens) == len(logprob_student) == len(logprob_teacher)):
        logger.warning(
            "Mismatch in lengths between tokens and logprobs! Metrics calculation might be skewed."
        )

    # 첫 번째 샘플 로그 (디버깅용)
    if len(logprob_student) > 0:
        logger.debug(f"First logprob sample (Student): {logprob_student[0]}")
        logger.debug(f"First logprob sample (Teacher): {logprob_teacher[0]}")

    n = len(logprob_student)
    metrics = []

    logger.info("Computing Reverse KL divergence per token...")
    for i in range(n):
        r_kl = get_reverse_KL(logprob_student[i], logprob_teacher[i])
        metrics.append(
            {
                "reverse_KL": r_kl,
                "logprob_student": logprob_student[i],
                "logprob_teacher": logprob_teacher[i],
                "token": tokens[i],
                "token_decoded": tokenizer.decode([tokens[i]]),
            }
        )

    logger.info(f"Metrics computation finished. Total items: {len(metrics)}")
    return metrics


import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize

# from matplotlib import cm  <-- 더 이상 필요하지 않음 (matplotlib.colormaps 사용)


def create_kl_heatmap(metrics, filename="kl_heatmap.png", width=12, fontsize=12):
    """
    metrics: list of dicts {'reverse_KL', 'token_decoded', ...}
    filename: output filename
    width: figure width in inches
    """
    logger.info(f"Creating heatmap visualization: {filename}")

    # 1. 데이터 추출 및 정규화
    kl_values = [m["reverse_KL"] for m in metrics]
    max_val = max(kl_values) if kl_values else 1.0
    min_val = min(kl_values) if kl_values else 0.0

    logger.info(f"KL Divergence Stats - Min: {min_val:.4f}, Max: {max_val:.4f}")

    # 정규화 객체
    norm = Normalize(vmin=min_val, vmax=max_val)

    # [FIX 1] MatplotlibDeprecationWarning 해결
    # cm.get_cmap("Reds") 대신 matplotlib.colormaps 사용
    cmap = matplotlib.colormaps["Reds"]

    # 2. 캔버스 설정
    fig = plt.figure(figsize=(width, width))
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    # 초기 커서 위치 설정
    x_pos = 0.01
    y_pos = 0.95
    line_height = 0.05

    renderer = fig.canvas.get_renderer()
    trans = ax.transData.inverted()

    for item in metrics:
        token_text = item["token_decoded"]
        score = item["reverse_KL"]

        # [FIX 2] Glyph 65039 (\N{VARIATION SELECTOR-16}) 경고 해결
        # 이모지 스타일 선택자(\ufe0f) 제거
        token_text = token_text.replace("\ufe0f", "")

        # 제어 문자 제거 (줄바꿈 제외)
        token_text = "".join(ch for ch in token_text if ch == "\n" or ch.isprintable())

        # Matplotlib 수식 파싱 방지 ($ -> \$)
        display_text = token_text.replace("$", r"\$")

        # 줄바꿈 문자가 있는 경우 처리
        if "\n" in display_text:
            parts = display_text.split("\n")
            for i, part in enumerate(parts):
                if part:
                    bg_color = cmap(norm(score))
                    text_obj = ax.text(
                        x_pos,
                        y_pos,
                        part,
                        fontsize=fontsize,
                        fontfamily="monospace",
                        bbox=dict(facecolor=bg_color, edgecolor="none", pad=1.5),
                        verticalalignment="top",
                    )
                    bbox = text_obj.get_window_extent(renderer=renderer)
                    bbox_data = bbox.transformed(trans)
                    width_data = bbox_data.width

                    x_pos += width_data
                    if x_pos > 0.95:
                        x_pos = 0.01
                        y_pos -= line_height

                if i < len(parts) - 1:
                    x_pos = 0.01
                    y_pos -= line_height
            continue

        # 일반 텍스트 처리
        bg_color = cmap(norm(score))
        text_obj = ax.text(
            x_pos,
            y_pos,
            display_text,
            fontsize=fontsize,
            fontfamily="monospace",
            bbox=dict(facecolor=bg_color, edgecolor="none", pad=1.5),
            verticalalignment="top",
        )

        bbox = text_obj.get_window_extent(renderer=renderer)
        bbox_data = bbox.transformed(trans)
        width_data = bbox_data.width

        x_pos += width_data
        if x_pos > 0.95:
            x_pos = 0.01
            y_pos -= line_height

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap saved successfully to {filename}")


def main():
    logger.info("Script started.")

    # problem = "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."

    problem = "Let $ABC$ be a triangle inscribed in circle $\omega$. Let the tangents to $\omega$ at $B$ and $C$ intersect at point $D$, and let $\overline{AD}$ intersect $\omega$ at $P$. If $AB=5$, $BC=9$, and $AC=10$, $AP$ can be written as the form $\frac{m}{n}$, where $m$ and $n$ are relatively prime integers. Find $m + n$."

    student_model = "Qwen/Qwen3-8B"
    teacher_model = "Qwen/Qwen3-32B"

    try:
        # 1. Metrics 생성
        metrics = generate(student_model, teacher_model, problem)

        # 2. 이미지 생성 함수 호출
        output_filename = "reverse_kl_visualization.png"
        create_kl_heatmap(metrics, filename=output_filename)

    except Exception as e:
        logger.error("An error occurred during execution", exc_info=True)

    logger.info("Script execution finished.")


if __name__ == "__main__":
    main()
