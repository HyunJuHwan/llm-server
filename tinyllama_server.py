# filename: tinyllama_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import json

app = FastAPI()

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True
)
model.eval()

class ChatRequest(BaseModel):
    messages: list  # [{"role": "user" | "system", "content": "..."}]

# JSON 추출 유틸
def extract_json_block(text: str) -> dict:
    try:
        # 코드 블럭 제거
        text = text.replace("```json", "").replace("```", "").strip()

        # 가장 바깥 중괄호 JSON 블럭들 찾기
        matches = re.findall(r'{[^{}]*(?:{[^{}]*}[^{}]*)*[^{}]*}', text)

        for match in matches:
            try:
                return json.loads(match)
            except:
                continue

        raise ValueError("파싱 가능한 JSON 블록이 없습니다.")
    except Exception as e:
        raise ValueError(f"JSON 파싱 실패: {e}\n원본 응답:\n{text}")


@app.post("/generate")
async def generate(req: ChatRequest):
    # MCP JSON 명령만 출력하게 하는 system 프롬프트
    tool_system_prompt = """
너는 MCP 툴 호출용 JSON만 생성하는 LLM이다.
사용자의 요청을 아래 JSON 형식으로만 출력해야 한다:

예:
{
  "tool": "createCharacter",
  "input": {
    "style": "2d",
    "prompt": "여자 캐릭터"
  }
}

툴 목록:
- createCharacter: { style: "2d" | "3d", prompt: string }
- createScene: { description: string }

❗주의: 설명, 말투, 마크다운, 코드 블럭 등은 절대 포함하지 말고 오직 순수 JSON만 출력해야 함.
""".strip()

    all_messages = [
        { "role": "system", "content": tool_system_prompt }
    ] + req.messages

    input_ids = tokenizer.apply_chat_template(
        all_messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cpu")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    raw_output = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
    print("[DEBUG] raw_output:\n", raw_output)
    # JSON만 추출
    try:
        parsed = extract_json_block(raw_output)
        return { "output": parsed }
    except Exception as e:
        return { "error": str(e) }
