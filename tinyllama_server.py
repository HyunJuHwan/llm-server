# filename: gemma_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
import json
import re
import os

app = FastAPI()

# 모델 경로 설정
model_path = "./gemma-3-4b-it-q4_0.gguf"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

# GGUF 모델 로드
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,   # 시스템에 맞게 조정
    n_gpu_layers=20,  # GPU가 있다면 (없으면 0)
    verbose=True
)

class ChatRequest(BaseModel):
    messages: list  # [{"role": "user" | "system", "content": "..."}]

def extract_json_block(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text)
    text = text.replace("```", "").strip()
    try:
        return json.loads(text)
    except:
        matches = re.findall(r'{[^{}]*(?:{[^{}]*}[^{}]*)*[^{}]*}', text)
        for m in matches:
            try:
                return json.loads(m)
            except:
                continue
        raise ValueError("Failed to parse JSON")

ALLOWED_TOOLS = {
    "createCharacter", "createScene", "updateCharacter",
    "confirmCharacter", "buildWebtoon", "buildVideo"
}

@app.post("/generate")
async def generate(req: ChatRequest):
    tool_system_prompt = """
You are an API planner that ONLY returns a JSON array of tool calls.

---

VERY IMPORTANT RULES:

- Your output MUST be a raw JSON array. Nothing else.
- DO NOT copy or repeat the examples below. Use them only to learn the format.
- DO NOT include:
  - Explanations or natural language
  - Markdown (```), code blocks or comments

---

Examples (for format reference only):

[
  {
    "tool": "createCharacter",
    "input": {
      "style": "2d",
      "prompt": "female character"
    }
  },
  {
    "tool": "createScene",
    "input": {
      "character_ids": ["c-1"],
      "scene_description": "a girl standing in a park"
    }
  }
]

---

✅ Available tools:
- createCharacter: { style: "2d" | "3d", prompt: string }
- createScene: { character_ids: string[], scene_description: string }
- updateCharacter: { old_character_id: string, prompt: string, style: "2d" | "3d" }
- confirmCharacter: { character_ids: string[] }
- buildWebtoon: { scene_ids: string[], speech_bubbles?: any }
- buildVideo: { frame_folder: "scene" }

Respond ONLY with a JSON array of tool calls appropriate to the user request.
""".strip()

    # Prompt 구성
    user_msgs = "\n".join([f"{m['role']}: {m['content']}" for m in req.messages])
    full_prompt = f"{tool_system_prompt}\n\n{user_msgs}\nassistant:"

    # 모델에 입력
    response = llm(
        full_prompt,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        stop=["</s>", "user:", "system:", "assistant:"]
    )

    raw_output = response["choices"][0]["text"].strip()
    print("[DEBUG] raw_output:\n", raw_output)

    try:
        parsed = extract_json_block(raw_output)
        if isinstance(parsed, list):
            parsed = [p for p in parsed if isinstance(p, dict) and p.get("tool") in ALLOWED_TOOLS]
        return { "output": parsed }
    except Exception as e:
        return { "error": str(e), "raw_output": raw_output }
