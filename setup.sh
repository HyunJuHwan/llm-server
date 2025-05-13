#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install fastapi pydantic transformers torch uvicorn
pip freeze > requirements.txt
echo "✅ 설치 완료! 서버 실행: source venv/bin/activate && uvicorn tinyllama_server:app --reload --port 8000"
