# Hugging Face Spaces(Docker)용 — 2026-11 Render 무료(512MB)에서 목소리 분석 모델(torch·CNN·Whisper)을 올리다
# "Ran out of memory (used over 512MB)"로 서버가 계속 꺼져서, 메모리가 넉넉한 HF Spaces 무료(CPU basic)로 옮김.
# 코드는 Render와 같고 실행 방식만 이 파일로 정한다. Render(render.yaml)는 그대로 남겨 둠.
FROM python:3.11-slim

# ffmpeg: 브라우저 녹음(webm)을 librosa·Whisper가 읽고, 2단계 영상 변환에도 쓴다(Render 기본 서버엔 없었음)
# libgl1·libglib2.0-0: mediapipe/opencv가 불러오는 시스템 라이브러리
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces는 uid 1000 사용자로 실행한다(권장 설정)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Whisper 모델을 이미지에 미리 받아 둔다 — 안 그러면 서버가 켜질 때마다 첫 분석에서 ~140MB를 새로 받느라
# 마이크 확인이 한참 걸린다. app.py의 whisper.load_model("base")와 같은 모델·같은 저장 위치(~/.cache/whisper)
RUN python -c "import whisper; whisper.load_model('base')"

COPY --chown=user . .

# HF Spaces가 보는 포트(README 앞머리 app_port와 같아야 함). 첫 분석은 모델을 올리느라 오래 걸려 timeout을 넉넉히
EXPOSE 7860
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "300"]
