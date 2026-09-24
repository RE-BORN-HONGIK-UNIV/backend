"""
Re-born 보이스 터치 - Flask 백엔드 (최종 버전)
- CNN: 슬라이딩 윈도우(3초/1초 stride) 추론
- 채움말: librosa 소리구간 + Whisper word timestamp 결합
- 멈춤: librosa 기반 불안한 멈춤 분석
"""

import os
from dotenv import load_dotenv
load_dotenv()  # .env 파일 읽어서 환경변수로 등록

import io
import base64
import subprocess
import tempfile
import time
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from flask_cors import CORS
from PIL import Image
from step3.tts import synthesize_speech

# torch/torchvision, whisper, step2의 mediapipe·cv2 기반 모듈(landmark_face_points,
# gaze_analyzer)은 여기서 import하지 않고 실제로 쓰는 함수 안에서 지연 import한다.
# 이유: 이것들만으로도 수백MB가 프로세스 메모리에 항상 올라가는데, /health·로그인·
# 커뮤니티처럼 이 라이브러리가 전혀 필요 없는 라우트도 같은 워커에서 뜨기 때문에,
# 모듈 최상단에서 import하면 그 라우트만 받는 워커도 무조건 이 비용을 낸다
# (배포 환경 메모리 제한 때문에 2026-09 경량화 작업으로 옮김. _init_torch()/load_models() 참고).

try:
    from dotenv import load_dotenv
    load_dotenv()  # .env 파일이 있으면 읽어옴 (없으면 그냥 넘어감 — python-dotenv는 선택 설치)
except ImportError:
    pass

# Step2 시선/깜빡임 모듈 import — blink_analyzer/expression_analyzer/scoring/set_baseline은
# numpy만 써서 가벼우니 그대로 상단에 둠. mediapipe/cv2가 필요한 landmark_face_points·
# gaze_analyzer는 analyze_gaze_blink() 안에서 지연 import.
from step2.blink_analyzer import (
    compute_ear_series, detect_blinks,
    compute_blink_blendshape_series, detect_blinks_from_blendshape,
)
from step2.expression_analyzer import compute_expression_series, summarize_expression, detect_expression_segments
from step2.scoring import score_blink_rate, score_gaze_segments, score_expression
from step2.set_baseline import calibrate_baseline_ear, calibrate_baseline_gaze

# Step1 LLM 코칭 피드백 (선택적 — 키 없으면 자동 폴백)
from step1.llm_feedback import generate_feedback

# Step3 면접 질문 생성 (선택적 — 키 없으면 자동 폴백)
from step3.interview_question import generate_question

# whisper 모듈 자체는 load_models() 안에서 지연 import. 그 전까지는 "아직 모른다"가
# 아니라 "안 붙어있다"로 취급 — analyze_filler()/interview_transcribe()가 load_models()
# 호출 전에 이 값을 참조할 일은 없지만(두 라우트 모두 진입 시점에 load_models()를 먼저
# 부름), 혹시 몰라 안전한 기본값으로 초기화해둔다.
WHISPER_AVAILABLE = False

def _require_env(key):
    """비밀번호/서명키처럼 코드에 하드코딩하면 안 되는 값 — .env 또는 환경변수로 필수 주입.
    backend/.env.example 참고해서 backend/.env 만들 것 (.env는 .gitignore에 이미 포함됨)."""
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(
            f"환경변수 {key}가 설정되지 않았습니다. backend/.env.example을 참고해 "
            f"backend/.env 파일을 만들어주세요."
        )
    return value


app = Flask(__name__)
CORS(app)

# Render가 주는 connectionString은 "postgres://" 스킴인데, SQLAlchemy 1.4+
# (Flask-SQLAlchemy 3.x)는 이 스킴을 완전히 버려서 NoSuchModuleError로 죽는다.
# psycopg2 드라이버가 여전히 처리 가능한 "postgresql://"로 바꿔서 넣어준다.
_database_url = _require_env('DATABASE_URL')
if _database_url.startswith('postgres://'):
    _database_url = _database_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = _database_url
app.config['SECRET_KEY'] = _require_env('SECRET_KEY')
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(80), nullable=False)
    nickname = db.Column(db.String(80), nullable=True)
    birthdate = db.Column(db.String(10), nullable=True)
    terms_agreed = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)


class Stage2Result(db.Model):
    """2단계(표정·시선) 분석 결과 — 세션 간 비교("지난번엔 ~했어요")를 위해 유저별로 쌓아둔다.
    프론트 localStorage 이력을 대체 (기기 바뀌면 날아가던 문제 해결)."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    blink_rate_per_min = db.Column(db.Float, nullable=False)
    blink_status = db.Column(db.String(20), nullable=False)
    blink_score = db.Column(db.Float, nullable=False)

    avg_fixation_sec = db.Column(db.Float, nullable=False)
    gaze_score = db.Column(db.Float, nullable=False)

    smile_ratio = db.Column(db.Float, nullable=False)
    tension_ratio = db.Column(db.Float, nullable=False)
    expression_score = db.Column(db.Float, nullable=False)
    expression_status = db.Column(db.String(20), nullable=False)

    def to_entry(self):
        """프론트 Stage2Entry와 동일한 shape — 그대로 previous 비교에 쓸 수 있게."""
        return {
            "at": self.created_at.isoformat(),
            "blinkRatePerMin": self.blink_rate_per_min,
            "avgFixationSec": self.avg_fixation_sec,
            "smileRatio": self.smile_ratio,
            "tensionRatio": self.tension_ratio,
        }


class Post(db.Model):
    """'이야기' 자유 게시판 글."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def to_dict(self, comment_count=0, current_user=None):
        author = User.query.get(self.user_id)
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'author': (author.nickname or author.name) if author else '탈퇴한 사용자',
            'created_at': self.created_at.isoformat(),
            'comment_count': comment_count,
            'is_own': bool(current_user and current_user.id == self.user_id),
        }


class Comment(db.Model):
    """게시글에 달린 댓글."""
    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def to_dict(self, current_user=None):
        author = User.query.get(self.user_id)
        return {
            'id': self.id,
            'content': self.content,
            'author': (author.nickname or author.name) if author else '탈퇴한 사용자',
            'created_at': self.created_at.isoformat(),
            'is_own': bool(current_user and current_user.id == self.user_id),
        }


def get_current_user():
    """Authorization: Bearer <token> 헤더의 JWT를 검증해 User를 반환. 없거나 무효하면 None."""
    header = request.headers.get('Authorization', '')
    if not header.startswith('Bearer '):
        return None
    token = header[len('Bearer '):]
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
    except jwt.PyJWTError:
        return None
    return User.query.get(payload.get('user_id'))


# 테이블 생성 — 예전엔 `if __name__ == '__main__':` 블록에서만 호출돼서, gunicorn
# 같은 WSGI 서버로 띄우면(배포 시 쓰는 방식) 이 코드가 한 번도 안 돌고 테이블이
# 영영 안 생겨서 모든 DB 라우트가 깨지는 문제가 있었다 — 모듈 로드 시점(워커 프로세스
# 시작 시 1회)에 바로 실행하도록 옮김. DB 연결 실패 시에도 앱 자체는 뜨게 폴백.
try:
    with app.app_context():
        db.create_all()
except Exception as e:
    print(f"[WARNING] DB 연결 실패, DB 없이 서버 실행: {e}")

MODEL_PATH = os.environ.get("MODEL_PATH", "best_size_large.pth")
PAUSE_THRESHOLD = 1.2

# ── CNN 슬라이딩 윈도우 설정 (inference_cnn_final.py 동일) ──────────
TARGET_SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
F_MIN = 300
F_MAX = 2500
IMG_SIZE = 224
WINDOW_SEC = 3.0
STRIDE_SEC = 1.0

# ── 채움말 판정 설정 (analyze_filler_final.py 동일) ─────────────────
FILLER_MIN_DURATION = 0.1
FILLER_MAX_DURATION = 0.8

# ── CNN 모델 구조 ────────────────────────────────────────────────────
# torch/torchvision은 _init_torch()가 처음 호출될 때만 import한다 (모듈 상단 import
# 제거 사유는 파일 상단 주석 참고). SpeechAnxietyCNN/transform은 그 전까지 None이고,
# load_models()가 항상 _init_torch()를 먼저 불러서 채워준다.
SpeechAnxietyCNN = None
transform = None

def _init_torch():
    """torch/torchvision을 최초 1회만 import하고 CNN 클래스·전처리 파이프라인을 준비."""
    global torch, SpeechAnxietyCNN, transform
    if SpeechAnxietyCNN is not None:
        return
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms

    class _SpeechAnxietyCNN(nn.Module):
        def __init__(self, dropout=0.5, size='large'):
            super(_SpeechAnxietyCNN, self).__init__()
            sizes = {
                'small':  [32, 64, 128],
                'medium': [64, 128, 256],
                'large':  [128, 256, 512]
            }
            filters = sizes[size]
            self.features = nn.Sequential(
                nn.Conv2d(3, filters[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(filters[0]),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(filters[1]),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(filters[2]),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(filters[2], 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 3)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.gap(x)
            x = self.classifier(x)
            return x

    SpeechAnxietyCNN = _SpeechAnxietyCNN
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

cnn_model = None
whisper_model = None
_models_loaded = False

def load_models():
    """CNN/Whisper 모델 로드 — 최초 요청 시점에 한 번만 실행되는 지연 로딩.
    예전엔 `if __name__ == '__main__':` 블록에서만 불려서, gunicorn 등 WSGI 서버로
    띄우면(배포 환경이 이 방식) 이 함수 자체가 한 번도 안 불리고 모델이 영영 None으로
    남는 버그가 있었다 — /analyze, /interview/transcribe 라우트 진입 시 직접 호출하는
    방식으로 바꿔서 실행 방식(개발 서버/gunicorn 등)과 무관하게 항상 동작하게 했다."""
    global cnn_model, whisper_model, WHISPER_AVAILABLE, _models_loaded
    if _models_loaded:
        return
    _models_loaded = True

    if os.path.exists(MODEL_PATH):
        try:
            _init_torch()  # CNN 가중치 파일이 실제로 있을 때만 torch를 메모리에 올림 —
            # 지금처럼 파일이 없으면(데모 모드) /analyze를 아무리 호출해도 torch 자체가
            # 로드되지 않는다 (2026-09 경량화 후속: 순서 뒤집기 전엔 데모 모드에서도
            # 매번 쓸데없이 torch 전체를 import하고 있었음).
            m = SpeechAnxietyCNN(dropout=0.5, size='large')
            m.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            m.eval()
            cnn_model = m
            print(f"[OK] CNN 모델 로드 완료: {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] CNN 모델 로드 실패: {e}")
    else:
        print(f"[WARNING] CNN 모델 파일 없음: {MODEL_PATH}")

    try:
        import whisper
        WHISPER_AVAILABLE = True
    except ImportError:
        WHISPER_AVAILABLE = False
        print("[WARNING] whisper 없음 - 발화 유창성 0점 고정")

    if WHISPER_AVAILABLE:
        try:
            whisper_model = whisper.load_model("base")
            print("[OK] Whisper 모델 로드 완료")
        except Exception as e:
            print(f"[ERROR] Whisper 로드 실패: {e}")

def save_temp_file(file_storage):
    """Windows 호환 임시 파일 저장"""
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"reborn_audio_{os.getpid()}.wav")
    file_storage.save(tmp_path)
    print(f"[INFO] 임시 파일 저장: {tmp_path}")
    return tmp_path


def save_temp_video(file_storage):
    """영상 업로드를 원본 확장자 그대로 임시 저장 (mp4/mov/webm 등)."""
    tmp_dir = tempfile.gettempdir()
    ext = os.path.splitext(file_storage.filename or "")[1].lower() or ".mp4"
    tmp_path = os.path.join(tmp_dir, f"reborn_video_{os.getpid()}{ext}")
    file_storage.save(tmp_path)
    print(f"[INFO] 임시 영상 저장: {tmp_path}")
    return tmp_path


def ensure_mp4(video_path):
    """영상을 mp4로 통일한다 — mov/webm 등이 들어오면 ffmpeg으로 변환.
    이미 mp4면 그대로 반환. 변환 실패(ffmpeg 없음 등) 시에는 원본 경로를 그대로
    반환하고 분석은 계속 진행한다 (OpenCV가 원본 포맷을 읽을 수 있는 경우가 많음).
    반환값이 입력과 다르면 호출측에서 변환된 파일도 정리(삭제)해야 한다."""
    if video_path.lower().endswith(".mp4"):
        return video_path
    mp4_path = os.path.splitext(video_path)[0] + "_converted.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", mp4_path],
            check=True, capture_output=True, timeout=120,
        )
        print(f"[OK] mp4 변환 완료: {video_path} -> {mp4_path}")
        return mp4_path
    except Exception as e:
        print(f"[WARNING] mp4 변환 실패, 원본 포맷으로 분석 진행: {e}")
        return video_path


def build_highlight_clip(video_path, segments, max_total_sec=15.0, pad_sec=0.3):
    """(start, end) 구간들만 앞뒤로 pad_sec씩 여유를 두고 잘라 이어붙인 짧은 하이라이트
    mp4를 만들어 data URL(base64)로 반환한다. 구간이 없거나 실패하면 None.
    - 원본 영상을 그대로 다 보여주는 대신, "이 표정/시선이 나온 부분"만 편집해서 보여주기 위함.
    - max_total_sec: 하이라이트 총 길이 상한 (구간이 너무 많으면 앞에서부터 이만큼만 사용).
    """
    if not segments:
        return None

    tmp_dir = tempfile.gettempdir()
    uid = f"{os.getpid()}_{int(time.time() * 1000)}"
    part_paths = []
    try:
        total = 0.0
        for i, (start, end) in enumerate(segments):
            if total >= max_total_sec:
                break
            s = max(0.0, start - pad_sec)
            dur = min((end - start) + 2 * pad_sec, max_total_sec - total)
            if dur <= 0.05:
                continue
            part_path = os.path.join(tmp_dir, f"rb_hl_{uid}_{i}.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{s:.3f}", "-i", video_path, "-t", f"{dur:.3f}",
                 "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", part_path],
                check=True, capture_output=True, timeout=60,
            )
            part_paths.append(part_path)
            total += dur

        if not part_paths:
            return None

        if len(part_paths) == 1:
            final_path = part_paths[0]
        else:
            list_path = os.path.join(tmp_dir, f"rb_hl_{uid}_list.txt")
            with open(list_path, "w", encoding="utf-8") as f:
                for p in part_paths:
                    f.write(f"file '{p}'\n")
            final_path = os.path.join(tmp_dir, f"rb_hl_{uid}_final.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", final_path],
                check=True, capture_output=True, timeout=60,
            )
            os.remove(list_path)

        with open(final_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")

        if final_path not in part_paths:
            os.remove(final_path)
        return f"data:video/mp4;base64,{data}"
    except Exception as e:
        print(f"[WARNING] 하이라이트 클립 생성 실패: {e}")
        return None
    finally:
        for p in part_paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass

# ── 멜스펙트로그램 변환 (3초 조각) ───────────────────────────────────
def audio_chunk_to_melspectrogram(y_chunk, sr):
    mel_spec = librosa.feature.melspectrogram(
        y=y_chunk, sr=sr,
        n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=F_MIN, fmax=F_MAX
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    ax.set_position([0, 0, 1, 1])
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=HOP_LENGTH,
                             fmin=F_MIN, fmax=F_MAX, cmap='magma', ax=ax)
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return Image.open(buf).convert('RGB')

# ── CNN 추론 (슬라이딩 윈도우, inference_cnn_final.py 동일 로직) ────
def predict_cnn(audio_path):
    y, sr = librosa.load(audio_path, sr=TARGET_SR)

    window_len = int(WINDOW_SEC * sr)
    stride_len = int(STRIDE_SEC * sr)

    # 3초보다 짧으면 반사 패딩
    if len(y) < window_len:
        pad_len = window_len - len(y)
        y = np.pad(y, (0, pad_len), mode='reflect')

    # 슬라이딩 윈도우로 자르기
    chunks = []
    start = 0
    while start + window_len <= len(y):
        chunks.append(y[start:start + window_len])
        start += stride_len

    # 마지막 구간 보정
    if not chunks or (len(y) - (len(chunks) - 1) * stride_len) > stride_len:
        chunks.append(y[-window_len:])

    # 각 윈도우별 추론
    all_probs = []
    for chunk in chunks:
        img = audio_chunk_to_melspectrogram(chunk, sr)
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = cnn_model(img_tensor)
            probs = torch.sigmoid(output)
        all_probs.append(probs.squeeze(0).numpy())

    avg_probs = np.mean(all_probs, axis=0)

    result = {
        'prolongation': round(float(avg_probs[0]), 4),
        'tremor':       round(float(avg_probs[1]), 4),
        'energy':       round(float(avg_probs[2]), 4),
    }
    # ▲▲▲ [수정 끝] ▲▲▲
    scores = {
        'prolongation_score': round(100 * (1 - result['prolongation']), 1),
        'tremor_score':       round(100 * (1 - result['tremor']), 1),
        'energy_score':       round(100 * (1 - result['energy']), 1),
    }
    print(f"[CNN] 윈도우 수: {len(chunks)}, 확률: {result}")
    return {**result, **scores, 'window_count': len(chunks)}

# ── 채움말 분석 (librosa 소리구간 + Whisper word timestamp) ─────────
def analyze_filler(audio_path):
    try:
        # 1) librosa로 소리 구간 탐지
        y, sr = librosa.load(audio_path, sr=None)
        total_duration = librosa.get_duration(y=y, sr=sr)

        sound_segments = librosa.effects.split(y, top_db=30)
        sound_segments_sec = [(start / sr, end / sr) for start, end in sound_segments]

        # 2) Whisper word timestamp (ffmpeg 없이 numpy 배열로 직접 전달)
        word_segments = []
        if WHISPER_AVAILABLE and whisper_model is not None:
            y16, _ = librosa.load(audio_path, sr=16000, mono=True)
            result = whisper_model.transcribe(y16, language="ko", word_timestamps=True)
            for segment in result["segments"]:
                for word_info in segment.get("words", []):
                    word_segments.append((word_info["start"], word_info["end"]))

        # 3) 채움말 후보 판정
        filler_count = 0
        for seg_start, seg_end in sound_segments_sec:
            seg_duration = seg_end - seg_start
            if not (FILLER_MIN_DURATION <= seg_duration <= FILLER_MAX_DURATION):
                continue

            overlap_found = False
            for word_start, word_end in word_segments:
                overlap = min(seg_end, word_end) - max(seg_start, word_start)
                if overlap > 0 and overlap / seg_duration >= 0.5:
                    overlap_found = True
                    break

            if not overlap_found:
                filler_count += 1

        # 4) 점수 환산
        sound_segment_count = len(sound_segments_sec)
        filler_ratio = filler_count / sound_segment_count if sound_segment_count > 0 else 0
        fluency_score = round(max(0, min(100, 100 * (1 - filler_ratio))), 1)

        print(f"[Filler] 소리구간:{sound_segment_count}, 채움말:{filler_count}, 비율:{filler_ratio}")

        return {
            "total_duration_sec":   round(total_duration, 2),
            "sound_segment_count":  sound_segment_count,
            "filler_count":         filler_count,
            "filler_ratio":         round(filler_ratio, 4),
            "fluency_score":        fluency_score,
        }
    except Exception:
        import traceback
        print(f"[ERROR] filler 분석 실패: {traceback.format_exc()}")
        return {
            "total_duration_sec": 0, "sound_segment_count": 0,
            "filler_count": 0, "filler_ratio": 0, "fluency_score": 0,
        }

# ── 멈춤 분석 (analyze_pause.py 동일) ───────────────────────────────
def analyze_pause(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        total_duration = librosa.get_duration(y=y, sr=sr)
        non_silent = librosa.effects.split(y, top_db=30)

        anxious_pause_count = 0
        anxious_pause_total = 0.0
        all_pause_durations = []

        for i in range(1, len(non_silent)):
            prev_end = non_silent[i-1][1] / sr
            curr_start = non_silent[i][0] / sr
            pause_duration = curr_start - prev_end
            if pause_duration > 0.1:
                all_pause_durations.append(pause_duration)
                if pause_duration >= PAUSE_THRESHOLD:
                    anxious_pause_count += 1
                    anxious_pause_total += pause_duration

        anxious_pause_ratio = anxious_pause_total / total_duration if total_duration > 0 else 0
        pause_score = round(max(0, min(100, 100 * (1 - anxious_pause_ratio))), 1)

        print(f"[Pause] 모든 멈춤: {all_pause_durations}")
        print(f"[Pause] 전체:{total_duration}, 불안멈춤합:{anxious_pause_total}")

        return {
            "pause_score":             pause_score,
            "pause_count":             len(all_pause_durations),
            "anxious_pause_count":     anxious_pause_count,
            "anxious_pause_total_sec": round(anxious_pause_total, 2),
            "anxious_pause_ratio":     round(anxious_pause_ratio, 4),
            "total_duration_sec":      round(total_duration, 2),
        }
    except Exception as e:
        print(f"[ERROR] pause 분석 실패: {e}")
        return {"pause_score": 0, "pause_count": 0, "anxious_pause_count": 0,
                "anxious_pause_total_sec": 0, "anxious_pause_ratio": 0, "total_duration_sec": 0}

# ── API ──────────────────────────────────────────────────────────────
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')
    nickname = data.get('nickname')
    birthdate = data.get('birthdate')
    terms_agreed = data.get('terms_agreed')

    if not email or not password or not name:
        return jsonify({'error': '필수 항목을 입력해주세요.'}), 400

    if not terms_agreed:
        return jsonify({'error': '이용약관에 동의해주세요.'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'error': '이미 가입된 이메일입니다.'}), 409

    user = User(
        email=email,
        password_hash=generate_password_hash(password),
        name=name,
        nickname=nickname,
        birthdate=birthdate,
        terms_agreed=terms_agreed,
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': '회원가입 완료'}), 201


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'error': '이메일 또는 비밀번호가 올바르지 않습니다.'}), 401

    token = jwt.encode(
        {
            'user_id': user.id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7),
        },
        app.config['SECRET_KEY'],
        algorithm='HS256',
    )
    return jsonify({'token': token, 'name': user.name})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'cnn_model_loaded': cnn_model is not None,
        'whisper_loaded': whisper_model is not None,
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    load_models()  # 최초 호출 시에만 실제로 torch/whisper를 로드 (이후엔 즉시 리턴)

    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다'}), 400

    f = request.files['file']
    allowed = {'.wav', '.mp3', '.m4a', '.ogg', '.webm', '.flac'}
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in allowed:
        return jsonify({'error': f'지원하지 않는 형식: {ext}'}), 400

    tmp_path = None
    try:
        tmp_path = save_temp_file(f)

        # 1) CNN (슬라이딩 윈도우) → 음성 안정성, 발화 지속성, 발화 에너지
        if cnn_model is not None:
            cnn_result = predict_cnn(tmp_path)
            stability  = cnn_result['tremor_score']
            continuity = cnn_result['prolongation_score']
            calm       = cnn_result['energy_score']
            cnn_probs  = {
                'tremor':       round(cnn_result['tremor'] * 100, 1),
                'prolongation': round(cnn_result['prolongation'] * 100, 1),
                'energy':       round(cnn_result['energy'] * 100, 1),
            }
            window_count = cnn_result['window_count']
        else:
            stability = continuity = calm = 0
            cnn_probs = {'tremor': 0, 'prolongation': 0, 'energy': 0}
            window_count = 0

        # 2) 채움말 (librosa + Whisper) → 발화 유창성
        filler_result = analyze_filler(tmp_path)
        fluency = filler_result['fluency_score']

        # 3) 멈춤 (librosa) → 침묵 조절력
        pause_result = analyze_pause(tmp_path)
        pause_ctrl = pause_result['pause_score']

        return jsonify({
            'scores': {
                'stability':  stability,
                'fluency':    fluency,
                'pause_ctrl': pause_ctrl,
                'continuity': continuity,
                'calm':       calm,
            },
            'model_scores': {
                '음성 안정성 (tremor)':       stability,
                '발화 지속성 (prolongation)': continuity,
                '발화 에너지 (energy)':     calm,
                '발화 유창성 (filler)':       fluency,
                '침묵 조절력 (pause)':        pause_ctrl,
            },
            'probabilities': cnn_probs,
            'filler_detail': {
                'filler_count':       filler_result['filler_count'],
                'sound_segment_count': filler_result['sound_segment_count'],
                'filler_ratio_pct':   round(filler_result['filler_ratio'] * 100, 1),
            },
            'pause_detail': {
                'pause_count':             pause_result['pause_count'],
                'anxious_pause_count':     pause_result['anxious_pause_count'],
                'anxious_pause_total_sec': pause_result['anxious_pause_total_sec'],
                'anxious_pause_ratio':     pause_result['anxious_pause_ratio'],
                'total_duration_sec':      pause_result['total_duration_sec'],
            },
            'cnn_window_count': window_count,
            'demo_mode': cnn_model is None,
        })

    except Exception as e:
        import traceback
        print(f"[ERROR] 분석 실패: {traceback.format_exc()}")
        return jsonify({'error': str(e), 'detail': traceback.format_exc()}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

# ▼▼▼ [추가] Step2 시선/깜빡임 분석 라우트 ▼▼▼
@app.route('/analyze/gaze-blink', methods=['POST'])
def analyze_gaze_blink():
    # mediapipe/cv2는 이 라우트에서만 필요해서 여기서 지연 import (모듈 상단 import를
    # 뺀 이유는 파일 상단 주석 참고) — /health·로그인·커뮤니티 라우트만 받는 워커는
    # 이 라우트가 한 번도 안 불리면 mediapipe/cv2를 아예 메모리에 안 올린다.
    from step2.landmark_face_points import (
        extract_landmarks_from_video, LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
        POSE_LANDMARK_IDX, LEFT_IRIS_IDX, RIGHT_IRIS_IDX,
    )
    from step2.gaze_analyzer import detect_gaze_segments

    user = get_current_user()
    if user is None:
        return jsonify({'error': '로그인이 필요합니다'}), 401

    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다'}), 400

    video_file = request.files['file']
    tmp_path = None
    converted_path = None
    try:
        tmp_path = save_temp_video(video_file)
        analysis_path = ensure_mp4(tmp_path)
        if analysis_path != tmp_path:
            converted_path = analysis_path  # mov/webm -> mp4 변환 결과, 별도 정리 필요

        frames = extract_landmarks_from_video(analysis_path)
        duration_sec = frames[-1]["t"] if frames else 0

        # 앞 5초를 캘리브레이션(개인별 기준값) 구간으로 사용
        calib_frames = [f for f in frames if f["t"] <= 5.0]
        baseline_ear = calibrate_baseline_ear(calib_frames, LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX)
        baseline_yaw, baseline_pitch = calibrate_baseline_gaze(
            calib_frames, POSE_LANDMARK_IDX, LEFT_IRIS_IDX, RIGHT_IRIS_IDX,
            LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
        )

        # 눈 깜빡임 — blendshape(신경망 학습 신호)가 있으면 그걸 우선 쓰고,
        # 없는 영상(구버전 모델 등)이면 EAR 기하학 계산으로 폴백
        blink_series = compute_blink_blendshape_series(frames)
        if any(score is not None for _, score in blink_series):
            blinks = detect_blinks_from_blendshape(blink_series)
        else:
            ear_series = compute_ear_series(frames, LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX)
            blinks = detect_blinks(ear_series, baseline_ear)
        blink_result = score_blink_rate(len(blinks), duration_sec)

        # 시선 고정
        gaze_segments = detect_gaze_segments(
            frames, POSE_LANDMARK_IDX, LEFT_IRIS_IDX, RIGHT_IRIS_IDX,
            LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
            baseline_yaw=baseline_yaw, baseline_pitch=baseline_pitch,
        )
        gaze_result = score_gaze_segments(gaze_segments)

        # 표정 (미소/긴장)
        expression_series = compute_expression_series(frames)
        expression_summary = summarize_expression(expression_series)
        expression_result = score_expression(
            expression_summary["smile_ratio"], expression_summary["tension_ratio"]
        )
        expression_segments = detect_expression_segments(expression_series)

        # 이번 결과를 저장하고, 비교에 쓸 "직전 기록"을 먼저 가져온다 (저장 전에 조회해야
        # 방금 넣은 레코드 자신이 previous로 잡히지 않는다).
        previous_row = (
            Stage2Result.query
            .filter_by(user_id=user.id)
            .order_by(Stage2Result.created_at.desc())
            .first()
        )
        previous_entry = previous_row.to_entry() if previous_row else None

        db.session.add(Stage2Result(
            user_id=user.id,
            blink_rate_per_min=blink_result['rate_per_min'],
            blink_status=blink_result['status'],
            blink_score=blink_result['score'],
            avg_fixation_sec=gaze_result['avg_fixation_sec'],
            gaze_score=gaze_result['score'],
            smile_ratio=expression_summary['smile_ratio'],
            tension_ratio=expression_summary['tension_ratio'],
            expression_score=expression_result['score'],
            expression_status=expression_result['status'],
        ))
        db.session.commit()

        # 지표별 하이라이트 클립 — 원본을 통째로 보여주는 대신, 해당 순간만 편집한 짧은 영상.
        # 시선/표정은 "눈에 띄는" 쪽(회피/미소·긴장)만, 깜빡임은 이벤트 전부를 재료로 쓴다.
        blink_highlight = build_highlight_clip(
            analysis_path, [(e["start"], e["end"]) for e in blinks]
        )
        gaze_highlight = build_highlight_clip(
            analysis_path, [(s["start"], s["end"]) for s in gaze_segments if s["type"] == "aversion"]
        )
        expression_highlight = build_highlight_clip(
            analysis_path, [(s["start"], s["end"]) for s in expression_segments if s["type"] != "neutral"]
        )

        return jsonify({
            "blink": {**blink_result, "events": blinks, "highlight": blink_highlight},
            "gaze": {**gaze_result, "segments": gaze_segments, "highlight": gaze_highlight},
            "expression": {
                **expression_result, **expression_summary,
                "segments": expression_segments, "highlight": expression_highlight,
            },
            "previous": previous_entry,
        })

    except Exception as e:
        import traceback
        print(f"[ERROR] gaze-blink 분석 실패: {traceback.format_exc()}")
        return jsonify({'error': str(e), 'detail': traceback.format_exc()}), 500

    finally:
        for p in (tmp_path, converted_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass


@app.route('/analyze/gaze-blink/latest', methods=['GET'])
def get_latest_gaze_blink():
    """로그인한 유저의 가장 최근 2단계(표정·시선) 결과 — 새로 영상을 분석하지 않고
    이미 저장된 값만 조회. 3단계(모의면접) 난이도 산정(combineAnxietyScore)에서
    쓸 stage2Avg를 여기서 가져온다 — 영상 재분석 없이 DB에 저장된 점수만 읽음.
    1단계는 아직 DB 테이블이 없어서(localProgress.ts 참고) 대응하는 GET이 없다 —
    1단계도 DB로 옮겨지면 같은 /analyze/<stage>/latest 형태로 이름을 맞출 것."""
    user = get_current_user()
    if user is None:
        return jsonify({'error': '로그인이 필요합니다'}), 401

    row = (
        Stage2Result.query
        .filter_by(user_id=user.id)
        .order_by(Stage2Result.created_at.desc())
        .first()
    )
    if row is None:
        return jsonify({'result': None})

    overall_score = round((row.blink_score + row.gaze_score + row.expression_score) / 3, 1)

    return jsonify({
        'result': {
            'at': row.created_at.isoformat(),
            'blinkScore': row.blink_score,
            'blinkStatus': row.blink_status,
            'gazeScore': row.gaze_score,
            'expressionScore': row.expression_score,
            'expressionStatus': row.expression_status,
            'overallScore': overall_score,
        }
    })
# ▲▲▲ [추가 끝] ▲▲▲

@app.route('/community/posts', methods=['GET'])
def list_posts():
    """'이야기' 게시판 · 전체 글 목록 (최신순). 로그인 안 해도 볼 수 있음."""
    current_user = get_current_user()
    posts = Post.query.order_by(Post.created_at.desc()).all()
    result = []
    for p in posts:
        count = Comment.query.filter_by(post_id=p.id).count()
        result.append(p.to_dict(comment_count=count, current_user=current_user))
    return jsonify({'posts': result})


@app.route('/community/posts', methods=['POST'])
def create_post():
    """'이야기' 게시판 · 글쓰기. 로그인 필요."""
    user = get_current_user()
    if not user:
        return jsonify({'error': '로그인이 필요합니다.'}), 401

    data = request.get_json(silent=True) or {}
    title = (data.get('title') or '').strip()
    content = (data.get('content') or '').strip()
    if not title or not content:
        return jsonify({'error': '제목과 내용을 입력해주세요.'}), 400

    post = Post(user_id=user.id, title=title, content=content)
    db.session.add(post)
    db.session.commit()
    return jsonify(post.to_dict(current_user=user)), 201


@app.route('/community/posts/<int:post_id>', methods=['GET'])
def get_post(post_id):
    """'이야기' 게시판 · 글 상세 + 댓글 목록."""
    current_user = get_current_user()
    post = Post.query.get(post_id)
    if not post:
        return jsonify({'error': '글을 찾을 수 없습니다.'}), 404

    comments = Comment.query.filter_by(post_id=post_id).order_by(Comment.created_at.asc()).all()
    data = post.to_dict(comment_count=len(comments), current_user=current_user)
    data['comments'] = [c.to_dict(current_user=current_user) for c in comments]
    return jsonify(data)


@app.route('/community/posts/<int:post_id>', methods=['DELETE'])
def delete_post(post_id):
    """'이야기' 게시판 · 글 삭제. 본인 글만 가능."""
    user = get_current_user()
    if not user:
        return jsonify({'error': '로그인이 필요합니다.'}), 401

    post = Post.query.get(post_id)
    if not post:
        return jsonify({'error': '글을 찾을 수 없습니다.'}), 404
    if post.user_id != user.id:
        return jsonify({'error': '본인 글만 삭제할 수 있습니다.'}), 403

    Comment.query.filter_by(post_id=post_id).delete()
    db.session.delete(post)
    db.session.commit()
    return jsonify({'message': '삭제됐습니다.'})


@app.route('/community/posts/<int:post_id>/comments', methods=['POST'])
def create_comment(post_id):
    """'이야기' 게시판 · 댓글 작성. 로그인 필요."""
    user = get_current_user()
    if not user:
        return jsonify({'error': '로그인이 필요합니다.'}), 401

    post = Post.query.get(post_id)
    if not post:
        return jsonify({'error': '글을 찾을 수 없습니다.'}), 404

    data = request.get_json(silent=True) or {}
    content = (data.get('content') or '').strip()
    if not content:
        return jsonify({'error': '댓글 내용을 입력해주세요.'}), 400

    comment = Comment(post_id=post_id, user_id=user.id, content=content)
    db.session.add(comment)
    db.session.commit()
    return jsonify(comment.to_dict(current_user=user)), 201


@app.route('/analyze/feedback', methods=['POST'])
def analyze_feedback():
    """Step 1 · LLM 코칭 피드백 — /analyze 결과(JSON)를 받아 한 문단 생성."""
    result = request.get_json(silent=True)
    if not result or 'scores' not in result:
        return jsonify({'error': 'scores가 필요합니다'}), 400
    text = generate_feedback(result)
    return jsonify({'feedback': text, 'source': 'llm' if text else 'template'})


@app.route('/interview/next-question', methods=['POST'])
def interview_next_question():
    """Step 3 · 다음 면접 질문 생성.
    body: {
      "tier": "warmup" | "standard" | "practice",
      "previous_questions": [str, ...],
      "previous_answer": str (선택 — STT로 변환된 방금 답변 텍스트, 꼬리질문용)
    }
    """
    data = request.get_json(silent=True) or {}
    tier = data.get('tier', 'standard')
    previous_questions = data.get('previous_questions', [])
    previous_answer = data.get('previous_answer')

    if tier not in ('warmup', 'standard', 'practice'):
        return jsonify({'error': "tier는 'warmup' | 'standard' | 'practice' 중 하나여야 합니다"}), 400

    question, source = generate_question(tier, previous_questions, previous_answer)
    return jsonify({'question': question, 'source': source})


@app.route('/interview/tts', methods=['POST'])
def interview_tts():
    """Step 3 · 질문 텍스트 → mp3 음성. body: { "text": "..." }"""
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'text가 필요합니다'}), 400

    audio_bytes = synthesize_speech(text)
    if audio_bytes is None:
        return jsonify({'error': 'TTS 생성 실패 (키 없음 또는 호출 오류)'}), 502

    return Response(audio_bytes, mimetype='audio/mpeg')


@app.route('/interview/transcribe', methods=['POST'])
def interview_transcribe():
    """Step 3 · 답변 영상 → 텍스트 변환 (Whisper 재사용).
    꼬리질문 생성에 쓸 '방금 사용자가 뭐라고 답했는지'를 얻기 위함.
    """
    load_models()  # 최초 호출 시에만 실제로 torch/whisper를 로드 (이후엔 즉시 리턴)

    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다'}), 400

    if not (WHISPER_AVAILABLE and whisper_model is not None):
        # whisper 없는 환경 — 에러 대신 빈 텍스트로 응답 (꼬리질문 없이 진행되게)
        return jsonify({'text': '', 'available': False})

    video_file = request.files['file']
    tmp_path = None
    converted_path = None
    try:
        tmp_path = save_temp_video(video_file)
        analysis_path = ensure_mp4(tmp_path)
        if analysis_path != tmp_path:
            converted_path = analysis_path

        y16, _ = librosa.load(analysis_path, sr=16000, mono=True)
        result = whisper_model.transcribe(y16, language="ko")
        text = result.get("text", "").strip()

        return jsonify({'text': text, 'available': True})

    except Exception as e:
        import traceback
        print(f"[ERROR] transcribe 실패: {traceback.format_exc()}")
        return jsonify({'error': str(e), 'text': '', 'available': False}), 500

    finally:
        for p in (tmp_path, converted_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass


if __name__ == '__main__':
    load_models()  # db.create_all()은 이제 모듈 로드 시점에 이미 실행됨
    app.run(host='0.0.0.0', port=5000, debug=True)
