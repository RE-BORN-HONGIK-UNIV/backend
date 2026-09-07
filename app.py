"""
Re-born 보이스 터치 - Flask 백엔드 (최종 버전)
- CNN: 슬라이딩 윈도우(3초/1초 stride) 추론
- 채움말: librosa 소리구간 + Whisper word timestamp 결합
- 멈춤: librosa 기반 불안한 멈춤 분석
"""

import os
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
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from flask_cors import CORS
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Step2 시선/깜빡임 모듈 import
from step2.landmark_face_points import (
    extract_landmarks_from_video, LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
    POSE_LANDMARK_IDX, LEFT_IRIS_IDX, RIGHT_IRIS_IDX,
)
from step2.blink_analyzer import (
    compute_ear_series, detect_blinks,
    compute_blink_blendshape_series, detect_blinks_from_blendshape,
)
from step2.gaze_analyzer import detect_gaze_segments
from step2.expression_analyzer import compute_expression_series, summarize_expression, detect_expression_segments
from step2.scoring import score_blink_rate, score_gaze_segments, score_expression
from step2.set_baseline import calibrate_baseline_ear, calibrate_baseline_gaze

# Step1 LLM 코칭 피드백 (선택적 — 키 없으면 자동 폴백)
from step1_llm_feedback import generate_feedback

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[WARNING] whisper 없음 - 발화 유창성 0점 고정")

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:qaqa8150!@localhost/reborn_db'
app.config['SECRET_KEY'] = 'dev-secret-key-change-later'
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
class SpeechAnxietyCNN(nn.Module):
    def __init__(self, dropout=0.5, size='large'):
        super(SpeechAnxietyCNN, self).__init__()
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

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

cnn_model = None
whisper_model = None

def load_models():
    global cnn_model, whisper_model
    if os.path.exists(MODEL_PATH):
        try:
            m = SpeechAnxietyCNN(dropout=0.5, size='large')
            m.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            m.eval()
            cnn_model = m
            print(f"[OK] CNN 모델 로드 완료: {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] CNN 모델 로드 실패: {e}")
    else:
        print(f"[WARNING] CNN 모델 파일 없음: {MODEL_PATH}")

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
    except Exception as e:
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
# ▲▲▲ [추가 끝] ▲▲▲

@app.route('/analyze/feedback', methods=['POST'])
def analyze_feedback():
    """Step 1 · LLM 코칭 피드백 — /analyze 결과(JSON)를 받아 한 문단 생성."""
    result = request.get_json(silent=True)
    if not result or 'scores' not in result:
        return jsonify({'error': 'scores가 필요합니다'}), 400
    text = generate_feedback(result)
    return jsonify({'feedback': text, 'source': 'llm' if text else 'template'})


if __name__ == '__main__':
    load_models()
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=False)