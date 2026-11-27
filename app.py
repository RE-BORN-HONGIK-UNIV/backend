"""공통 · 서버 — Flask 앱 하나에 DB 표, 로그인·게스트, 1·2·3단계 분석, 면접 기록, 코치·하루 대화 API를 모은다.
무거운 라이브러리(torch·whisper·mediapipe)는 쓰는 함수 안에서 불러 가벼운 API의 메모리를 아낀다. 배경: docs/공통_설계노트.md
"""

import os
from dotenv import load_dotenv
load_dotenv(encoding='utf-8')

import base64
import secrets
import subprocess
import tempfile
import threading
import time
import numpy as np
import librosa
from flask import Flask, request, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
import json
from flask_cors import CORS
from sqlalchemy.exc import IntegrityError
from step3.tts import synthesize_speech
from db_url import normalize_database_url

# 2단계 중 numpy만 쓰는 모듈은 여기서, mediapipe·cv2가 필요한 모듈은 analyze_gaze_blink() 안에서 불러온다
from step2.blink_analyzer import (
    compute_ear_series, detect_blinks,
    compute_blink_blendshape_series, detect_blinks_from_blendshape,
)
from step2.expression_analyzer import compute_expression_series, summarize_expression, detect_expression_segments
from step2.scoring import score_blink_rate, score_gaze_segments, score_expression
from step2.set_baseline import calibrate_baseline_ear, calibrate_baseline_gaze
from step2.live_payload import parse_live_result

from step1.analyze_filler import analyze_filler, analyze_calibration
from step1.analyze_pause import analyze_pause
from step1.report_detail import build_report_detail
from step1.llm_feedback import generate_feedback
# CNN 입력 손질은 voice-cnn에서 복사한 것 — 학습 때와 똑같이 손질해야 한다
from step1 import audio_prep, energy_rule
from step1.stage1_score import overall_score as stage1_overall_score, window_score

from step3.interview_question import clean_about_me, generate_question
from step3.anxiety import anxiety_score, tier as anxiety_tier
from step3.anxiety import blink_item as anxiety_blink_item, gaze_item as anxiety_gaze_item, worse as anxiety_worse

from coach.agent import MODEL as COACH_MODEL, generate_note as generate_coach_note
from coach.report import generate_report_coach, parse_report_request
from coach.practice import RateLimiter, generate_hint, parse_hint_request
from haru.talk import generate_reply, parse_profile_request, parse_reply_request, summarize_profile

from step3.session_payload import (
    MAX_TURNS_PER_SESSION, parse_answer, parse_session_start, parse_turn,
)

# load_models()가 whisper를 불러오기 전까지는 "없음"으로 둔다
WHISPER_AVAILABLE = False

def _require_env(key):
    """비밀 값(DB 주소·서명 키)을 환경변수에서 읽고, 없으면 .env.example을 안내하며 멈춘다."""
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(
            f"환경변수 {key}가 설정되지 않았습니다. backend/.env.example을 참고해 "
            f"backend/.env 파일을 만들어주세요."
        )
    return value


app = Flask(__name__)
CORS(app)

# 드라이버를 psycopg2로 못 박는다 (db_url.py)
_database_url = normalize_database_url(_require_env('DATABASE_URL'))
app.config['SQLALCHEMY_DATABASE_URI'] = _database_url
app.config['SECRET_KEY'] = _require_env('SECRET_KEY')
db = SQLAlchemy(app)

class User(db.Model):
    """회원·게스트 계정. 게스트는 이메일 도메인으로 구분한다(GUEST_EMAIL_DOMAIN)."""
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(80), nullable=False)
    nickname = db.Column(db.String(80), nullable=True)
    birthdate = db.Column(db.String(10), nullable=True)
    terms_agreed = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)


class Stage1Result(db.Model):
    """1단계 목소리 점수 — 3단계 난이도와 코치 이력 조회에 쓴다."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    # 오각형 5축 점수 (0~100, 높을수록 안정적) — /analyze 응답의 scores와 동일 키
    stability = db.Column(db.Float, nullable=False)
    fluency = db.Column(db.Float, nullable=False)
    pause_ctrl = db.Column(db.Float, nullable=False)
    continuity = db.Column(db.Float, nullable=False)
    calm = db.Column(db.Float, nullable=False)
    overall_score = db.Column(db.Float, nullable=False)

    def to_entry(self):
        """프론트가 쓰는 camelCase 모양으로 돌려준다."""
        return {
            'at': self.created_at.isoformat(),
            'stability': self.stability,
            'fluency': self.fluency,
            'pauseCtrl': self.pause_ctrl,
            'continuity': self.continuity,
            'calm': self.calm,
            'overallScore': self.overall_score,
        }


class Stage2VoiceResult(db.Model):
    """2단계 통화의 목소리 점수 — 3단계 불안도에만 쓴다(기존 DB에 칸을 못 더해 새 표로 둠)."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    stability = db.Column(db.Float, nullable=False)
    fluency = db.Column(db.Float, nullable=False)
    pause_ctrl = db.Column(db.Float, nullable=False)
    continuity = db.Column(db.Float, nullable=False)
    calm = db.Column(db.Float, nullable=False)


class Stage2Result(db.Model):
    """2단계 표정·시선 결과 — "지난번 대비" 비교와 3단계 난이도에 쓴다."""
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
        """프론트 Stage2Entry 모양으로 돌려준다(previous 비교용)."""
        return {
            "at": self.created_at.isoformat(),
            "blinkRatePerMin": self.blink_rate_per_min,
            "avgFixationSec": self.avg_fixation_sec,
            "smileRatio": self.smile_ratio,
            "tensionRatio": self.tension_ratio,
        }


class InterviewSession(db.Model):
    """3단계 면접 1회. 중간에 나가도 기록은 남고 completed_at만 빈다."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    tier = db.Column(db.String(20), nullable=False)  # warmup / standard / practice (면접관)
    started_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    completed_at = db.Column(db.DateTime, nullable=True)

    # 대답 글·인용·다듬은 대답이 들어 있어 세션을 지우면 함께 지운다
    turns = db.relationship(
        'InterviewTurn', backref='session', cascade='all, delete-orphan',
        order_by='InterviewTurn.order_index',
    )
    coach_note = db.relationship('CoachNote', backref='session', uselist=False, cascade='all, delete-orphan')
    report_coach = db.relationship('ReportCoach', backref='session', uselist=False, cascade='all, delete-orphan')

    def to_summary(self):
        """질문·대답 본문 없이 요약만 돌려준다."""
        return {
            'id': self.id,
            'tier': self.tier,
            'startedAt': self.started_at.isoformat(),
            'completedAt': self.completed_at.isoformat() if self.completed_at else None,
            'turnCount': len(self.turns),
        }

    def to_detail(self):
        """요약에 질문·대답 전체를 붙여 돌려준다."""
        return {**self.to_summary(), 'turns': [t.to_entry() for t in self.turns]}


class InterviewTurn(db.Model):
    """질문 1개와 대답 글(사용자가 확인한 음성 인식). 영상·음성 파일은 저장하지 않는다."""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('interview_session.id'), nullable=False, index=True)
    order_index = db.Column(db.Integer, nullable=False)  # 0부터, 질문이 나온 순서
    kind = db.Column(db.String(20), nullable=False)      # main(기본 질문) / follow_up(꼬리질문)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    answered_at = db.Column(db.DateTime, nullable=True)

    def to_entry(self):
        """프론트가 쓰는 모양으로 돌려준다."""
        return {
            'id': self.id,
            'order': self.order_index,
            'kind': self.kind,
            'question': self.question,
            'answer': self.answer,
            'askedAt': self.created_at.isoformat(),
            'answeredAt': self.answered_at.isoformat() if self.answered_at else None,
        }


class CoachNote(db.Model):
    """면접 직후 코치 노트 JSON. 면접당 1개라 다시 열어도 AI를 다시 부르지 않는다."""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('interview_session.id'), nullable=False, unique=True)
    content = db.Column(db.Text, nullable=False)       # 노트 JSON (coach/schema.py 형태)
    source = db.Column(db.String(20), nullable=False)  # llm / fallback / care
    model = db.Column(db.String(60), nullable=True)    # source가 llm일 때만
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)


class ReportCoach(db.Model):
    """리포트 3·4번 칸(다듬은 대답·다음 연습 계획) JSON. 면접당 1개."""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('interview_session.id'), nullable=False, unique=True)
    content = db.Column(db.Text, nullable=False)       # 결과 JSON
    source = db.Column(db.String(20), nullable=False)  # llm / fallback / care
    model = db.Column(db.String(60), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)


class Post(db.Model):
    """'이야기' 자유 게시판 글."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def to_dict(self, comment_count=0, current_user=None):
        """작성자 이름과 본인 글 여부를 붙여 돌려준다."""
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
        """작성자 이름과 본인 댓글 여부를 붙여 돌려준다."""
        author = User.query.get(self.user_id)
        return {
            'id': self.id,
            'content': self.content,
            'author': (author.nickname or author.name) if author else '탈퇴한 사용자',
            'created_at': self.created_at.isoformat(),
            'is_own': bool(current_user and current_user.id == self.user_id),
        }


def get_current_user():
    """Bearer 토큰(JWT)을 검증해 User를 돌려준다. 없거나 잘못되면 None."""
    header = request.headers.get('Authorization', '')
    if not header.startswith('Bearer '):
        return None
    token = header[len('Bearer '):]
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
    except jwt.PyJWTError:
        return None
    return User.query.get(payload.get('user_id'))


# gunicorn은 __main__을 거치지 않으므로 불러오는 시점에 표를 만든다. DB가 없어도 서버는 켠다
try:
    with app.app_context():
        db.create_all()
except Exception as e:
    print(f"[WARNING] DB 연결 실패, DB 없이 서버 실행: {e}")

# ── CNN 모델 (라벨별 이진 모델) ──────────────────────────────────────
# models/<label>/에 model.pth와 config.json(손질 값·정규화·기준선)을 손으로 넣는다
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
CNN_LABELS = ('prolongation', 'tremor', 'energy')
cnn_models = {}  # label -> (model, cfg). 없는 라벨은 pending
whisper_model = None
_models_loaded = False
# 미리 불러오는 중에 요청이 와도 반쯤 불린 모델을 쓰지 않게 한다
_models_lock = threading.Lock()

def load_models():
    """CNN·Whisper를 처음 필요할 때 한 번만 불러온다(모델을 쓰는 API가 맨 앞에서 부름)."""
    global whisper_model, WHISPER_AVAILABLE, _models_loaded
    if _models_loaded:
        return
    with _models_lock:
        if _models_loaded:
            return
        _load_models_locked()
        _models_loaded = True


def _load_models_locked():
    """라벨별 CNN과 Whisper를 불러온다. 실패한 라벨은 pending으로 두고 서버는 그대로 켠다."""
    global whisper_model, WHISPER_AVAILABLE

    for label in CNN_LABELS:
        folder = os.path.join(MODELS_DIR, label)
        if not os.path.exists(os.path.join(folder, 'model.pth')):
            print(f"[WARNING] CNN 모델 없음(pending): {folder}")
            continue
        try:
            # 모델 파일이 있을 때만 torch를 메모리에 올린다
            import torch
            from step1.model import SmallCNN
            with open(os.path.join(folder, 'config.json'), encoding='utf-8') as fp:
                cfg = json.load(fp)
            m = SmallCNN()
            m.load_state_dict(torch.load(os.path.join(folder, 'model.pth'), map_location='cpu', weights_only=True)['state_dict'])
            m.eval()
            cnn_models[label] = (m, cfg)
            print(f"[OK] CNN 모델 로드 완료: {folder}")
        except Exception as e:
            print(f"[ERROR] !!!!! CNN 모델 로드 실패, pending 처리: {folder} — {e}")

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

# PRELOAD_MODELS=1이면 켜지자마자 뒤에서 모델을 불러 첫 사용자의 1~3분 대기를 없앤다. 메모리 2GB 이상에서만 켠다
if os.environ.get('PRELOAD_MODELS') == '1':
    threading.Thread(target=load_models, name='preload-models', daemon=True).start()


def save_temp_file(file_storage):
    """업로드한 음성을 임시 폴더에 wav 이름으로 저장한다."""
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"reborn_audio_{os.getpid()}.wav")
    file_storage.save(tmp_path)
    print(f"[INFO] 임시 파일 저장: {tmp_path}")
    return tmp_path


def save_temp_video(file_storage):
    """업로드한 영상을 원래 확장자 그대로 임시 저장한다."""
    tmp_dir = tempfile.gettempdir()
    ext = os.path.splitext(file_storage.filename or "")[1].lower() or ".mp4"
    tmp_path = os.path.join(tmp_dir, f"reborn_video_{os.getpid()}{ext}")
    file_storage.save(tmp_path)
    print(f"[INFO] 임시 영상 저장: {tmp_path}")
    return tmp_path


def ensure_mp4(video_path):
    """mp4가 아니면 ffmpeg으로 바꾼 경로를, 실패하면 원본 경로를 돌려준다(바뀐 파일은 부른 쪽이 지운다)."""
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
    """구간들만 잘라 이어 붙인 짧은 mp4를 data URL로 돌려준다(최대 max_total_sec초). 없거나 실패하면 None."""
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

# ── CNN 추론 ─────────────────────────────────────────────────────────
def predict_cnn(audio_path):
    """준비된 라벨마다 (3초 조각 확률 최댓값, 조각 수, [(시작초, 확률)])을 구한다. 거의 무음인 조각은 건너뛴다."""
    import torch
    out = {}
    for label, (model, cfg) in cnn_models.items():
        y = audio_prep.load(audio_path, cfg)
        pairs = [(s, audio_prep.logmel(y, s, cfg)) for s in audio_prep.starts(y, cfg, cfg['judge_hop_sec'])]
        pairs = [(s, audio_prep.normalize(m, cfg)) for s, m in pairs if m is not None]
        if not pairs:
            out[label] = (0.0, 0, [])
            continue
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(np.stack([m for _, m in pairs])).float().unsqueeze(1))).reshape(-1).tolist()
        if 'fade_db' in cfg:
            # 에너지: 조각 안에서 실제로 fade_db 이상 작아진 조각만 인정 (step1/energy_rule.py)
            n = int(cfg['win_sec'] * cfg['sr'])
            probs = [p if energy_rule.faded(y[int(round(s * cfg['sr'])):int(round(s * cfg['sr'])) + n], cfg['fade_db']) else 0.0
                     for (s, _), p in zip(pairs, probs)]
        out[label] = (max(probs), len(pairs), [(s, p) for (s, _), p in zip(pairs, probs)])
    if 'energy' in out and 'prolongation' in out and 'fade_db' in cnn_models['energy'][1]:
        # 에너지: 연장으로 잡힌 조각은 뺀다 (말 늘이기를 작아짐으로 헷갈리는 오탐)
        wins = energy_rule.drop_prolonged(out['energy'][2], out['prolongation'][2], cnn_models['prolongation'][1]['threshold'])
        out['energy'] = (max((p for _, p in wins), default=0.0), out['energy'][1], wins)
    print(f"[CNN] { {k: v[:2] for k, v in out.items()} }")
    return out


# ── API ──────────────────────────────────────────────────────────────
@app.route('/api/signup', methods=['POST'])
def signup():
    """회원가입. 필수 항목·약관 동의·이메일 중복을 확인한다."""
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
    """로그인. 맞으면 7일짜리 토큰을 준다."""
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'error': '이메일 또는 비밀번호가 올바르지 않습니다.'}), 401

    return jsonify({'token': _issue_token(user, days=7), 'name': user.name})


def _issue_token(user, days):
    """user_id와 만료일을 담은 JWT를 만든다."""
    return jwt.encode(
        {
            'user_id': user.id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=days),
        },
        app.config['SECRET_KEY'],
        algorithm='HS256',
    )


# ── 로그인 없이 시작하기 (게스트) ─────────────────────────────────────
# 이름 없는 계정을 만들어 토큰을 주면 로그인이 필요한 API를 그대로 쓸 수 있다. 이유와 숫자 근거: docs/공통_설계노트.md
# .invalid는 실제 메일과 겹치지 않는 예약 도메인(RFC 2606)이다
GUEST_EMAIL_DOMAIN = 'guest.reborn.invalid'
GUEST_TOKEN_DAYS = 30
_guest_limiter = RateLimiter(limit=20)


def _client_ip():
    """접속 IP. Render는 프록시 뒤라 X-Forwarded-For 첫 값을 쓴다."""
    forwarded = request.headers.get('X-Forwarded-For', '')
    return forwarded.split(',')[0].strip() or request.remote_addr or 'unknown'


@app.route('/api/guest', methods=['POST'])
def start_guest():
    """게스트 계정을 만들고 30일짜리 토큰을 준다(IP당 시간당 20개)."""
    if not _guest_limiter.allow(('guest', _client_ip())):
        return jsonify({'error': '잠시 뒤에 다시 시도해주세요.'}), 429
    user = User(
        email=f'guest-{secrets.token_hex(8)}@{GUEST_EMAIL_DOMAIN}',
        password_hash=generate_password_hash(secrets.token_urlsafe(32)),
        name='게스트',
        terms_agreed=False,
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({'token': _issue_token(user, days=GUEST_TOKEN_DAYS), 'name': user.name, 'guest': True}), 201


NICKNAME_MAX_LEN = 20


@app.route('/api/me/nickname', methods=['PUT'])
def set_nickname():
    """화면에서 불러 줄 이름을 저장한다(게스트는 시작 직후 정함). 커뮤니티 작성자 이름에도 쓰인다."""
    user = get_current_user()
    if user is None:
        return jsonify({'error': '로그인이 필요합니다'}), 401
    nickname = ((request.get_json(silent=True) or {}).get('nickname') or '')
    nickname = nickname.strip() if isinstance(nickname, str) else ''
    if not nickname:
        return jsonify({'error': '불러 드릴 이름을 입력해주세요.'}), 400
    if len(nickname) > NICKNAME_MAX_LEN:
        return jsonify({'error': f'이름은 {NICKNAME_MAX_LEN}자까지 쓸 수 있어요.'}), 400
    user.nickname = nickname
    db.session.commit()
    return jsonify({'name': nickname})

@app.route('/health', methods=['GET'])
def health():
    """서버 상태와 모델이 불렸는지 알려 준다."""
    return jsonify({
        'status': 'ok',
        'cnn_model_loaded': bool(cnn_models),
        'whisper_loaded': whisper_model is not None,
    })

@app.route('/api/stage1/calibrate', methods=['POST'])
def stage1_calibrate():
    """1단계 마이크 확인 녹음으로 개인 기준값을 잰다. 점수에는 안 들어가고 /analyze의 채움말·멈춤 기준 보정에만 쓴다."""
    load_models()

    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다'}), 400

    f = request.files['file']
    tmp_path = None
    try:
        tmp_path = save_temp_file(f)
        calibration_result = analyze_calibration(tmp_path, whisper_model=whisper_model)
        return jsonify(calibration_result)
    except Exception as e:
        import traceback
        print(f"[ERROR] 캘리브레이션 실패: {traceback.format_exc()}")
        return jsonify({'error': str(e), 'detail': traceback.format_exc()}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

@app.route('/analyze', methods=['POST'])
def analyze():
    """녹음 하나로 목소리 5개 점수와 리포트 세부 값을 낸다. 1·2단계 결과는 로그인한 사용자 것만 저장한다."""
    load_models()

    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다'}), 400

    f = request.files['file']
    allowed = {'.wav', '.mp3', '.m4a', '.ogg', '.webm', '.flac'}
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in allowed:
        return jsonify({'error': f'지원하지 않는 형식: {ext}'}), 400

    # 마이크 확인 결과가 없으면 이 녹음 자체에서 개인 기준값을 추정한다
    calibration = None
    calibration_raw = request.form.get('calibration')
    if calibration_raw:
        try:
            import json
            calibration = json.loads(calibration_raw)
        except (ValueError, TypeError):
            print("[WARNING] calibration 파싱 실패, 무시하고 진행")

    # 3단계 대답도 같은 분석을 쓰지만 저장은 1·2단계만 한다 (step1/analyze_options.py)
    from step1.analyze_options import parse_analyze_stage, saves_stage1_result
    stage = parse_analyze_stage(request.form.get('stage'))

    tmp_path = None
    try:
        tmp_path = save_temp_file(f)

        # 1) CNN → 음성 안정성·발화 지속성·발화 에너지. 모델이 없는 라벨은 None(0은 '아주 좋음'으로 읽힌다)
        cnn_result = predict_cnn(tmp_path)
        cnn_probs, cnn_scores, detected = {}, {}, {}
        for label in CNN_LABELS:
            p = cnn_result[label][0] if label in cnn_result else None
            cnn_probs[label]  = None if p is None else round(p * 100, 1)
            cnn_scores[label] = None if p is None else window_score(cnn_result[label][2], cnn_models[label][1]['threshold'])
            detected[label]   = None if p is None else p >= cnn_models[label][1]['threshold']
        # config.json의 status가 reference면 시험을 통과하지 못한 참고용 모델이다(떨림 v1)
        model_status = {label: cnn_models[label][1].get('status', 'ready') if label in cnn_models else 'pending'
                        for label in CNN_LABELS}
        stability, continuity, calm = cnn_scores['tremor'], cnn_scores['prolongation'], cnn_scores['energy']
        window_count = max((n for _, n, _ in cnn_result.values()), default=0)

        # 2) 채움말 (librosa + Whisper) → 발화 유창성
        filler_result = analyze_filler(tmp_path, whisper_model=whisper_model, calibration=calibration)
        fluency = filler_result['fluency_score']

        # 3) 멈춤 (librosa) → 침묵 조절력
        pause_result = analyze_pause(tmp_path, calibration=calibration)
        pause_ctrl = pause_result['pause_score']

        # 4) 리포트 세부 값. 실패해도 점수 응답은 그대로 보낸다
        report_detail = {}
        try:
            y_report, sr_report = librosa.load(tmp_path, sr=16000, mono=True)
            report_detail = build_report_detail(
                y_report, sr_report, filler_result['events'], pause_result['events'], filler_result['words'],
                {label: {'windows': cnn_result[label][2], 'threshold': cnn_models[label][1]['threshold'],
                         'win_sec': cnn_models[label][1]['win_sec']} for label in cnn_result},
            )
        except Exception:
            import traceback
            print(f"[WARNING] report_detail 생성 실패: {traceback.format_exc()}")

        # CNN 세 모델이 다 있을 때만 저장한다(빈 점수는 난이도를 왜곡). 저장이 실패해도 응답은 보낸다
        user = get_current_user()
        if stage in (1, 2) and user is not None and len(cnn_models) == len(CNN_LABELS):
            try:
                scores_to_save = {
                    'stability': stability, 'fluency': fluency, 'pause_ctrl': pause_ctrl,
                    'continuity': continuity, 'calm': calm,
                }
                if saves_stage1_result(stage):
                    db.session.add(Stage1Result(
                        user_id=user.id,
                        overall_score=stage1_overall_score(scores_to_save),
                        **scores_to_save,
                    ))
                else:
                    db.session.add(Stage2VoiceResult(user_id=user.id, **scores_to_save))
                db.session.commit()
            except Exception:
                db.session.rollback()
                import traceback
                print(f"[WARNING] {stage}단계 목소리 결과 저장 실패: {traceback.format_exc()}")

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
            'detected': detected,
            'model_status': model_status,
            'filler_detail': {
                'filler_count':       filler_result['filler_count'],
                'sound_segment_count': filler_result['sound_segment_count'],
                'filler_ratio_pct':   round(filler_result['filler_ratio'] * 100, 1),
                'events':             report_detail.get('filler_events'),
            },
            'pause_detail': {
                'pause_count':             pause_result['pause_count'],
                'anxious_pause_count':     pause_result['anxious_pause_count'],
                'anxious_pause_total_sec': pause_result['anxious_pause_total_sec'],
                'anxious_pause_ratio':     pause_result['anxious_pause_ratio'],
                'total_duration_sec':      pause_result['total_duration_sec'],
                'long_pauses':             report_detail.get('long_pauses'),
            },
            # 값이 없으면 null → 프론트 최종 리포트는 '준비 중' 칸으로 보인다
            'prolong_detail': report_detail.get('prolong_detail'),
            'energy_detail':  report_detail.get('energy_detail'),
            'tremor_detail':  report_detail.get('tremor_detail'),
            'cnn_window_count': window_count,
            'demo_mode': not cnn_models,
            # 받아쓴 단어와 시각(리포트 표시용). 대답 내용이라 응답으로만 돌려주고 로그·DB에는 남기지 않는다
            'words': [{'text': w['text'], 'start': round(w['start'], 2), 'end': round(w['end'], 2)}
                      for w in filler_result.get('words', [])],
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

@app.route('/analyze/stage1/latest', methods=['GET'])
def get_latest_stage1():
    """내 가장 최근 1단계 결과를 {'result': ... | None}으로 돌려준다."""
    user = get_current_user()
    if user is None:
        return jsonify({'error': '로그인이 필요합니다'}), 401

    row = (
        Stage1Result.query
        .filter_by(user_id=user.id)
        .order_by(Stage1Result.created_at.desc())
        .first()
    )
    return jsonify({'result': row.to_entry() if row else None})


@app.route('/analyze/gaze-blink', methods=['POST'])
def analyze_gaze_blink():
    """2단계 업로드 영상으로 깜빡임·시선·표정을 분석해 저장하고, 지표별 하이라이트 영상과 직전 기록을 준다."""
    # mediapipe·cv2는 여기서만 불러온다
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
            converted_path = analysis_path

        frames = extract_landmarks_from_video(analysis_path)
        duration_sec = frames[-1]["t"] if frames else 0

        # 앞 5초를 개인 기준값 구간으로 쓴다
        calib_frames = [f for f in frames if f["t"] <= 5.0]
        baseline_ear = calibrate_baseline_ear(calib_frames, LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX)
        baseline_yaw, baseline_pitch = calibrate_baseline_gaze(
            calib_frames, POSE_LANDMARK_IDX, LEFT_IRIS_IDX, RIGHT_IRIS_IDX,
            LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
        )

        # 깜빡임: blendshape 값이 있으면 그것을, 없으면 눈 비율(EAR)로 계산한다
        blink_series = compute_blink_blendshape_series(frames)
        if any(score is not None for _, score in blink_series):
            blinks = detect_blinks_from_blendshape(blink_series)
        else:
            ear_series = compute_ear_series(frames, LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX)
            blinks = detect_blinks(ear_series, baseline_ear)
        blink_result = score_blink_rate(len(blinks), duration_sec)

        gaze_segments = detect_gaze_segments(
            frames, POSE_LANDMARK_IDX, LEFT_IRIS_IDX, RIGHT_IRIS_IDX,
            LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
            baseline_yaw=baseline_yaw, baseline_pitch=baseline_pitch,
        )
        gaze_result = score_gaze_segments(gaze_segments)

        expression_series = compute_expression_series(frames)
        expression_summary = summarize_expression(expression_series)
        expression_result = score_expression(
            expression_summary["smile_ratio"], expression_summary["tension_ratio"]
        )
        expression_segments = detect_expression_segments(expression_series)

        # 저장 전에 직전 기록을 먼저 읽어야 방금 넣은 기록이 previous로 잡히지 않는다
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

        # 하이라이트: 시선은 피한 순간, 표정은 미소·긴장, 깜빡임은 전부
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


@app.route('/analyze/gaze-blink/live', methods=['POST'])
def save_live_gaze_blink():
    """2단계 실시간 통화 결과 저장. 영상 없이 브라우저가 계산한 숫자만 받아 업로드와 같은 표에 쌓는다."""
    user = get_current_user()
    if user is None:
        return jsonify({'error': '로그인이 필요합니다'}), 401

    try:
        v = parse_live_result(request.get_json(silent=True))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # 저장 전에 직전 기록을 먼저 읽는다
    previous_row = (
        Stage2Result.query
        .filter_by(user_id=user.id)
        .order_by(Stage2Result.created_at.desc())
        .first()
    )
    previous_entry = previous_row.to_entry() if previous_row else None

    db.session.add(Stage2Result(
        user_id=user.id,
        blink_rate_per_min=v['blinkRatePerMin'],
        blink_status=v['blinkStatus'],
        blink_score=v['blinkScore'],
        avg_fixation_sec=v['avgFixationSec'],
        gaze_score=v['gazeScore'],
        smile_ratio=v['smileRatio'],
        tension_ratio=v['tensionRatio'],
        expression_score=v['expressionScore'],
        expression_status=v['expressionStatus'],
    ))
    db.session.commit()
    return jsonify({'previous': previous_entry})


@app.route('/analyze/stage2/latest', methods=['GET'])
def get_latest_stage2():
    """내 가장 최근 2단계 결과(업로드·실시간 구분 없음)를 {'result': ... | None}으로 돌려준다."""
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


@app.route('/interview/anxiety', methods=['GET'])
def get_anxiety():
    """1·2단계 최신 결과로 불안도와 3단계 난이도를 낸다. 기록이 없으면 score None, 난이도 warmup."""
    user = get_current_user()
    if user is None:
        return jsonify({'error': '로그인이 필요합니다'}), 401

    s1 = Stage1Result.query.filter_by(user_id=user.id).order_by(Stage1Result.created_at.desc()).first()
    v2 = Stage2VoiceResult.query.filter_by(user_id=user.id).order_by(Stage2VoiceResult.created_at.desc()).first()
    s2 = Stage2Result.query.filter_by(user_id=user.id).order_by(Stage2Result.created_at.desc()).first()
    items = {}
    voice = [r for r in (s1, v2) if r]
    if voice:
        # 목소리 5개는 1·2단계 중 더 긴장한 쪽 값을 쓴다
        cols = dict(pause='pause_ctrl', filler='fluency', prolong='continuity', energy='calm', tremor='stability')
        items.update({k: anxiety_worse(*(getattr(r, c) for r in voice)) for k, c in cols.items()})
    if s2:
        items.update(gaze=anxiety_gaze_item(s2.avg_fixation_sec), blink=anxiety_blink_item(s2.blink_rate_per_min),
                     tension=100 * (1 - s2.tension_ratio))
    score = anxiety_score(items)
    return jsonify({'score': score, 'tier': anxiety_tier(score), 'items': items})

@app.route('/community/posts', methods=['GET'])
def list_posts():
    """게시판 글 목록(최신순). 로그인 없이 볼 수 있다."""
    current_user = get_current_user()
    posts = Post.query.order_by(Post.created_at.desc()).all()
    result = []
    for p in posts:
        count = Comment.query.filter_by(post_id=p.id).count()
        result.append(p.to_dict(comment_count=count, current_user=current_user))
    return jsonify({'posts': result})


@app.route('/community/posts', methods=['POST'])
def create_post():
    """게시판 글쓰기. 로그인이 필요하다."""
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
    """게시판 글 하나와 댓글 목록."""
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
    """게시판 글 삭제(댓글 포함). 본인 글만."""
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
    """댓글 쓰기. 로그인이 필요하다."""
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
    """1단계 코칭 한 문단. /analyze 결과를 받아 AI로 쓰고, 못 쓰면 source가 template이다."""
    result = request.get_json(silent=True)
    if not result or 'scores' not in result:
        return jsonify({'error': 'scores가 필요합니다'}), 400
    text = generate_feedback(result)
    return jsonify({'feedback': text, 'source': 'llm' if text else 'template'})


def _load_score_profile(user):
    """최신 1·2단계 세부 점수(질문 방식 조절용). 없거나 조회가 실패하면 None — 질문 생성은 그래도 된다."""
    if user is None:
        return None
    try:
        s1 = (Stage1Result.query.filter_by(user_id=user.id)
              .order_by(Stage1Result.created_at.desc()).first())
        s2 = (Stage2Result.query.filter_by(user_id=user.id)
              .order_by(Stage2Result.created_at.desc()).first())
        return {
            'stage1': None if s1 is None else {
                'stability': s1.stability, 'fluency': s1.fluency, 'pause_ctrl': s1.pause_ctrl,
                'continuity': s1.continuity, 'calm': s1.calm,
            },
            'stage2': None if s2 is None else {
                'blink': s2.blink_score, 'gaze': s2.gaze_score, 'expression': s2.expression_score,
            },
        }
    except Exception as e:
        print(f"[WARNING] 점수 profile 조회 실패, 없이 진행: {e}")
        return None


def _load_past_questions(user):
    """지난 면접의 기본 질문 문장(최근 순) — 이번 질문이 겹치지 않게. 없거나 실패하면 빈 목록."""
    if user is None:
        return []
    try:
        rows = (
            db.session.query(InterviewTurn.question)
            .join(InterviewSession, InterviewTurn.session_id == InterviewSession.id)
            .filter(InterviewSession.user_id == user.id, InterviewTurn.kind == 'main')
            .order_by(InterviewTurn.created_at.desc(), InterviewTurn.id.desc())
            .limit(60)  # 중복·자기소개를 걷어내고도 MAX_PAST_QUESTIONS개가 남도록 여유 있게 읽음
            .all()
        )
        return [r[0] for r in rows]
    except Exception as e:
        print(f"[WARNING] 지난 면접 질문 조회 실패, 없이 진행: {e}")
        return []


@app.route('/interview/next-question', methods=['POST'])
def interview_next_question():
    """3단계 다음 질문(기본 또는 꼬리). 로그인했으면 1·2단계 점수와 지난 질문도 반영한다.
    body: {tier, mode?, previous_questions, previous_answer?, about_me?}"""
    data = request.get_json(silent=True) or {}
    tier = data.get('tier', 'standard')
    mode = data.get('mode', 'main')
    previous_questions = data.get('previous_questions', [])
    previous_answer = data.get('previous_answer')

    if tier not in ('warmup', 'standard', 'practice'):
        return jsonify({'error': "tier는 'warmup' | 'standard' | 'practice' 중 하나여야 합니다"}), 400
    if mode not in ('main', 'follow_up'):
        return jsonify({'error': "mode는 'main' | 'follow_up' 중 하나여야 합니다"}), 400

    user = get_current_user()
    profile = _load_score_profile(user)
    past_questions = _load_past_questions(user) if mode == 'main' else []
    about_me = clean_about_me(data.get('about_me'))
    question, source = generate_question(
        tier, previous_questions, previous_answer, mode, profile=profile, past_questions=past_questions,
        about_me=about_me,
    )
    return jsonify({'question': question, 'source': source})


def _own_interview_session(user, session_id):
    """본인 면접만 돌려준다. 남의 것도 None(404)이라 존재 여부가 드러나지 않는다."""
    return InterviewSession.query.filter_by(id=session_id, user_id=user.id).first()


@app.route('/interview/sessions', methods=['POST'])
def start_interview_session():
    """면접 시작 — 세션을 만들고 id를 돌려준다. body: {tier}"""
    user = get_current_user()
    if user is None:
        return jsonify({'error': '로그인이 필요합니다'}), 401
    try:
        tier = parse_session_start(request.get_json(silent=True))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    session = InterviewSession(user_id=user.id, tier=tier)
    db.session.add(session)
    db.session.commit()
    return jsonify({'id': session.id}), 201


@app.route('/interview/sessions', methods=['GET'])
def list_interview_sessions():
    """내 면접 요약 목록(최신순 20개). 본문은 /interview/sessions/<id>에서."""
    user = get_current_user()
    if user is None:
        return jsonify({'error': '로그인이 필요합니다'}), 401
    rows = (
        InterviewSession.query.filter_by(user_id=user.id)
        .order_by(InterviewSession.started_at.desc(), InterviewSession.id.desc())
        .limit(20).all()
    )
    return jsonify({'sessions': [r.to_summary() for r in rows]})


@app.route('/interview/sessions/<int:session_id>', methods=['GET'])
def get_interview_session(session_id):
    """면접 1회의 상세 — 질문과 답변 텍스트 전체. 본인 것만."""
    user = get_current_user()
    if user is None:
        return jsonify({'error': '로그인이 필요합니다'}), 401
    session = _own_interview_session(user, session_id)
    if session is None:
        return jsonify({'error': '면접 기록을 찾을 수 없습니다'}), 404
    return jsonify(session.to_detail())


@app.route('/interview/sessions/<int:session_id>', methods=['DELETE'])
def delete_interview_session(session_id):
    """면접 기록 삭제 — 질문·답변 텍스트가 함께 지워진다. 본인 것만."""
    user = get_current_user()
    if user is None:
        return jsonify({'error': '로그인이 필요합니다'}), 401
    session = _own_interview_session(user, session_id)
    if session is None:
        return jsonify({'error': '면접 기록을 찾을 수 없습니다'}), 404
    db.session.delete(session)
    db.session.commit()
    return jsonify({'message': '삭제했습니다'})


@app.route('/interview/sessions/<int:session_id>/turns', methods=['POST'])
def add_interview_turn(session_id):
    """질문이 화면에 나올 때 저장하고 turn id를 돌려준다. body: {kind, question}"""
    user = get_current_user()
    if user is None:
        return jsonify({'error': '로그인이 필요합니다'}), 401
    session = _own_interview_session(user, session_id)
    if session is None:
        return jsonify({'error': '면접 기록을 찾을 수 없습니다'}), 404
    try:
        kind, question = parse_turn(request.get_json(silent=True))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    order = len(session.turns)
    if order >= MAX_TURNS_PER_SESSION:
        return jsonify({'error': f'한 면접에 저장할 수 있는 질문은 {MAX_TURNS_PER_SESSION}개까지입니다'}), 400

    turn = InterviewTurn(session_id=session.id, order_index=order, kind=kind, question=question)
    db.session.add(turn)
    db.session.commit()
    return jsonify({'id': turn.id, 'order': order}), 201


@app.route('/interview/sessions/<int:session_id>/turns/<int:turn_id>/answer', methods=['PUT'])
def save_interview_answer(session_id, turn_id):
    """확정된 대답 글을 저장한다(다시 부르면 덮어씀). body: {answer}"""
    user = get_current_user()
    if user is None:
        return jsonify({'error': '로그인이 필요합니다'}), 401
    session = _own_interview_session(user, session_id)
    if session is None:
        return jsonify({'error': '면접 기록을 찾을 수 없습니다'}), 404
    turn = InterviewTurn.query.filter_by(id=turn_id, session_id=session.id).first()
    if turn is None:
        return jsonify({'error': '질문을 찾을 수 없습니다'}), 404
    try:
        answer = parse_answer(request.get_json(silent=True))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    turn.answer = answer
    turn.answered_at = datetime.datetime.utcnow()
    db.session.commit()
    return jsonify({'message': '저장했습니다'})


@app.route('/interview/sessions/<int:session_id>/complete', methods=['POST'])
def complete_interview_session(session_id):
    """면접 완료 시각을 남긴다. 여러 번 불러도 처음 시각을 유지한다."""
    user = get_current_user()
    if user is None:
        return jsonify({'error': '로그인이 필요합니다'}), 401
    session = _own_interview_session(user, session_id)
    if session is None:
        return jsonify({'error': '면접 기록을 찾을 수 없습니다'}), 404
    if session.completed_at is None:
        session.completed_at = datetime.datetime.utcnow()
        db.session.commit()
    return jsonify(session.to_summary())


def _build_coach_facts(user, session):
    """코치 에이전트가 볼 facts를 이 사용자 것만으로 만든다 — 도구가 이것만 보므로 남의 기록은 못 본다."""
    turns = [{'kind': t.kind, 'question': t.question, 'answer': t.answer or ''} for t in session.turns]
    previous = (
        InterviewSession.query
        .filter(InterviewSession.user_id == user.id, InterviewSession.id != session.id,
                InterviewSession.started_at <= session.started_at)
        .order_by(InterviewSession.started_at.desc(), InterviewSession.id.desc())
        .limit(5).all()
    )
    return {
        'session': {'id': session.id, 'tier': session.tier, 'completed': session.completed_at is not None, 'turns': turns},
        'previous_sessions': [
            {
                'id': p.id, 'tier': p.tier, 'completed': p.completed_at is not None,
                'question_count': len(p.turns),
                'answered_count': sum(1 for t in p.turns if (t.answer or '').strip()),
            }
            for p in previous
        ],
        'stage_scores': _load_score_profile(user) or {'stage1': None, 'stage2': None},
    }


@app.route('/interview/sessions/<int:session_id>/coach-note', methods=['POST'])
def get_or_create_coach_note(session_id):
    """면접 직후 코치 노트. 저장된 게 있으면 그대로, 없으면 만들어 저장한다. 응답: {note, source}"""
    user = get_current_user()
    if user is None:
        return jsonify({'error': '로그인이 필요합니다'}), 401
    session = _own_interview_session(user, session_id)
    if session is None:
        return jsonify({'error': '면접 기록을 찾을 수 없습니다'}), 404

    existing = session.coach_note
    if existing is not None:
        return jsonify({'note': json.loads(existing.content), 'source': existing.source})

    note, source = generate_coach_note(_build_coach_facts(user, session))
    row = CoachNote(
        session_id=session.id,
        content=json.dumps(note, ensure_ascii=False),
        source=source,
        model=COACH_MODEL if source == 'llm' else None,
    )
    try:
        db.session.add(row)
        db.session.commit()
    except IntegrityError:
        # 같은 면접의 결과 화면이 동시에 두 번 열려 먼저 저장된 게 있으면 그걸 쓴다
        db.session.rollback()
        row = CoachNote.query.filter_by(session_id=session.id).first()
        if row is None:
            raise
        return jsonify({'note': json.loads(row.content), 'source': row.source})
    return jsonify({'note': note, 'source': source})


@app.route('/interview/sessions/<int:session_id>/report-coach', methods=['POST'])
def get_or_create_report_coach(session_id):
    """리포트 3·4번 칸. 저장된 게 있으면 그대로, 없으면 만들어 저장한다.
    body: {turns: [{question, fillers, pauses}], candidates: [{key, line}]} · 응답: {result, source}"""
    user = get_current_user()
    if user is None:
        return jsonify({'error': '로그인이 필요합니다'}), 401
    session = _own_interview_session(user, session_id)
    if session is None:
        return jsonify({'error': '면접 기록을 찾을 수 없습니다'}), 404

    existing = session.report_coach
    if existing is not None:
        return jsonify({'result': json.loads(existing.content), 'source': existing.source})
    try:
        stats, cands = parse_report_request(request.get_json(silent=True))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    turns = [{'question': t.question, 'answer': t.answer or ''} for t in session.turns]
    result, source = generate_report_coach(turns, stats, cands)
    row = ReportCoach(session_id=session.id, content=json.dumps(result, ensure_ascii=False),
                      source=source, model=COACH_MODEL if source == 'llm' else None)
    try:
        db.session.add(row)
        db.session.commit()
    except IntegrityError:
        # 리포트가 동시에 두 번 열려 먼저 저장된 게 있으면 그걸 쓴다
        db.session.rollback()
        row = ReportCoach.query.filter_by(session_id=session.id).first()
        if row is None:
            raise
        return jsonify({'result': json.loads(row.content), 'source': row.source})
    return jsonify({'result': result, 'source': source})


# 요청마다 AI 비용이 들어 사용자당 시간당 횟수를 막는다
_practice_limiter = RateLimiter()


@app.route('/interview/practice-hint', methods=['POST'])
def get_practice_hint():
    """맞춤 연습 힌트(시작 문장 틀·말하는 순서). AI를 못 쓰면 일반 힌트.
    body: {question, previous_answer?} · 응답: {hint: {opening, steps}, source}"""
    user = get_current_user()
    if user is None:
        return jsonify({'error': '로그인이 필요합니다'}), 401
    try:
        question, previous_answer = parse_hint_request(request.get_json(silent=True))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    if not _practice_limiter.allow(user.id):
        return jsonify({'error': '연습 힌트를 너무 많이 요청했어요. 잠시 뒤에 다시 시도해주세요.'}), 429

    hint, source = generate_hint(question, previous_answer)
    return jsonify({'hint': hint, 'source': source})


# 하루 대화는 대답마다 AI를 불러(1·2단계 합쳐 8번 안팎) 넉넉하게 잡되 연타는 막는다
_haru_limiter = RateLimiter(limit=120)


@app.route('/haru/reply', methods=['POST'])
def haru_reply():
    """하루 대화: 방금 대답에 한 문장 반응하고 다음 질문으로 잇는다. AI를 못 쓰면 정해 둔 공감 문장.
    body: {question, answer, next, retried, index, history} · 응답: {say, retry, source}"""
    user = get_current_user()
    if user is None:
        return jsonify({'error': '로그인이 필요합니다'}), 401
    try:
        req = parse_reply_request(request.get_json(silent=True))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    if not _haru_limiter.allow(user.id):
        return jsonify({'error': '잠시 뒤에 다시 이야기해요.'}), 429
    out, source = generate_reply(req)
    return jsonify({**out, 'source': source})


@app.route('/haru/profile', methods=['POST'])
def haru_profile():
    """'하루가 이렇게 이해했어요' 카드: 대답을 항목별로 짧게 정리한다(저장하지 않음).
    body: {items: [{label, question, answer}]} · 응답: {items: [{label, value}], source}"""
    user = get_current_user()
    if user is None:
        return jsonify({'error': '로그인이 필요합니다'}), 401
    try:
        items = parse_profile_request(request.get_json(silent=True))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    if not _haru_limiter.allow(user.id):
        return jsonify({'error': '잠시 뒤에 다시 시도해주세요.'}), 429
    out, source = summarize_profile(items)
    return jsonify({'items': out, 'source': source})


@app.route('/interview/tts', methods=['POST'])
def interview_tts():
    """문장을 mp3로 읽어 준다. tier로 면접관 목소리를, "haru"면 하루 목소리를 고른다. body: {text, tier?}"""
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()
    tier = data.get('tier')
    if not text:
        return jsonify({'error': 'text가 필요합니다'}), 400

    audio_bytes = synthesize_speech(text, tier=tier)
    if audio_bytes is None:
        return jsonify({'error': 'TTS 생성 실패 (키 없음 또는 호출 오류)'}), 502

    return Response(audio_bytes, mimetype='audio/mpeg')


@app.route('/interview/transcribe', methods=['POST'])
def interview_transcribe():
    """3단계 대답 영상을 Whisper로 받아쓴다(꼬리질문 재료). Whisper가 없으면 빈 글로 답한다."""
    load_models()

    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다'}), 400

    if not (WHISPER_AVAILABLE and whisper_model is not None):
        # 에러 대신 빈 글로 답해 꼬리질문 없이 진행되게 한다
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
        # Whisper는 한 번에 한 요청만 쓴다(겹치면 단어 시각이 깨짐 — step1/whisper_guard.py)
        from step1.whisper_guard import transcribe as guarded_transcribe
        result = guarded_transcribe(whisper_model, y16, language="ko")
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
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True)
