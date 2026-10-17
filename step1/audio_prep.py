"""음성 파일을 모델 입력(1채널 로그-멜 배열)으로 손질한다: 샘플레이트 맞추기 → 음량 맞추기(클립 또는 조각 단위) → 3초 조각 → 로그-멜 → 정규화.
학습과 판별의 손질이 똑같아야 하므로 backend는 이 파일을 그대로 복사해 쓴다. 숫자는 모두 설정 JSON(cfg)에서 읽는다.
"""
import librosa
import numpy as np
import pyloudnorm


def match_loudness(y, cfg):
    """소리 전체 음량을 cfg의 LUFS로 맞춘다. 음량을 잴 수 없으면(거의 무음이거나 0.4초 미만) None."""
    if len(y) < 0.4 * cfg["sr"]:  # pyloudnorm은 0.4초 블록 하나는 있어야 잰다
        return None
    loudness = pyloudnorm.Meter(cfg["sr"]).integrated_loudness(y)
    if not np.isfinite(loudness):
        return None
    return pyloudnorm.normalize.loudness(y, loudness, cfg["lufs"]).astype(np.float32)


def load(path, cfg):
    """파일을 읽어 cfg의 샘플레이트로 바꾼다. lufs_scope가 window가 아니면 클립 전체 음량도 맞춘다."""
    y, _ = librosa.load(path, sr=cfg["sr"], mono=True)
    if cfg.get("lufs_scope") != "window":
        matched = match_loudness(y, cfg)
        if matched is not None:  # 완전 무음이면 맞출 음량이 없다
            y = matched
    return y.astype(np.float32)


def starts(y, cfg, hop_sec):
    """조각 시작초 목록. 마지막 조각은 클립 끝에 맞추고, 조각보다 짧은 클립은 0초 하나다."""
    dur, win = len(y) / cfg["sr"], cfg["win_sec"]
    if dur <= win:
        return [0.0]
    out = list(np.arange(0, dur - win, hop_sec))
    if not np.isclose(out[-1], dur - win):
        out.append(dur - win)
    return [round(float(s), 3) for s in out]


def logmel(y, start, cfg):
    """start초부터 조각 하나를 잘라(짧으면 뒤를 무음으로 채워) 바닥값을 둔 로그-멜 배열(멜 × 프레임)로 바꾼다.
    lufs_scope가 window면 조각마다 음량을 맞춘 뒤 무음을 채우고, 음량을 잴 수 없는 조각은 None이다."""
    n = int(cfg["win_sec"] * cfg["sr"])
    i = int(round(start * cfg["sr"]))
    chunk = y[i:i + n]
    if cfg.get("lufs_scope") == "window":
        chunk = match_loudness(chunk, cfg)
        if chunk is None:
            return None
    chunk = np.pad(chunk, (0, max(0, n - len(chunk))))
    mel = librosa.feature.melspectrogram(
        y=chunk, sr=cfg["sr"], n_fft=cfg["n_fft"], win_length=cfg["win_length"], hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"], fmin=cfg["fmin"], fmax=cfg["fmax"], power=2.0)
    db = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
    return np.maximum(db, cfg["db_floor"]).astype(np.float32)


def normalize(db, cfg):
    """공부용 전체에서 구한 평균·표준편차로 정규화한다."""
    return (db - cfg["mean"]) / cfg["std"]
