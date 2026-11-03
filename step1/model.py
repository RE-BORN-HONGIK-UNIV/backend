"""라벨별 이진 판별 CNN 설계도(size small). 입력은 1채널 로그-멜 배열(배치 × 1 × 멜 × 프레임)이고 출력은 '있음' 로짓 하나다.
backend는 이 파일을 그대로 복사해 state_dict를 연다. 바꾸면 이미 저장한 모델이 열리지 않는다.
"""
from torch import nn


def block(c_in, c_out):
    """3×3 합성곱 → 배치정규화 → ReLU → 2×2 최대풀링 한 묶음."""
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2))


class SmallCNN(nn.Module):
    """합성곱 4묶음 뒤 전체 평균풀링으로 길이와 상관없이 하나의 점수를 낸다."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 64))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        """로그-멜 배치를 받아 '있음' 로짓(시그모이드 전 값)을 돌려준다."""
        return self.head(self.features(x).mean(dim=(2, 3))).squeeze(1)
