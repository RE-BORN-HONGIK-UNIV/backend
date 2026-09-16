import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step2.smile_cascade import detect_smile_in_frame


def test_returns_none_when_no_face_in_frame():
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    assert detect_smile_in_frame(blank) is None
