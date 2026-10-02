import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_util import EFFORT_LEVELS, effort_options


@pytest.mark.parametrize("level", EFFORT_LEVELS)
def test_effort_is_passed_through_extra_body_not_as_a_keyword_argument(level):
    # output_config=... 를 직접 넘기면 고정된 anthropic==0.69.0에서 TypeError가 난다(→ 폴백으로 삼켜져 AI가 안 쓰임).
    # extra_body는 SDK 버전과 무관하게 요청 본문에 합쳐진다.
    opts = effort_options(level)
    assert set(opts) == {"extra_body"}
    assert opts["extra_body"] == {"output_config": {"effort": level}}


def test_unknown_effort_level_is_rejected():
    with pytest.raises(ValueError):
        effort_options("ultra")


def test_kwargs_are_accepted_by_the_installed_anthropic_sdk():
    # 실제 SDK 시그니처로 확인 — 고정 버전(0.69.0)이든 최신이든 TypeError가 나면 안 된다.
    # CI(순수 로직 테스트)엔 anthropic이 없어서 건너뛰고, 설치된 환경(배포·로컬)에서만 검사한다.
    anthropic = pytest.importorskip("anthropic")
    import inspect

    sig = inspect.signature(anthropic.Anthropic(api_key="x").messages.create)
    sig.bind(model="m", max_tokens=1, messages=[], **effort_options("low"))  # 맞지 않으면 TypeError
