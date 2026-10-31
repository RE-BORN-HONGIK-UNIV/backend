import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step1.report_detail import MAX_EVENTS, filler_events, long_pause_events


def test_filler_events_keep_only_scored_types_in_time_order():
    events = [
        {"text": "그", "start": 5.0, "end": 5.2, "type": "conditional"},
        {"text": "(비언어적 채움말 후보)", "start": 1.0, "end": 1.3, "type": "acoustic_candidate"},
        {"text": " 음 ", "start": 0.4, "end": 0.7, "type": "high"},
    ]
    assert filler_events(events) == [
        {"text": "음", "start": 0.4, "end": 0.7},
        {"text": "그", "start": 5.0, "end": 5.2},
    ]


def test_filler_events_skip_broken_rows():
    assert filler_events([None, {"type": "high", "text": "", "start": 1, "end": 2}, {"type": "high", "text": "어", "start": "x", "end": 1}]) == []
    assert filler_events(None) == []


def test_long_pause_events_drop_short_pauses():
    events = [
        {"start": 3.0, "end": 3.4, "duration": 0.4, "type": "short_pause"},
        {"start": 10.0, "end": 13.5, "duration": 3.5, "type": "long_pause"},
    ]
    assert long_pause_events(events) == [{"start": 10.0, "end": 13.5, "duration": 3.5}]


def test_event_lists_are_capped():
    many = [{"text": "음", "start": i, "end": i + 0.2, "type": "high"} for i in range(MAX_EVENTS + 50)]
    assert len(filler_events(many)) == MAX_EVENTS
