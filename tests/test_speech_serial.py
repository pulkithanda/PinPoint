from backend.guidance.engine import Say
from backend.guidance.speech import SpeechScheduler, tone_wav
from backend.hardware.serial_monitor import FakeSerialMonitor, diagnose, parse_line


def test_rate_limit_and_latest_wins():
    s = SpeechScheduler(min_interval_s=0.8, repeat_after_s=4.0)
    s.submit(Say("Left two ports.", "steer", "h:left:2"), 0.0)
    assert s.pop_ready(0.0).text == "Left two ports."
    s.submit(Say("Left one port.", "steer", "h:left:1"), 0.2)
    s.submit(Say("Down one.", "steer", "v:down:1"), 0.3)
    assert s.pop_ready(0.5) is None              # too soon
    assert s.pop_ready(0.9).text == "Down one."  # latest wins
    s.submit(Say("Down one.", "steer", "v:down:1"), 1.0)
    assert s.pop_ready(2.0) is None              # unchanged instruction, said recently


def test_info_never_dropped_and_urgent_first():
    s = SpeechScheduler()
    s.submit(Say("a"), 0)
    s.submit(Say("b"), 0)
    s.submit(Say("STOP", "urgent", "at:D7"), 0)
    assert [s.pop_ready(t).text for t in (0, 0, 0)] == ["STOP", "a", "b"]
    assert s.pop_ready(0, busy=True) is None


def test_parse_line():
    assert parse_line("dist=34 lcd=1\r\n") == {"dist": 34, "lcd": 1}
    assert parse_line("garbage") == {}


def test_serial_status():
    m = FakeSerialMonitor()
    st = m.status(0.0)
    assert not st.connected and diagnose(st).startswith("I can't see the Arduino")
    m.fake_connected = True
    for i, d in enumerate([100, 80, 50, 30, 20]):
        m.feed(f"dist={d} lcd=0", now=1 + i * 0.2)
    st = m.status(2.0)
    assert st.receiving and st.dist_varies and st.lcd_ok is False and not st.works
    assert "screen" in diagnose(st)


def test_tone_wav():
    assert tone_wav(800)[:4] == b"RIFF"
