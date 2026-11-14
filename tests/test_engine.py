"""Step through the proximity alert build with fake fingertip input."""
import pytest

from backend.config import Settings
from backend.guidance.engine import (Finished, GuidanceEngine, Observation, Phase, Project,
                                     RequestIdentify, RequestPlugCheck, Say, parse_command)
from backend.hardware.serial_monitor import FakeSerialMonitor
from backend.vision.shield import ShieldLayout
from backend.vision.verify import ModuleID, PlugCheck


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


@pytest.fixture
def env():
    s = Settings()
    layout = ShieldLayout.load(s.layout_path)
    project = Project.load(s.projects_dir / "proximity_alert.yaml")
    clock = Clock()
    eng = GuidanceEngine(project, layout, mode="voice", clock=clock)
    return eng, layout, clock


def obs(eng, clock, fp, dt=0.1):
    clock.t += dt
    return eng.on_observation(Observation(clock.t, True, fp))


def says(actions):
    return [a.text for a in actions if isinstance(a, Say)]


def at(layout, port):
    p = layout.ports[port]
    return (p.x, p.y)


def test_parse_command():
    assert parse_command("What am I holding?") == "holding"
    assert parse_command("next") == "next"
    assert parse_command("Voice mode") == "voice"
    assert parse_command("okay I'm done") == "done"
    assert parse_command("repeat that") == "repeat"
    assert parse_command("go back") == "back"
    assert parse_command("banana") is None


def test_mode_select_by_thumbs_up():
    s = Settings()
    eng = GuidanceEngine(Project.load(s.projects_dir / "proximity_alert.yaml"), ShieldLayout.load(s.layout_path))
    assert eng.start() and eng.phase == Phase.MODE_SELECT
    eng.on_observation(Observation(1.0, True, None, gesture="Thumb_Up"))
    assert eng.mode == "visual" and eng.phase == Phase.FIND_MODULE


def test_steering_words(env):
    eng, layout, clock = env
    eng.start()
    eng.on_identify_result(ModuleID("Ultrasonic Distance Sensor", True, ""))
    assert eng.phase == Phase.GUIDE
    # target D7 is row 1 col 1; D5 is row 0 col 3 -> larger axis first
    assert says(obs(eng, clock, at(layout, "D5"))) == ["Left two ports."]
    # D3 (row 0, col 1) -> straight down one row
    assert says(obs(eng, clock, at(layout, "D3"))) == ["Down one."]
    # still hand -> quiet
    assert says(obs(eng, clock, at(layout, "D3"))) == []
    assert says(obs(eng, clock, at(layout, "D7")))[0].startswith("Stop. That's D7")
    # way off the board
    assert says(obs(eng, clock, (-60.0, 20.0))) == ["Move right, onto the board."]


def test_full_build_with_planted_mistake(env):
    eng, layout, clock = env
    eng.start()
    assert eng.phase == Phase.FIND_MODULE

    # Step 1: wrong module first, then the right one
    acts = obs(eng, clock, (200.0, 200.0))
    assert any(isinstance(a, RequestIdentify) for a in acts)
    assert "That's the Buzzer" in says(eng.on_identify_result(ModuleID("Buzzer", True, "")))[0]
    eng.on_identify_result(ModuleID("Ultrasonic Distance Sensor", True, ""))
    assert eng.phase == Phase.GUIDE

    # Steer to D7, then the hand leaves for 2 s -> automatic plug check
    obs(eng, clock, at(layout, "D7"))
    acts = []
    for _ in range(25):
        acts += obs(eng, clock, None)
    checks = [a for a in acts if isinstance(a, RequestPlugCheck)]
    assert checks and checks[0].ports == ["D7"]
    eng.on_plug_result(PlugCheck(True, "D7", ""))
    assert eng.occupied == {"D7": "Ultrasonic Distance Sensor"} and eng.step.id == "buzzer"

    # Step 2: planted mistake - buzzer goes into D8 instead of D6
    eng.on_identify_result(ModuleID("Buzzer", True, ""))
    eng.on_command("done")
    assert eng.phase == Phase.CHECKING
    msg = says(eng.on_plug_result(PlugCheck(True, "D8", "")))[0]
    assert "D8, not D6" in msg and eng.phase == Phase.GUIDE
    eng.on_command("done")
    eng.on_plug_result(PlugCheck(True, "D6", ""))
    assert eng.step.id == "lcd"

    # Step 3: LCD into any I2C port
    eng.on_identify_result(ModuleID("LCD RGB Backlight", True, ""))
    assert set(eng.free_targets()) == {"I2C_1", "I2C_2", "I2C_3", "I2C_4"}
    eng.on_command("done")
    eng.on_plug_result(PlugCheck(True, "I2C_3", ""))
    assert eng.phase == Phase.ELECTRICAL

    # Electrical check: readings arrive and change
    ser = FakeSerialMonitor()
    ser.fake_connected = True
    out = []
    for d in (120, 90, 60, 35, 30, 80):
        clock.t += 0.2
        ser.feed(f"dist={d} lcd=1", now=clock.t)
        out += eng.on_serial_status(ser.status(clock.t))
    fin = [a for a in out if isinstance(a, Finished)]
    assert fin and eng.phase == Phase.DONE
    assert "It works" in " ".join(says(out))
    recap = fin[0].recap
    assert "D8 instead of D6" in recap and "I2C_3" in recap


def test_dead_sensor_is_diagnosed(env):
    eng, layout, clock = env
    eng.start()
    eng.step_idx = len(eng.project.steps) - 1
    eng.on_identify_result(ModuleID("LCD RGB Backlight", True, ""))
    eng.on_command("done")
    eng.on_plug_result(PlugCheck(True, "I2C_1", ""))
    assert eng.phase == Phase.ELECTRICAL
    ser = FakeSerialMonitor()
    out = []
    for _ in range(30):
        clock.t += 0.2
        ser.feed("dist=0 lcd=1", now=clock.t)
        out += eng.on_serial_status(ser.status(clock.t))
    assert any("Check the distance sensor is in D7" in t for t in says(out))
    assert eng.phase == Phase.ELECTRICAL


def test_no_verifier_flow():
    s = Settings()
    clock = Clock()
    eng = GuidanceEngine(Project.load(s.projects_dir / "proximity_alert.yaml"), ShieldLayout.load(s.layout_path),
                         mode="voice", verifier_available=False, clock=clock)
    assert "Say 'next'" in says(eng.start())[-1]
    eng.on_command("next")
    assert eng.phase == Phase.GUIDE
    eng.on_command("done")
    eng.on_plug_result(PlugCheck(None, None, "no verifier"))
    assert eng.occupied == {"D7": "Ultrasonic Distance Sensor"}
