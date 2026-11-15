"""
Breadboard + Arduino model: endpoints, which holes are joined, and how to find each one by touch.

Endpoint names (what plans use):
  holes      "E10"            column letter A-J + row 1-63 (830-point board)
  rails      "RAIL:N+" "RAIL:N-"  the plus / minus strips along the edge nearest the builder
             "RAIL:F+" "RAIL:F-"  the strips along the far edge (plans avoid them)
  Arduino    "UNO:D13" "UNO:A0" "UNO:5V" "UNO:3V3" "UNO:GND" "UNO:VIN"

Holes A-E in one row are joined, F-J in one row are joined, the groove separates them.
Each rail strip is one long net. All Arduino GND sockets are one net.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ROWS = 63
MAX_TOUCH_ROW = 30      # plans stay in rows 1-30: counting further by touch is slow and error-prone
KIT_PATH = Path(__file__).with_name("elegoo_most_complete.yaml")

_HOLE = re.compile(r"^([A-J])(\d{1,2})$")
_RAILS = {"RAIL:N+", "RAIL:N-", "RAIL:F+", "RAIL:F-"}
POWER_PINS = {"UNO:5V", "UNO:3V3", "UNO:VIN"}
GROUND = "UNO:GND"
_ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]


def ordinal(n: int) -> str:
    return _ORD[n - 1] if 1 <= n <= len(_ORD) else f"number {n}"


# ------------------------------------------------------------------ kit catalog
@dataclass(frozen=True)
class PartType:
    id: str
    name: str
    count: int
    touch: str
    layout: str
    leads: dict[str, str]
    polarity: bool = False
    resistor: int | None = None
    library: str | None = None


@dataclass(frozen=True)
class Kit:
    name: str
    parts: dict[str, PartType]
    arduino_far: tuple[tuple[str, ...], ...]
    arduino_near: tuple[tuple[str, ...], ...]
    pwm: frozenset[str]
    libraries: tuple[str, ...]

    @property
    def arduino_pins(self) -> set[str]:
        names = {p for block in self.arduino_far + self.arduino_near for p in block}
        return {f"UNO:{p}" for p in names - {"NC", "IOREF", "RESET", "AREF", "SCL", "SDA"}}

    def catalog_text(self) -> str:
        """The parts list as the planner sees it."""
        lines = []
        for p in self.parts.values():
            lead_names = ", ".join(p.leads)
            extra = f"; needs library '{p.library}'" if p.library else ""
            lines.append(f"- {p.id}: {p.name} (have {p.count}); layout={p.layout}; leads: {lead_names}{extra}")
        return "\n".join(lines)


@lru_cache(maxsize=4)
def load_kit(path: str | Path = KIT_PATH) -> Kit:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    parts = {
        pid: PartType(pid, p["name"], int(p.get("count", 1)), p["touch"], p.get("layout", "free"),
                      dict(p["leads"]), bool(p.get("polarity", False)), p.get("resistor"), p.get("library"))
        for pid, p in d["parts"].items()
    }
    a = d["arduino"]
    return Kit(d.get("kit", "kit"), parts, tuple(tuple(b) for b in a["far_blocks"]),
               tuple(tuple(b) for b in a["near_blocks"]), frozenset(f"UNO:{p}" for p in a.get("pwm", [])),
               tuple(d.get("libraries", [])))


# ------------------------------------------------------------------ endpoints
def parse_hole(name: str) -> tuple[str, int] | None:
    m = _HOLE.match(name.strip().upper())
    if not m:
        return None
    row = int(m.group(2))
    return (m.group(1), row) if 1 <= row <= ROWS else None


def normalize(name: str) -> str:
    n = name.strip().upper().replace(" ", "")
    if n.startswith("UNO:") or n.startswith("RAIL:"):
        return n
    return n


def is_valid_endpoint(name: str, kit: Kit) -> bool:
    n = normalize(name)
    return parse_hole(n) is not None or n in _RAILS or n in kit.arduino_pins


def strip_of(endpoint: str) -> str:
    """The net a single endpoint belongs to before any wires: holes join per half-row."""
    n = normalize(endpoint)
    h = parse_hole(n)
    if h:
        col, row = h
        return f"ROW{row}{'L' if col <= 'E' else 'R'}"
    return n


def side_of(col: str) -> str:
    return "AE" if col <= "E" else "FJ"


# ------------------------------------------------------------------ orientation + touch
@dataclass
class Orientation:
    """How the board sits in front of the builder (confirmed by the setup photo when possible)."""
    near_half: str = "AE"          # "AE": holes A-E are on the builder's side of the groove
    row1: str = "left"             # which end row 1 is at
    near_outer_rail: str = "-"     # the rail strip closest to the near edge: "+" or "-"

    def hole_text(self, name: str) -> str:
        col, row = parse_hole(name)
        near = side_of(col) == self.near_half
        # E and F touch the groove; A and J are the outermost.
        from_groove = ("EDCBA".index(col) + 1) if col <= "E" else ("FGHIJ".index(col) + 1)
        side = "your side" if near else "the far side"
        if from_groove == 1:
            pos = f"the hole right beside the groove, on {side}"
        elif from_groove == 5:
            pos = f"the outermost hole on {side}"
        else:
            pos = f"the {ordinal(from_groove)} hole from the groove, on {side}"
        return f"row {row}, counting {row} from the {self.row1} end, {pos}"

    def rail_text(self, rail: str) -> str:
        sign = rail[-1]
        edge = "nearest you" if rail.startswith("RAIL:N") else "furthest from you"
        if rail.startswith("RAIL:N"):
            outer = sign == self.near_outer_rail
        else:
            outer = sign != self.near_outer_rail
        which = "the outer strip, right at the edge" if outer else "the inner strip, just inside the outer one"
        name = "plus" if sign == "+" else "minus"
        return f"the {name} rail, the long strip of holes along the edge {edge}: {which}, any hole"


def pin_text(pin: str, kit: Kit) -> str:
    """Touch directions for an Arduino socket, with the USB socket pointing left."""
    short = pin.split(":", 1)[1]
    # Near strip first: it's the one beside the breadboard (GND exists on both; the near one is easier to reach).
    for edge, blocks in (("near", kit.arduino_near), ("far", kit.arduino_far)):
        for bi, block in enumerate(blocks):
            if short in block:
                idx = block.index(short) + 1
                block_word = "left-hand" if bi == 0 else "right-hand"
                extra = " (any GND socket works)" if short == "GND" else ""
                return (f"pin {short.replace('3V3', '3.3 volts').replace('5V', '5 volts')} on the Arduino: "
                        f"the socket strip along the {edge} edge, {block_word} block, "
                        f"{ordinal(idx)} socket from the left{extra}")
    return f"pin {short} on the Arduino"


def endpoint_text(name: str, kit: Kit, o: Orientation) -> str:
    n = normalize(name)
    if parse_hole(n):
        return o.hole_text(n)
    if n in _RAILS:
        return o.rail_text(n)
    return pin_text(n, kit)


# ------------------------------------------------------------------ nets
class Nets:
    """Union-find over strips; wires and internally-joined leads merge strips."""

    def __init__(self):
        self._parent: dict[str, str] = {}

    def find(self, a: str) -> str:
        self._parent.setdefault(a, a)
        while self._parent[a] != a:
            self._parent[a] = self._parent[self._parent[a]]
            a = self._parent[a]
        return a

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def same(self, a: str, b: str) -> bool:
        return self.find(a) == self.find(b)


@dataclass
class Connection:
    lead: str
    to: str


@dataclass
class Step:
    kind: str                       # "place" (a part) | "wire" (a jumper wire)
    ref: str = ""                   # part reference, e.g. "LED1" (place steps)
    connections: list[Connection] = field(default_factory=list)
    why: str = ""                   # one short sentence: what this does in the circuit


@dataclass
class PartRef:
    ref: str
    part: str


@dataclass
class TestSpec:
    type: str = "serial"            # "serial" | "listen" | "look"
    key: str = ""                   # serial: the key=value field to watch
    expect: str = "varies"          # serial: "varies" (changes when interacted with) | "present"
    instruction: str = ""           # what the builder should do to test it


@dataclass
class BuildPlan:
    id: str
    title: str
    summary: str
    parts: list[PartRef]
    steps: list[Step]
    sketch: str
    libraries: list[str] = field(default_factory=list)
    test: TestSpec = field(default_factory=TestSpec)
    difficulty: str = "basic"
    source: str = "planner"         # "planner" | "library"

    @classmethod
    def from_dict(cls, d: dict, source: str | None = None) -> "BuildPlan":
        steps = [Step(s.get("kind", "place"), s.get("ref", "") or "",
                      [Connection(c["lead"], normalize(c["to"])) for c in s.get("connections", [])],
                      s.get("why", "") or "")
                 for s in d.get("steps", [])]
        t = d.get("test") or {}
        return cls(
            id=d.get("id") or re.sub(r"[^a-z0-9]+", "_", d.get("title", "build").lower()).strip("_"),
            title=d.get("title", "Your build"), summary=d.get("summary", ""),
            parts=[PartRef(p["ref"], p["part"]) for p in d.get("parts", [])],
            steps=steps, sketch=d.get("sketch", ""), libraries=list(d.get("libraries") or []),
            test=TestSpec(t.get("type", "serial"), t.get("key", ""), t.get("expect", "varies"), t.get("instruction", "")),
            difficulty=d.get("difficulty", "basic"), source=source or d.get("source", "planner"),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id, "title": self.title, "summary": self.summary, "difficulty": self.difficulty,
            "parts": [{"ref": p.ref, "part": p.part} for p in self.parts],
            "steps": [{"kind": s.kind, "ref": s.ref, "why": s.why,
                       "connections": [{"lead": c.lead, "to": c.to} for c in s.connections]} for s in self.steps],
            "sketch": self.sketch, "libraries": self.libraries,
            "test": {"type": self.test.type, "key": self.test.key, "expect": self.test.expect,
                     "instruction": self.test.instruction},
        }

    def part_of(self, ref: str) -> str | None:
        return next((p.part for p in self.parts if p.ref == ref), None)

    @property
    def connection_count(self) -> int:
        return sum(len(s.connections) for s in self.steps)


# ------------------------------------------------------------------ validation
def validate(plan: BuildPlan, kit: Kit | None = None) -> list[str]:
    """Problems that would make the build unsafe, impossible or confusing. Empty list = OK."""
    kit = kit or load_kit()
    errs: list[str] = []
    refs: dict[str, str] = {}

    # parts exist and are in stock
    used: dict[str, int] = {}
    for p in plan.parts:
        if p.part not in kit.parts:
            errs.append(f"{p.ref}: '{p.part}' is not in the kit")
            continue
        if p.ref in refs:
            errs.append(f"part reference {p.ref} is used twice")
        refs[p.ref] = p.part
        used[p.part] = used.get(p.part, 0) + 1
    for pid, n in used.items():
        if n > kit.parts[pid].count:
            errs.append(f"needs {n} x {pid} but the kit has {kit.parts[pid].count}")
    for lib in plan.libraries:
        if lib not in kit.libraries:
            errs.append(f"library '{lib}' is not available (allowed: {', '.join(kit.libraries) or 'none'})")
    for pid in used:
        lib = kit.parts[pid].library
        if lib and lib not in plan.libraries:
            errs.append(f"{pid} needs the '{lib}' library in `libraries`")

    nets = Nets()
    hole_use: dict[str, str] = {}
    placed: dict[str, dict[str, str]] = {}      # ref -> lead -> endpoint
    member_count: dict[str, int] = {}           # endpoint strip -> how many leads/wire ends touch it

    def use_endpoint(ep: str, who: str) -> bool:
        if not is_valid_endpoint(ep, kit):
            errs.append(f"{who}: '{ep}' is not a breadboard hole, rail or Arduino pin")
            return False
        h = parse_hole(ep)
        if h:
            if h[1] > MAX_TOUCH_ROW:
                errs.append(f"{who}: row {h[1]} is too far to count by touch; keep everything in rows 1-{MAX_TOUCH_ROW}")
            if ep in hole_use:
                errs.append(f"{who}: hole {ep} is already used by {hole_use[ep]}; one lead or wire per hole")
            hole_use[ep] = who
        if ep.startswith("RAIL:F"):
            errs.append(f"{who}: use the near rails (RAIL:N+ / RAIL:N-) only, so the builder never reaches across the board")
        member_count[strip_of(ep)] = member_count.get(strip_of(ep), 0) + 1
        return True

    for i, s in enumerate(plan.steps, 1):
        if s.kind == "wire":
            if len(s.connections) != 2:
                errs.append(f"step {i}: a wire has exactly two ends")
                continue
            a, b = s.connections[0].to, s.connections[1].to
            ok = use_endpoint(a, f"step {i} wire") & use_endpoint(b, f"step {i} wire")
            if ok and strip_of(a) == strip_of(b):
                errs.append(f"step {i}: both ends of the wire are in the same connected strip ({a}, {b}); it does nothing")
            if ok:
                nets.union(strip_of(a), strip_of(b))
            continue
        if s.kind != "place":
            errs.append(f"step {i}: unknown step kind '{s.kind}'")
            continue
        pid = refs.get(s.ref)
        if pid is None:
            errs.append(f"step {i}: places {s.ref}, which is not in `parts`")
            continue
        pt = kit.parts[pid]
        if s.ref in placed:
            errs.append(f"step {i}: {s.ref} is placed twice")
        leads = {c.lead: c.to for c in s.connections}
        missing = set(pt.leads) - set(leads)
        unknown = set(leads) - set(pt.leads)
        if missing:
            errs.append(f"step {i}: {s.ref} ({pid}) leads not connected: {', '.join(sorted(missing))}")
        if unknown:
            errs.append(f"step {i}: {s.ref} ({pid}) has no lead called {', '.join(sorted(unknown))} "
                        f"(leads: {', '.join(pt.leads)})")
        for lead, ep in leads.items():
            use_endpoint(ep, f"{s.ref}.{lead}")
        placed[s.ref] = leads
        holes = {lead: parse_hole(ep) for lead, ep in leads.items()}
        if pt.layout == "inline":
            hs = [holes.get(l) for l in pt.leads]
            if all(hs):
                if len({side_of(c) for c, _ in hs}) != 1:
                    errs.append(f"step {i}: {s.ref} legs must all be on one side of the groove")
                rows = [r for _, r in hs]
                if rows != list(range(rows[0], rows[0] + len(rows))) and \
                        rows != list(range(rows[0], rows[0] - len(rows), -1)):
                    errs.append(f"step {i}: {s.ref} legs sit in consecutive rows in this order "
                                f"({', '.join(pt.leads)}); got rows {rows}")
            elif any(hs):
                errs.append(f"step {i}: {s.ref} legs are fixed in a row: put every leg in a breadboard hole")
        if pt.layout == "straddle":
            a, b = holes.get("leg_a"), holes.get("leg_b")
            if not (a and b) or side_of(a[0]) == side_of(b[0]) or abs(a[1] - b[1]) != 2 \
                    or a[0] not in "EF" or b[0] not in "EF":
                errs.append(f"step {i}: {s.ref} must straddle the groove: leg_a in E<row>, leg_b in F<row+2> "
                            f"(or the mirror), the two holes beside the groove")
        if pt.layout == "free" and len(leads) == 2 and all(holes.values()):
            l1, l2 = list(leads.values())
            if strip_of(l1) == strip_of(l2):
                errs.append(f"step {i}: both legs of {s.ref} are in the same connected strip, which shorts it out")

    for ref in refs:
        if ref not in placed:
            errs.append(f"{ref} is in `parts` but never placed")
    if errs:
        return errs

    # electrical checks on the finished netlist
    def net(ep: str) -> str:
        return nets.find(strip_of(ep))

    power_nets = {net(p) for p in POWER_PINS}
    gnd_net = net(GROUND)
    if gnd_net in power_nets:
        errs.append("SHORT CIRCUIT: a power pin (5V/3.3V/VIN) is wired straight to GND")
    for pin in kit.arduino_pins - POWER_PINS - {GROUND}:
        if net(pin) == gnd_net or net(pin) in power_nets:
            if member_count.get(pin, 0):
                errs.append(f"{pin} is wired straight to power or GND; driving it would short the pin")
    for rail in ("RAIL:N+", "RAIL:N-"):
        if member_count.get(rail):
            if not any(net(rail) == net(p) for p in kit.arduino_pins):
                errs.append(f"{rail} is used but never connected to the Arduino (add a wire to 5V or GND)")
    if member_count.get("RAIL:N+") and net("RAIL:N+") == gnd_net:
        errs.append("the plus rail is connected to GND")
    if member_count.get("RAIL:N-") and net("RAIL:N-") in power_nets:
        errs.append("the minus rail is connected to power")

    # every lead must share its net with something else
    net_members: dict[str, int] = {}
    for strip, n in member_count.items():
        net_members[nets.find(strip)] = net_members.get(nets.find(strip), 0) + n
    for pin in {nets.find(p) for p in kit.arduino_pins}:   # the Arduino itself is on the other end
        net_members[pin] = net_members.get(pin, 0) + 1
    for ref, leads in placed.items():
        for lead, ep in leads.items():
            if net_members.get(net(ep), 0) < 2:
                errs.append(f"{ref}.{lead} at {ep} is not connected to anything else")

    # LEDs need a series resistor
    resistor_nets = {net(ep) for ref, leads in placed.items() if kit.parts[refs[ref]].resistor
                     for ep in leads.values()}
    for ref, leads in placed.items():
        pid = refs[ref]
        if not pid.startswith("led_") and pid != "rgb_led":
            continue
        anodes = [ep for lead, ep in leads.items() if lead != "cathode"]
        cath = leads.get("cathode")
        for ep in anodes:
            if net(ep) not in resistor_nets and (cath is None or net(cath) not in resistor_nets):
                errs.append(f"{ref} has no current-limiting resistor in series; add a 220 ohm resistor")
                break
        if cath and net(cath) in power_nets:
            errs.append(f"{ref} cathode (short leg) goes to power; it must go toward GND")

    # polarity of buzzers
    for ref, leads in placed.items():
        if refs[ref].startswith("buzzer") and net(leads["negative"]) in power_nets:
            errs.append(f"{ref} negative leg goes to power; it must go to GND")

    if not plan.sketch.strip():
        errs.append("no Arduino sketch")
    elif "Serial.begin(9600)" not in re.sub(r"\s+", "", plan.sketch):
        errs.append("the sketch must call Serial.begin(9600) so Pinpoint can test it")
    if plan.test.type == "serial" and plan.test.key and f"{plan.test.key}=" not in plan.sketch:
        errs.append(f"the test watches '{plan.test.key}=' but the sketch never prints it")
    return errs
