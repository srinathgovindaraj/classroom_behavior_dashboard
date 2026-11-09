"""Measure one consenting learner's seat from a classroom camera or video file.

Only per-minute numbers are written to a CSV. Video frames, images, and faces are
never saved, and nothing outside the chosen seat area is analyzed. There is no face
recognition: the face detector only reports whether a forward-facing face is visible.

These are rough camera estimates for caregiver and teacher discussion. They do not
detect or diagnose ADHD or any other condition.

Example:
    python capture.py --source 0 --learner-id L-001 --consent-confirmed
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

FIELDS = [
    "timestamp",
    "learner_id",
    "engagement_pct",
    "movement_events",
    "fidgeting_events",
    "looking_away_events",
    "observation_minutes",
]
SAMPLE_SECONDS = 0.2
PROCESSING_WIDTH = 320
PIXEL_CHANGE_LEVEL = 25
LOW_FACE_SHARE = 0.1


@dataclass(frozen=True)
class Thresholds:
    fidgeting_motion: float = 0.005  # share of seat pixels changing
    movement_motion: float = 0.04
    event_gap_seconds: float = 2.0  # activity closer together than this is one event
    looking_away_seconds: float = 3.0


class SeatAnalyzer:
    """Turn motion and face-visibility samples into per-minute counts."""

    def __init__(self, learner_id: str, sample_seconds: float = SAMPLE_SECONDS, thresholds: Thresholds = Thresholds()):
        self.learner_id = learner_id
        self.sample_seconds = sample_seconds
        self.thresholds = thresholds
        self.minutes: dict[datetime, dict[str, int]] = {}
        self._last_movement: datetime | None = None
        self._last_fidget: datetime | None = None
        self._away_since: datetime | None = None
        self._away_counted = False
        self.samples = 0
        self.face_samples = 0

    def face_visible_share(self) -> float:
        return self.face_samples / self.samples if self.samples else 0.0

    def update(self, moment: datetime, motion: float, face_visible: bool) -> None:
        self.samples += 1
        self.face_samples += int(face_visible)
        minute = self.minutes.setdefault(
            moment.replace(second=0, microsecond=0),
            {"samples": 0, "engaged": 0, "movement": 0, "fidgeting": 0, "looking_away": 0},
        )
        minute["samples"] += 1
        limits = self.thresholds

        if motion >= limits.movement_motion:
            if self._is_new_event(moment, self._last_movement):
                minute["movement"] += 1
            self._last_movement = moment
        elif motion >= limits.fidgeting_motion:
            if self._is_new_event(moment, self._last_fidget):
                minute["fidgeting"] += 1
            self._last_fidget = moment

        if face_visible:
            self._away_since = None
            self._away_counted = False
            if motion < limits.movement_motion:
                minute["engaged"] += 1
        else:
            if self._away_since is None:
                self._away_since = moment
            away_for = (moment - self._away_since).total_seconds()
            if away_for >= limits.looking_away_seconds and not self._away_counted:
                minute["looking_away"] += 1
                self._away_counted = True

    def _is_new_event(self, moment: datetime, last: datetime | None) -> bool:
        return last is None or (moment - last).total_seconds() > self.thresholds.event_gap_seconds

    def rows(self) -> list[dict]:
        rows = []
        for moment, minute in sorted(self.minutes.items()):
            rows.append(
                {
                    "timestamp": moment.isoformat(timespec="minutes"),
                    "learner_id": self.learner_id,
                    "engagement_pct": round(minute["engaged"] / minute["samples"] * 100, 1),
                    "movement_events": minute["movement"],
                    "fidgeting_events": minute["fidgeting"],
                    "looking_away_events": minute["looking_away"],
                    "observation_minutes": round(minute["samples"] * self.sample_seconds / 60, 3),
                }
            )
        return rows


def parse_seat(text: str) -> tuple[int, int, int, int]:
    try:
        x, y, width, height = (int(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("use x,y,width,height, for example 100,80,300,300") from None
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("x and y must be 0 or more; width and height must be above 0")
    return x, y, width, height


def parse_learner_id(text: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", text):
        raise argparse.ArgumentTypeError("use a pseudonymous ID with only letters, numbers, - or _ (for example L-001)")
    return text


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> SeatAnalyzer:
    import cv2  # imported here so the dashboard and tests of the counting logic don't need OpenCV

    source = int(args.source) if str(args.source).isdigit() else args.source
    is_camera = isinstance(source, int)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video source {args.source!r}. Check the camera number or file path.")

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    analyzer = SeatAnalyzer(args.learner_id, thresholds=args.thresholds)
    started = datetime.now() if args.start is None else args.start
    seat = args.seat
    previous = None
    next_sample = 0.0
    ended_by = "end of video"

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if seat is None:
                # The full view is shown once on screen to pick the seat. It is not saved.
                seat = tuple(int(v) for v in cv2.selectROI("Drag a box around the learner's seat, then press Enter", frame))
                cv2.destroyAllWindows()
                if seat[2] == 0 or seat[3] == 0:
                    raise SystemExit("No seat area was selected, so nothing was measured.")

            if is_camera:
                moment = datetime.now()
                elapsed = (moment - started).total_seconds()
            else:
                elapsed = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                moment = started + timedelta(seconds=elapsed)

            if args.minutes is not None and elapsed >= args.minutes * 60:
                ended_by = "time limit"
                break
            if elapsed < next_sample:
                continue
            next_sample = elapsed + SAMPLE_SECONDS

            x, y, width, height = seat
            crop = frame[y : y + height, x : x + width]
            if crop.size == 0:
                raise SystemExit(f"The seat area {seat} is outside the {frame.shape[1]}x{frame.shape[0]} video.")

            scale = PROCESSING_WIDTH / crop.shape[1]
            small = cv2.resize(crop, (PROCESSING_WIDTH, max(1, round(crop.shape[0] * scale))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            if previous is not None:
                changed = cv2.threshold(cv2.absdiff(previous, blurred), PIXEL_CHANGE_LEVEL, 255, cv2.THRESH_BINARY)[1]
                motion = cv2.countNonZero(changed) / changed.size
                face_visible = len(detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))) > 0
                analyzer.update(moment, motion, face_visible)
            previous = blurred

            if args.preview:
                cv2.imshow("Seat area only (press q to stop)", crop)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    ended_by = "q pressed"
                    break
    except KeyboardInterrupt:
        ended_by = "Ctrl+C"
    finally:
        capture.release()
        if args.preview:
            cv2.destroyAllWindows()

    print(f"Stopped: {ended_by}.")
    return analyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write per-minute seat measurements for one consenting learner. No video is saved.")
    parser.add_argument("--source", default="0", help="camera number or video file path (default: camera 0)")
    parser.add_argument("--learner-id", required=True, type=parse_learner_id, help="pseudonymous ID such as L-001; never a name")
    parser.add_argument("--consent-confirmed", action="store_true", help="confirm caregiver consent and school approval are in place")
    parser.add_argument("--seat", type=parse_seat, help="seat area as x,y,width,height; omit to draw it on screen")
    parser.add_argument("--minutes", type=float, help="stop after this many minutes")
    parser.add_argument("--start", type=datetime.fromisoformat, help="video file start time, for example 2026-11-09T09:00")
    parser.add_argument("--output", type=Path, help="CSV to append to (default: observations/<learner-id>.csv)")
    parser.add_argument("--preview", action="store_true", help="show the seat area on screen while measuring")
    parser.add_argument("--fidgeting-motion", type=float, default=Thresholds.fidgeting_motion, help="share of seat pixels changing that counts as fidgeting")
    parser.add_argument("--movement-motion", type=float, default=Thresholds.movement_motion, help="share of seat pixels changing that counts as movement")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.consent_confirmed:
        parser.error("--consent-confirmed is required. Get caregiver consent and school approval before measuring a learner.")
    if not 0 < args.fidgeting_motion < args.movement_motion:
        parser.error("--fidgeting-motion must be above 0 and below --movement-motion.")
    args.thresholds = Thresholds(fidgeting_motion=args.fidgeting_motion, movement_motion=args.movement_motion)

    analyzer = analyze(args)
    rows = analyzer.rows()
    if rows and analyzer.face_visible_share() < LOW_FACE_SHARE:
        print(
            f"Warning: a face was visible in only {analyzer.face_visible_share():.0%} of samples. This usually means the camera "
            "position, distance, or lighting needs fixing, or the seat was empty. Don't read this session as low engagement.",
            file=sys.stderr,
        )
    if not rows:
        print("No measurements were recorded, so no file was written.", file=sys.stderr)
        return 1
    output = args.output or Path("observations") / f"{args.learner_id}.csv"
    append_rows(output, rows)
    print(f"Saved {len(rows)} minute(s) of measurements to {output}. Upload this file to the dashboard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
