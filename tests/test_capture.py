"""Run with: python -m unittest discover tests"""

import csv
import importlib.util
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture import SeatAnalyzer, main

START = datetime(2026, 11, 9, 9, 0, 0)
STILL, FIDGET, MOVING = 0.0, 0.01, 0.1


def feed(analyzer, seconds, motion, face_visible, start=START):
    """Send samples at 5 per second; return the moment after the last sample."""
    moment = start
    for _ in range(round(seconds * 5)):
        analyzer.update(moment, motion, face_visible)
        moment += timedelta(seconds=0.2)
    return moment


class SeatAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = SeatAnalyzer("L-001")

    def row(self):
        rows = self.analyzer.rows()
        self.assertEqual(len(rows), 1)
        return rows[0]

    def test_still_learner_facing_forward_is_fully_engaged(self):
        feed(self.analyzer, 30, STILL, True)
        row = self.row()
        self.assertEqual(row["engagement_pct"], 100.0)
        self.assertEqual((row["movement_events"], row["fidgeting_events"], row["looking_away_events"]), (0, 0, 0))

    def test_sustained_movement_is_one_event(self):
        feed(self.analyzer, 10, MOVING, True)
        self.assertEqual(self.row()["movement_events"], 1)

    def test_movement_separated_by_a_pause_is_two_events(self):
        moment = feed(self.analyzer, 2, MOVING, True)
        moment = feed(self.analyzer, 5, STILL, True, moment)
        feed(self.analyzer, 2, MOVING, True, moment)
        self.assertEqual(self.row()["movement_events"], 2)

    def test_small_motion_counts_as_fidgeting_not_movement(self):
        feed(self.analyzer, 5, FIDGET, True)
        row = self.row()
        self.assertEqual((row["fidgeting_events"], row["movement_events"]), (1, 0))

    def test_brief_look_away_is_ignored(self):
        moment = feed(self.analyzer, 5, STILL, True)
        moment = feed(self.analyzer, 2, STILL, False, moment)
        feed(self.analyzer, 5, STILL, True, moment)
        self.assertEqual(self.row()["looking_away_events"], 0)

    def test_long_look_away_is_one_event_until_the_face_returns(self):
        moment = feed(self.analyzer, 20, STILL, False)
        self.assertEqual(self.row()["looking_away_events"], 1)
        moment = feed(self.analyzer, 2, STILL, True, moment)
        feed(self.analyzer, 5, STILL, False, moment)
        self.assertEqual(self.row()["looking_away_events"], 2)

    def test_large_movement_lowers_engagement_even_when_facing_forward(self):
        moment = feed(self.analyzer, 10, STILL, True)
        feed(self.analyzer, 10, MOVING, True, moment)
        self.assertEqual(self.row()["engagement_pct"], 50.0)

    def test_tracks_how_often_a_face_was_visible(self):
        moment = feed(self.analyzer, 5, STILL, True)
        feed(self.analyzer, 15, STILL, False, moment)
        self.assertAlmostEqual(self.analyzer.face_visible_share(), 0.25)

    def test_samples_are_split_by_minute(self):
        feed(self.analyzer, 150, STILL, True)
        rows = self.analyzer.rows()
        self.assertEqual([row["timestamp"] for row in rows], ["2026-11-09T09:00", "2026-11-09T09:01", "2026-11-09T09:02"])
        self.assertAlmostEqual(rows[0]["observation_minutes"], 1.0)
        self.assertAlmostEqual(rows[2]["observation_minutes"], 0.5)


class CommandLineTests(unittest.TestCase):
    def test_refuses_to_run_without_consent(self):
        with self.assertRaises(SystemExit) as stopped:
            main(["--source", "missing.avi", "--learner-id", "L-001", "--seat", "0,0,10,10"])
        self.assertEqual(stopped.exception.code, 2)

    def test_rejects_ids_that_could_be_paths(self):
        with self.assertRaises(SystemExit):
            main(["--learner-id", "../secret", "--consent-confirmed", "--seat", "0,0,10,10"])

    def test_rejects_invalid_seat(self):
        with self.assertRaises(SystemExit):
            main(["--learner-id", "L-001", "--consent-confirmed", "--seat", "1,2,3"])


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is not installed")
class VideoFileTests(unittest.TestCase):
    def make_video(self, path, seconds=70, fps=10):
        """Quiet room, with one box moving in the seat and another outside it."""
        import cv2
        import numpy as np

        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 360))
        self.assertTrue(writer.isOpened())
        for index in range(seconds * fps):
            frame = np.full((360, 640, 3), 60, np.uint8)
            second = index / fps
            if 10 <= second < 15 or 40 <= second < 45:
                offset = (index * 17) % 120
                cv2.rectangle(frame, (20 + offset, 60), (100 + offset, 200), (240, 240, 240), -1)
            # Constant activity outside the seat must never be measured.
            neighbor = (index * 23) % 200
            cv2.rectangle(frame, (340 + neighbor, 60), (400 + neighbor, 300), (240, 240, 240), -1)
            writer.write(frame)
        writer.release()

    def test_measures_only_the_seat_and_saves_no_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            video = folder / "room.avi"
            self.make_video(video)
            output = folder / "out" / "L-001.csv"
            code = main([
                "--source", str(video), "--learner-id", "L-001", "--consent-confirmed",
                "--seat", "0,0,300,360", "--start", "2026-11-09T09:00", "--output", str(output),
            ])
            self.assertEqual(code, 0)
            with output.open(newline="", encoding="utf-8") as saved:
                rows = list(csv.DictReader(saved))

            self.assertEqual([row["timestamp"] for row in rows], ["2026-11-09T09:00", "2026-11-09T09:01"])
            self.assertEqual(sum(int(row["movement_events"]) for row in rows), 2)
            self.assertEqual({path.name for path in folder.rglob("*") if path.is_file()}, {"room.avi", "L-001.csv"})

    def test_seat_outside_the_video_gives_a_clear_error(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "room.avi"
            self.make_video(video, seconds=2)
            with self.assertRaises(SystemExit) as stopped:
                main(["--source", str(video), "--learner-id", "L-001", "--consent-confirmed", "--seat", "900,900,50,50", "--output", str(Path(folder) / "o.csv")])
            self.assertIn("outside", str(stopped.exception.code))


if __name__ == "__main__":
    unittest.main()
