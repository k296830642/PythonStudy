import unittest

import face_clip


class GroupAppearancesTests(unittest.TestCase):
    def test_groups_nearby_matches_and_extends_to_sample_end(self) -> None:
        matches = [
            face_clip.Match(timestamp=1.0, similarity=0.60),
            face_clip.Match(timestamp=1.5, similarity=0.70),
            face_clip.Match(timestamp=2.0, similarity=0.65),
        ]

        appearances = face_clip.group_appearances(matches, sample_fps=2, max_gap=1, minimum_duration=0.8)

        self.assertEqual(1, len(appearances))
        self.assertEqual(1.0, appearances[0].first_match)
        self.assertEqual(2.5, appearances[0].last_match)
        self.assertEqual(0.70, appearances[0].maximum_similarity)

    def test_splits_matches_after_configured_gap(self) -> None:
        matches = [
            face_clip.Match(timestamp=1.0, similarity=0.60),
            face_clip.Match(timestamp=4.0, similarity=0.70),
        ]

        appearances = face_clip.group_appearances(matches, sample_fps=2, max_gap=1, minimum_duration=0.5)

        self.assertEqual(2, len(appearances))

    def test_filters_short_appearances(self) -> None:
        matches = [face_clip.Match(timestamp=1.0, similarity=0.60)]

        appearances = face_clip.group_appearances(matches, sample_fps=2, max_gap=1, minimum_duration=1.0)

        self.assertEqual([], appearances)

    def test_adds_padding_without_exceeding_duration(self) -> None:
        appearances = [face_clip.Appearance(1.0, 3.0, 0.70)]
        config = {"before_seconds": 5.0, "after_seconds": 5.0, "clip_overlap_strategy": "trim"}

        clips = face_clip.make_clips(appearances, duration=6.0, config=config)

        self.assertEqual([face_clip.Clip(0.0, 6.0, 0.70)], clips)

    def test_trims_overlap_between_adjacent_clips(self) -> None:
        appearances = [
            face_clip.Appearance(10.0, 12.0, 0.70),
            face_clip.Appearance(13.0, 15.0, 0.80),
        ]
        config = {"before_seconds": 2.0, "after_seconds": 2.0, "clip_overlap_strategy": "trim"}

        clips = face_clip.make_clips(appearances, duration=30.0, config=config)

        self.assertEqual([face_clip.Clip(8.0, 12.5, 0.70), face_clip.Clip(12.5, 17.0, 0.80)], clips)

    def test_allows_overlap_when_configured(self) -> None:
        appearances = [
            face_clip.Appearance(10.0, 12.0, 0.70),
            face_clip.Appearance(13.0, 15.0, 0.80),
        ]
        config = {"before_seconds": 2.0, "after_seconds": 2.0, "clip_overlap_strategy": "allow"}

        clips = face_clip.make_clips(appearances, duration=30.0, config=config)

        self.assertEqual([face_clip.Clip(8.0, 14.0, 0.70), face_clip.Clip(11.0, 17.0, 0.80)], clips)


if __name__ == "__main__":
    unittest.main()
