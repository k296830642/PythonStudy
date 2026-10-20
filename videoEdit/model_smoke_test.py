"""Verify model download and reference-face embedding generation."""

import face_clip


config, base_dir = face_clip.load_config(face_clip.PROJECT_DIR / "config.json")
matcher = face_clip.FaceMatcher(config, base_dir)
print(f"reference_embedding_dimensions={matcher.reference_embedding.shape[0]}")
print(f"threshold={matcher.threshold}")
