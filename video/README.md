# TAGFinger research overview — version 2

An anonymous English research film with original vector-style diagrams,
offline synthesized narration, and burned-in English captions.

## Watch

- `../images/TAGFinger_Overview_v2.mp4`: complete narrated overview.
- `preview.html`: local viewing page with the video and storyboard.
- `../images/TAGFinger_Overview_v2_poster.jpg`: representative opening frame.
- `assets/TAGFinger_Overview_v2.srt` / `.vtt`: separate caption files.
- `assets/storyboard_contact.png`: twelve-scene overview for visual review.
- `assets/narration.wav`: reusable narration; re-rendering does not need a TTS model.

The earlier `../images/TAGFinger_Overview.mp4` is preserved, not overwritten.

## Scientific scope

**Method diagrams, text edits, probability bars, and votes are schematic.**
They explain the protocol and are not footage of an experimental run. In
particular, the example text substitution is not claimed to be a successfully
optimized fingerprint. The illustrative stability bars indicate ranking,
not measured similarity values. There is no fabricated alignment percentage.

**Empirical result panels are explicitly paper-reported.** Values come from
the submitted main paper in `../../sec/tagfinger_main.tex`:

| Panel | Setting | Entries used |
| --- | --- | --- |
| Table 1 | Cora-ML, GCN, node task, S@1 | ZeroMark 63.1; GNNFingers 69.7; SIGFinger 71.8; TAGFinger 95.4 |
| Table 3 | Cora-ML / Reddit, node task, S@1 | TAGFinger: 95.0, 94.4, 93.7, 91.1; SIGFinger: 46.5, 44.1, 42.6, 40.3, at 80/60/40/20% Cora-ML |
| Table 4 | GCN surrogate → GAT suspected model, Reddit, S@1 | TAGFinger 93.5 |

S@1 denotes verification success rate subject to FPR ≤ 1%. These are not a
new reproduction of the paper tables by the revised implementation. The
film does not add new experiments, fabricate a no-use control, or treat
zero evidence as a negative verdict.

Protocol details retained in the film:

- The released protected dataset is not pre-modified; audit copies receive edits.
- The suspected model remains frozen; only output probabilities are accessed.
- Candidate selection / prompt optimization use the surrogate, not extra suspect queries.
- Text generation is discrete; gradients do not pass through the language model.
- Task unification requires a compatible graph/subgraph classification interface.
- Target labels remain task-local; evidence indicators, not label meanings, are aggregated.
- A positive vote requires a non-target → target prediction change.
- A same-prediction response is zero, including target → same target.
- Another prediction change is negative. Insufficient votes remain uncertain.

## Re-render

Python 3.11+, Pillow, and an FFmpeg build with `libx264` are sufficient when
reusing `assets/narration.wav` and `assets/timeline.json`.

```bash
python render_video.py validate
python validate_content.py
python render_video.py stills
python render_video.py render --ffmpeg /path/to/ffmpeg
python render_video.py preview --ffmpeg /path/to/ffmpeg
```

To generate a short preview:

```bash
python render_video.py render --scene evidence --output /tmp/evidence.mp4 \
  --ffmpeg /path/to/ffmpeg
```

The delivered film is approximately 3 minutes 22 seconds. The default output
is 1920×1080, 24 fps, H.264 / 48 kHz AAC, YUV 4:2:0, with MP4
fast-start metadata. No browser, online renderer, external image asset,
paper screenshot, or background music is needed. Fonts use Avenir Next on
macOS; Arial or DejaVu Sans can be used on other systems.

### Regenerate narration after script changes

Narration was synthesized locally with Piper 1.8.0 using the US English
`en_US-ljspeech-high` voice. The upstream model card identifies its training
dataset as the public-domain LJ Speech dataset. The model and TTS packages
are not bundled in this repository.

Upstream identifiers:

```text
Piper package: piper-tts 1.8.0
Voice repository: rhasspy/piper-voices
Model path: en/en_US/ljspeech/high/en_US-ljspeech-high.onnx
Config path: en/en_US/ljspeech/high/en_US-ljspeech-high.onnx.json
Model card: en/en_US/ljspeech/high/MODEL_CARD
```

Edit `storyboard.json`, then:

```bash
python render_video.py audio --refresh-audio \
  --voice-model /path/to/en_US-ljspeech-high.onnx \
  --voice-config /path/to/en_US-ljspeech-high.onnx.json
python render_video.py validate
python render_video.py stills
python render_video.py render --ffmpeg /path/to/ffmpeg
```

Audio cue times are measured from synthesized PCM. Scene lengths and SRT/VTT
times are derived from those cue times rather than guessed slide durations.
Changing narration requires regenerating the timeline and the final video.

## Submission note

The film is anonymous and contains no author name, institution, local IDE
path, external project link, or code-hosting identifier. Submission-specific
file-size / duration / archive requirements should still be checked against
the actual submission portal; this renderer does not assume a conference limit.
