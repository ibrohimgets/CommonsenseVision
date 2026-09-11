# Commonsense-Guided Open-World Object Detection

Detect the object a person means, even when they describe its purpose instead
of naming its class.

> Example: a request for "something to write with" is mapped to likely objects
> such as a pen or pencil, then grounded to a bounding box in the image.

This research prototype combines scene understanding, semantic retrieval, and
region-level visual matching. It was developed at Dongguk University as part of
work on intent-aware open-world detection and is associated with an accepted
ASK 2025 (KIPS) paper and a Korean patent application.

## Why it matters

Conventional object detectors expect fixed class names. Real users describe
goals: "something to open the box," "what can I sit on?", or "find something
warm to wear." This pipeline translates those requests into visible object
candidates and returns localized results.

Potential applications include assistive vision, visual search, robotics,
inventory discovery, and natural-language interfaces for image collections.

## Pipeline

1. **LLaVA** summarizes the visible scene.
2. **SentenceTransformers** maps the user's intent to an object-trait group.
3. **YOLOv8** proposes candidate regions.
4. **CLIP** scores candidate crops against the inferred object concepts.
5. Non-maximum suppression removes duplicate boxes and the result is rendered.

<p align="center">
  <img src="assets/architecture.png" alt="CommonsenseVision pipeline" width="760" />
</p>

## Repository layout

```text
CommonsenseVision/
|- main.py                    # End-to-end research script
|- src/pipeline.py            # Reusable matching and box utilities
|- configs/knowledge_base.json
|- data/xmls/                 # Example annotations
|- assets/architecture.png
`- requirements.txt
```

## Setup

Python 3.10+ and an NVIDIA GPU are recommended. The LLaVA configuration uses
4-bit loading through `bitsandbytes`; CPU-only execution has not been validated.

```bash
git clone https://github.com/ibrohimgets/CommonsenseVision.git
cd CommonsenseVision
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Before running, set the image, knowledge-base, and output paths in the config
section at the top of `main.py`. Model weights are downloaded by their
respective libraries and are not committed to this repository.

```bash
python main.py
```

The script prompts for a natural-language intent and writes an annotated result
image to the configured output path.

## Current status and limitations

- This is research code, not a hosted production service.
- The current entry point contains experiment-specific file paths that must be
  configured locally.
- The candidate generator is YOLOv8, so objects it fails to propose cannot be
  recovered by the later CLIP stage.
- CLIP thresholds and trait groups require calibration for a new domain.
- No public benchmark report is included here yet; the repository therefore
  does not claim a production accuracy level.

These constraints are documented deliberately so that the evidence is clear
and reproducible claims are not overstated.

## Roadmap

- Replace file-level constants with a command-line interface.
- Add a small publishable example set with annotated successes and failures.
- Add automated tests for semantic matching, box conversion, and NMS.
- Package an API/demo after the research configuration is reproducible.

## Citation

```bibtex
@misc{muminov2025commonsenseod,
  title        = {Commonsense-Guided Open-World Object Detection Using LLMs and Visual-Semantic Reasoning},
  author       = {Muminov, Ibrohim and Kim, Jihie},
  howpublished = {\url{https://github.com/ibrohimgets/CommonsenseVision}},
  year         = {2025}
}
```

## Responsible use

Validate predictions before using this prototype in safety-critical or
high-impact decisions. Do not commit API keys, private images, or model weights.
