# Kitchen Assistant 🍳🍹

A fully local, offline AI teaching assistant for cooking and cocktail-making. Runs entirely on a MacBook M4 — no cloud, no API keys, no subscriptions.

Point it at any YouTube video, Instagram reel, or plain text recipe and it will guide you step by step with voice instructions, real-time camera feedback, and automatic timers — teaching you *why* each step matters, not just what to do.

---

## Features

- **Any recipe source** — YouTube, Instagram reels, plain text. Handles incomplete short-form content by filling gaps from the web.
- **Voice-guided sessions** — reads every step aloud with teaching context using macOS built-in TTS. No screen required.
- **Voice commands** — say "next", "repeat", "skip", "timer", or "stop" hands-free while cooking.
- **Automatic timers** — countdown alerts for timed steps, periodic camera checks for visual conditions.
- **Real-time vision feedback** — camera evaluates your progress and speaks coaching responses ("your onions still look pale, keep going").
- **Dual-camera support** — body-mounted camera for ingredients and pours, fixed overhead camera for shake/stir motion analysis (cocktail mode).
- **Two modes** — cooking and cocktail, sharing a common core with mode-specific vision prompts and step classification.
- **Passive training data collection** — every camera frame is saved with its label for future model fine-tuning.
- **100% local** — Whisper for transcription, Mistral 7B via Ollama for reasoning, moondream/LLaVA for vision. Nothing leaves your machine.

---

## How It Works

```
Input (YouTube / Instagram / text)
        ↓
  Extractor        yt-dlp + Whisper (local audio transcription)
        ↓
  Normalizer       Mistral 7B → structured step list
        ↓
  Gap Detector     DuckDuckGo search → fill missing times/temps/cues
        ↓
  Classifier       Labels each step: timed / condition / instruction
                   Assigns camera: body / fixed / both / none
        ↓
  Session          Walks user through steps with voice + timers + vision
        ↓
  Vision Engine    Camera burst → sharpest frame → moondream or LLaVA
        ↓
  Voice Output     macOS say → spoken teaching feedback
```

---

## Project Structure

```
kitchen_assistant/
├── run.py                          # unified entry point
│
├── core/                           # shared across all modes
│   ├── recipe_engine/
│   │   ├── extractor.py            # YouTube, Instagram, plain text → raw text
│   │   ├── normalizer.py           # raw text → structured step list (LLM)
│   │   ├── gap_detector.py         # web search → fill missing info
│   │   └── recipe_engine.py        # chains all four, CLI entry point
│   │
│   ├── timer_engine/
│   │   ├── session.py              # cooking session coordinator
│   │   ├── timer.py                # CountdownTimer + ConditionPoller
│   │   ├── voice.py                # all spoken output (macOS say)
│   │   └── voice_input.py          # voice command listener (Whisper)
│   │
│   └── vision_engine/
│       ├── vision.py               # stub/startup check, data collection
│       ├── camera.py               # burst capture, sharpness scoring
│       ├── prompts.py              # shared parse_response base
│       └── model_router.py         # shared query_model base
│
└── profiles/                       # mode-specific overrides
    ├── cooking/
    │   ├── classifier.py           # cooking step categories
    │   ├── prompts.py              # colour, size, texture, doneness, liquid
    │   ├── model_router.py         # routes to moondream / LLaVA
    │   └── cook.py                 # cooking entry point
    │
    └── cocktail/
        ├── classifier.py           # shake, stir, pour, garnish categories
        ├── prompts.py              # motion, bottle ID, presentation
        ├── model_router.py         # fixed cam / body cam routing
        ├── camera.py               # adds multi-frame motion capture
        └── bartend.py              # cocktail entry point
```

---

## Requirements

### Hardware
- MacBook M4 (Apple Silicon Neural Engine accelerates Whisper and local models)
- Webcam — built-in or external (index 0)
- Optional: second fixed/overhead camera for cocktail motion analysis (index 1)

### Software

```bash
# Homebrew dependencies
brew install ollama ffmpeg portaudio

# Ollama models (download once)
ollama pull mistral      # recipe parsing + gap filling (~4GB)
ollama pull moondream    # vision checks, fast (~1.6GB)
ollama pull llava        # optional, better for size/motion checks (~4GB)

# Python dependencies
pip install -r requirements.txt
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/atharva9520-hub/kitchen_assistant.git
cd kitchen_assistant

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Start Ollama (keep this running in a separate terminal)
ollama serve
```

---

## Usage

### Cooking mode

```bash
# From a YouTube video
python run.py --mode cook "https://youtube.com/watch?v=..."

# From an Instagram reel
python run.py --mode cook "https://instagram.com/reel/..."

# From plain text
python run.py --mode cook "Dice onion, sauté 8 minutes, add garlic, simmer 15 minutes"

# Use an existing recipe (skip rebuilding)
python run.py --mode cook --recipe recipe_output.json

# Skip web gap-filling (faster, works offline)
python run.py --mode cook --skip-gaps "https://youtube.com/watch?v=..."
```

### Cocktail mode

```bash
python run.py --mode bartend "https://youtube.com/watch?v=..."
python run.py --mode bartend --recipe recipe_output.json
```

### Voice commands during a session

| Say | Action |
|-----|--------|
| "next" / "done" / "ready" | Advance to next step |
| "repeat" / "again" | Hear the current step again |
| "skip" | Skip current step |
| "timer" / "how much time left" | Remaining countdown |
| "stop" / "pause" | Pause session |
| "help" | List all commands |

---

## Vision Module

Vision is disabled by default (`VISION_ENABLED = False` in `core/vision_engine/vision.py`) so the app works fully without a camera during development.

To enable:

```bash
pip install opencv-python
ollama pull moondream
```

Then in `core/vision_engine/vision.py`:
```python
VISION_ENABLED = True
```

### How vision checks work

Each step is classified by category at recipe load time. The category determines which model is used and what the prompt asks:

| Category | Camera | Model | Checks |
|----------|--------|-------|--------|
| colour | body | moondream | golden, translucent, brown |
| size | body | LLaVA | dice evenness, slice thickness |
| texture | body | LLaVA | dough smoothness, sauce consistency |
| liquid | body | moondream | simmer vs boil, reduction |
| doneness | body | moondream | cooked through, set, roasted |
| shake | fixed | LLaVA | vigour, duration (multi-frame) |
| stir | fixed | LLaVA | rotation, ice contact (multi-frame) |
| pour | body | moondream | volume, colour in glass |
| ingredient | body | moondream | correct bottle / label |
| garnish | body | moondream | placement, citrus expression |

### Training data collection

Every frame evaluated by the vision model is automatically saved to `vision_data/frames/` alongside its label (what the model said, whether it called it done). After 20–30 sessions you'll have enough data to fine-tune a lightweight on-device classifier using Apple's Core ML Tools.

---

## Requirements File

```
yt-dlp
openai-whisper
requests
ollama
opencv-python
SpeechRecognition
pyaudio
```

---

## Roadmap

- [ ] Enable vision module with real camera
- [ ] Fine-tune lightweight Core ML classifier on collected training data
- [ ] Simple screen UI showing current step and camera feed
- [ ] PDF / image recipe support
- [ ] Second camera support for cocktail motion analysis
- [ ] Support for more recipe sources (TikTok, recipe websites)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
