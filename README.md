# Paper2Pod

> Turn any academic paper into a two-host podcast you can interrupt and argue with.

Built for **AI Hackfest** (MLH, April 17-19, 2026).

Paper2Pod takes a PDF and hands you a streaming, fully-voiced conversation between two AI hosts: **Alex** (the expert) and **Sam** (the curious co-host). It finishes in under a minute. Mid-episode, you can pause and ask your own question. Sam voices it, Alex answers it using the paper's actual text, and the episode resumes right where you left off.

Everything runs straight through: one PDF in, one podcast out, no account, no database, no ceremony.

[Live Demo]([URL](https://frontend-rho-lilac-20.vercel.app/))

---

## What makes this different from NotebookLM

NotebookLM turns papers into monologue-style audio overviews. Paper2Pod is a **live conversation you can interrupt**. The headline feature isn't the script. It's the `Ask Alex` overlay that pauses the episode, takes your follow-up, generates a voiced exchange grounded in the paper, and resumes.

We also pitch the script at the listener's level (_Explain like I'm 15_, _Curious outsider_, _Grad student_), stream every pipeline stage to the UI over SSE, synthesize voice lines concurrently so audio starts before the full episode is done rendering, and let you download the whole thing (including your Q&A interjections) as a single MP3.

---

## Sponsor prize targets

### Best Use of Gemini API

Gemini 2.5 Flash Lite is the creative and reasoning core of the app:

- **Structured dialogue synthesis.** A system prompt with cast definitions, pacing constraints, and spoken-style rules (contractions, disfluencies, no stage directions) produces a strict JSON array of turns via Gemini's `response_mime_type="application/json"`.
- **Long-context grounding.** The full paper (up to ~60K chars) is passed in a single shot, with no chunking and no vector store, because Flash handles it cleanly.
- **Dynamic audience calibration.** The system prompt is composed at request time from one of three expertise profiles (ELI5 / Undergrad / Grad student), demonstrably shifting vocabulary, analogy density, and depth.
- **Two-role generation for the Q&A feature.** When the user asks a follow-up, Gemini is called twice: first to rephrase the question in Sam's voice, then to answer as Alex using the paper as ground truth. Both answers chain through TTS.

### Best Use of ElevenLabs

ElevenLabs is responsible for every sound the user hears:

- **Two distinct, expressive voices** (Adam for Alex, Bella for Sam) synthesized via `eleven_turbo_v2_5` for production-latency playback.
- **Concurrent TTS.** Lines are rendered in parallel behind an `asyncio.Semaphore(4)`, with clips streamed to the client by base64 over SSE as each one completes. Listeners start hearing the episode before the last lines are even rendered.
- **An interactive audio companion.** The `Ask Alex` feature makes the podcast _conversational_: the user's typed question becomes a fresh, in-character, emotionally-pitched exchange generated on the fly. This is the prize brief's "interactive AI companion" made literal.
- **Client-side concatenation.** All clips (including inserted Q&A turns) can be downloaded as a single MP3, assembled in the browser with no re-encoding.

---

## Quickstart

You'll need Python 3.10+, Node 18+, a Gemini API key, and an ElevenLabs API key.

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then paste your real keys into .env
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# http://localhost:5173
```

Drop a PDF. Wait ~30–60 seconds. Press **A** at any point during playback to ask Alex a question.

---

## Architecture

```
   PDF upload
        │
        ▼
  ┌───────────────────────┐
  │ FastAPI /api/process  │── SSE ──► React frontend
  │                       │           (live stage updates,
  │  1. PyPDF2 extract    │            per-clip streaming,
  │  2. Gemini → script   │            visualizer, player)
  │  3. ElevenLabs (×N    │
  │     concurrent)       │
  └───────────────────────┘

   Mid-episode listener question
        │
        ▼
  ┌───────────────────────┐
  │ FastAPI /api/ask      │
  │                       │
  │  Gemini(Sam rephrase) │─┐
  │  Gemini(Alex answer)  │ │── paper text + recent dialogue
  │  ElevenLabs × 2       │ │   as grounding context
  │                       │ │
  └───────────────────────┘─┘
        │
        ▼
   inline insertion into player queue,
   auto-resume at the exact timestamp
```

---

## Project structure

```
paper-to-pod/
├── backend/
│   ├── main.py             Single-file FastAPI app (pipeline + /api/ask)
│   ├── requirements.txt
│   ├── .env                Your real keys (gitignored)
│   └── .env.example
└── frontend/
    ├── index.html
    ├── package.json        Vite 5 + React 18 + Tailwind v4
    ├── vite.config.ts      @tailwindcss/vite plugin
    └── src/
        ├── App.tsx         Single-file UI (~900 LOC)
        ├── main.tsx
        └── index.css       Tailwind v4 @theme tokens + ambient bg
```

---

## Feature matrix

| Feature                                 | Where it lives                                          | Why it scores                                                                 |
| --------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------- |
| SSE streaming pipeline                  | `backend/main.py` · `/api/process`                      | Real progress UI. Judges see real stage transitions, not fake spinners        |
| Concurrent TTS with `asyncio.Semaphore` | `backend/main.py` · `worker()`                          | ~4× faster episode generation; clips start arriving before the rest finish    |
| Ask Alex mid-episode Q&A                | `backend/main.py` · `/api/ask`, `AskModal` in `App.tsx` | Core originality play, novel vs NotebookLM, doubles Gemini + ElevenLabs usage |
| Expertise calibration                   | `EXPERTISE_FLAVOR` in backend, pills in Hero            | Same paper, three very different episodes. Easy to demo in a video            |
| Auto paper-title extraction             | `guess_title()`                                         | Header polish; no extra call                                                  |
| Radial Web Audio visualizer             | `useAudioVisualizer`                                    | Speaker-tinted frequency bars driven by `AnalyserNode`                        |
| Click-to-jump transcript                | `Player` component                                      | UX signal, shows Q&A inline under the clip it was inserted after              |
| Download full episode as MP3            | `downloadEpisode()`                                     | Client-side byte concatenation across main + Q&A clips                        |
| Keyboard shortcuts                      | `Space` = play/pause, `A` = Ask Alex                    | Demo-friendly                                                                 |

---

## License

MIT. See `LICENSE`.
