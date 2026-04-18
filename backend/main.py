"""
Paper2Pod FastAPI backend.

Pipeline:
  1. Upload a PDF.
  2. PyPDF2 extracts text.
  3. Gemini writes a two-host podcast script (Alex = expert, Sam = curious beginner)
     as strict JSON, tuned to a selected expertise level.
  4. ElevenLabs synthesizes each line concurrently.
  5. Base64 audio clips stream back to the client as Server-Sent Events
     so the UI can render real progress and start playback early.

Bonus endpoint /api/ask lets a listener pause mid-episode, ask a question,
and get a voiced follow-up exchange grounded in the paper's text.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from io import BytesIO
from typing import Any, AsyncIterator, Literal

import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not GEMINI_API_KEY or not ELEVENLABS_API_KEY:
    raise RuntimeError("GEMINI_API_KEY and ELEVENLABS_API_KEY must be set in .env")

ALEX_VOICE_ID = os.getenv("ALEX_VOICE_ID", "pNInz6obpgDQGcFmaJgB")  # Adam
SAM_VOICE_ID = os.getenv("SAM_VOICE_ID", "EXAVITQu4vr4xnSDxMAc")    # Bella

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
ELEVEN_MODEL = os.getenv("ELEVEN_MODEL", "eleven_turbo_v2_5")
TTS_CONCURRENCY = int(os.getenv("TTS_CONCURRENCY", "4"))
MAX_PDF_CHARS = int(os.getenv("MAX_PDF_CHARS", "180000"))

genai.configure(api_key=GEMINI_API_KEY)
eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY)

Expertise = Literal["eli5", "undergrad", "expert"]

EXPERTISE_FLAVOR: dict[Expertise, str] = {
    "eli5": (
        "Audience: a bright high schooler who has never taken this field. "
        "Use vivid analogies. Assume no jargon: either avoid terms or "
        "define them in-line with a one-sentence metaphor."
    ),
    "undergrad": (
        "Audience: an undergraduate in an adjacent field (e.g. a CS undergrad "
        "hearing a biology paper). Explain key terms the first time they appear. "
        "Go one layer deeper than pop-science; stay warm and accessible."
    ),
    "expert": (
        "Audience: a graduate student inside the field. Do not define standard "
        "terminology. Engage seriously with methodology and limitations. "
        "Assume the listener already knows the prior literature roughly."
    ),
}

BASE_SYSTEM_PROMPT = """You are the head writer for "Paper2Pod", a podcast that distills
dense academic research into a warm, intellectually honest conversation between two hosts.

CAST
- Alex: the expert. Deep domain knowledge. Explains with vivid analogies, occasionally
  visibly excited by an elegant idea. Cites specifics from the paper. Never condescending.
- Sam: the curious co-host. Sharp, genuinely curious, asks the questions the audience
  is silently thinking. Pushes back gently when something is hand-wavy. Reacts with wonder.

FORMAT
- Open with a specific hook from the paper: a claim, a number, a counterintuitive result.
  Do NOT open with "welcome to the podcast" or "hello everyone".
- 18–28 total exchanges, alternating speakers, natural rhythm.
- Cover: the problem, the core contribution, the method (make it concrete), the result,
  and why a listener outside the field should care.
- Include one "wait, hold on" moment where Sam challenges a claim and Alex refines it.
- End with a one-sentence takeaway.

SPEECH STYLE
- These lines will be spoken by a TTS model. Write for the ear, not the page.
- Short sentences. Contractions. Occasional natural disfluencies ("yeah", "right",
  "I mean", "so") used sparingly to sound human, not cluttered.
- No stage directions. No "[laughs]". No asterisks. No markdown. No section headers.
- Each line is one to three sentences of pure spoken dialogue.

OUTPUT
Respond with ONLY a JSON array of turns. No preamble, no code fences, no wrapper:

[
  {"speaker": "Alex", "text": "..."},
  {"speaker": "Sam",  "text": "..."}
]

The only allowed values for "speaker" are "Alex" and "Sam".
"""

app = FastAPI(title="Paper2Pod", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Pipeline helpers
# --------------------------------------------------------------------------- #

def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:MAX_PDF_CHARS]


def guess_title(text: str) -> str:
    """Heuristic paper-title extraction from the first ~1500 chars.

    Research PDFs typically lead with the title on its own line(s), followed
    by authors. We take the first non-trivial line that looks like a title.
    """
    head = text[:1500]
    candidates = []
    for line in head.splitlines():
        s = line.strip()
        if not s or len(s) < 10 or len(s) > 220:
            continue
        if s.lower().startswith(("abstract", "introduction", "keywords", "doi:", "arxiv")):
            continue
        # Filter author-like lines (lots of commas, or email addresses)
        if "@" in s or s.count(",") >= 3:
            continue
        candidates.append(s)
        if len(candidates) >= 3:
            break
    if not candidates:
        return "Untitled Paper"
    # Pick the longest of the first few candidates (titles tend to be longer
    # than section labels that slip through).
    return max(candidates, key=len)[:180]


def _strip_code_fences(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def parse_dialogue(raw: str) -> list[dict[str, str]]:
    cleaned = _strip_code_fences(raw)
    data = json.loads(cleaned)
    if isinstance(data, dict):
        for key in ("dialogue", "script", "lines", "turns"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("Gemini did not return a JSON array")
    out: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("speaker", "")).strip().capitalize()
        text = str(item.get("text", "")).strip()
        if speaker not in ("Alex", "Sam") or not text:
            continue
        out.append({"speaker": speaker, "text": text})
    if not out:
        raise ValueError("No valid dialogue turns found in model output")
    return out


async def generate_dialogue(paper_text: str, expertise: Expertise) -> list[dict[str, str]]:
    system = BASE_SYSTEM_PROMPT + "\n\nAUDIENCE\n" + EXPERTISE_FLAVOR[expertise]
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.85,
            top_p=0.95,
            max_output_tokens=8192,
        ),
    )
    prompt = f"Here is the paper text. Write the episode.\n\n---\n{paper_text}\n---"
    response = await asyncio.to_thread(model.generate_content, prompt)
    if not response or not getattr(response, "text", None):
        raise ValueError("Gemini returned an empty response")
    return parse_dialogue(response.text)


async def generate_short(prompt: str, system: str, temperature: float = 0.7) -> str:
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=400,
        ),
    )
    response = await asyncio.to_thread(model.generate_content, prompt)
    text = (getattr(response, "text", "") or "").strip()
    # Kill leading/trailing quotes the model sometimes adds.
    return text.strip('"').strip()


def _synthesize_sync(text: str, voice_id: str) -> bytes:
    audio_iter = eleven.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=ELEVEN_MODEL,
        output_format="mp3_44100_128",
    )
    return b"".join(audio_iter)


async def synthesize_line(text: str, speaker: str) -> bytes:
    voice_id = ALEX_VOICE_ID if speaker == "Alex" else SAM_VOICE_ID
    return await asyncio.to_thread(_synthesize_sync, text, voice_id)


# --------------------------------------------------------------------------- #
# SSE streaming endpoint
# --------------------------------------------------------------------------- #

def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/process")
async def process(
    file: UploadFile = File(...),
    expertise: str = Form("undergrad"),
) -> StreamingResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload must be a .pdf file")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    level: Expertise = expertise if expertise in EXPERTISE_FLAVOR else "undergrad"  # type: ignore[assignment]

    async def event_stream() -> AsyncIterator[str]:
        try:
            yield _sse({"stage": "extracting"})
            text = await asyncio.to_thread(extract_pdf_text, pdf_bytes)
            if not text.strip():
                yield _sse({"stage": "error", "message": "Could not extract text from PDF"})
                return

            title = guess_title(text)
            # The frontend caches the full paper text so the /api/ask endpoint
            # can be called later without re-uploading the PDF.
            yield _sse({
                "stage": "extracted",
                "charCount": len(text),
                "title": title,
                "paperText": text,
            })

            yield _sse({"stage": "scripting"})
            dialogue = await generate_dialogue(text, level)
            yield _sse({
                "stage": "scripted",
                "lineCount": len(dialogue),
                "dialogue": dialogue,
            })

            total = len(dialogue)
            yield _sse({"stage": "synthesizing", "done": 0, "total": total})

            semaphore = asyncio.Semaphore(TTS_CONCURRENCY)
            ready_queue: asyncio.Queue[tuple[int, dict[str, str], bytes]] = asyncio.Queue()

            async def worker(idx: int, line: dict[str, str]) -> None:
                async with semaphore:
                    try:
                        audio = await synthesize_line(line["text"], line["speaker"])
                        await ready_queue.put((idx, line, audio))
                    except Exception as exc:  # noqa: BLE001
                        await ready_queue.put((idx, line, b""))
                        await ready_queue.put((-1, {"error": str(exc), "idx": idx}, b""))

            tasks = [asyncio.create_task(worker(i, l)) for i, l in enumerate(dialogue)]

            done = 0
            while done < total:
                idx, line, audio = await ready_queue.get()
                if idx == -1:
                    yield _sse({"stage": "warning", "message": line.get("error", "")})
                    continue
                done += 1
                yield _sse({
                    "stage": "clip",
                    "index": idx,
                    "speaker": line["speaker"],
                    "text": line["text"],
                    "audio": base64.b64encode(audio).decode("utf-8") if audio else "",
                    "done": done,
                    "total": total,
                })

            await asyncio.gather(*tasks, return_exceptions=True)
            yield _sse({"stage": "done", "total": total})
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            yield _sse({"stage": "error", "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# --------------------------------------------------------------------------- #
# "Ask Alex": interactive follow-up grounded in the paper
# --------------------------------------------------------------------------- #

class PriorTurn(BaseModel):
    speaker: Literal["Alex", "Sam"]
    text: str


class AskRequest(BaseModel):
    paper: str = Field(..., min_length=20)
    question: str = Field(..., min_length=2, max_length=500)
    recent: list[PriorTurn] = Field(default_factory=list)
    expertise: Expertise = "undergrad"


SAM_PHRASING_SYSTEM = """You are Sam, the curious co-host of the Paper2Pod podcast.
A listener just submitted a raw question. Rephrase it as Sam would naturally ask Alex
in the middle of a live conversation: one short sentence, casual, no quotes, no preamble,
no "so anyway" or "quick break". If the question is already well-phrased, keep it nearly
verbatim but swap to Sam's voice.

Respond with the rephrased question only."""


def alex_answer_system(expertise: Expertise) -> str:
    return (
        "You are Alex, the expert host of the Paper2Pod podcast. Sam just asked a "
        "question in the middle of your episode. Answer in character using ONLY the "
        "paper text provided as ground truth. Be specific: cite numbers, names, or "
        "concrete details from the paper when relevant.\n\n"
        f"{EXPERTISE_FLAVOR[expertise]}\n\n"
        "RULES\n"
        "- 2 to 4 sentences.\n"
        "- Spoken style, contractions, warm and natural.\n"
        "- If the paper truly does not address the question, say so plainly in one "
        "sentence and offer the closest adjacent thing the paper does cover.\n"
        "- No preamble (no 'great question', no 'so'). No markdown. No quotes around "
        "the answer."
    )


@app.post("/api/ask")
async def ask(req: AskRequest) -> dict[str, Any]:
    paper = req.paper[:MAX_PDF_CHARS]
    recent_str = "\n".join(f"{t.speaker}: {t.text}" for t in req.recent[-6:])

    sam_line = await generate_short(
        prompt=f"Listener asked: {req.question}",
        system=SAM_PHRASING_SYSTEM,
        temperature=0.6,
    )
    if not sam_line:
        sam_line = req.question.strip()

    alex_prompt = (
        (f"RECENT CONVERSATION:\n{recent_str}\n\n" if recent_str else "")
        + f"SAM JUST ASKED:\n{sam_line}\n\n"
        + f"PAPER TEXT (ground truth):\n---\n{paper}\n---\n\n"
        + "Now answer as Alex."
    )
    alex_line = await generate_short(
        prompt=alex_prompt,
        system=alex_answer_system(req.expertise),
        temperature=0.55,
    )
    if not alex_line:
        raise HTTPException(status_code=502, detail="Model returned an empty answer")

    sam_audio, alex_audio = await asyncio.gather(
        synthesize_line(sam_line, "Sam"),
        synthesize_line(alex_line, "Alex"),
    )

    return {
        "turns": [
            {
                "speaker": "Sam",
                "text": sam_line,
                "audio": base64.b64encode(sam_audio).decode("utf-8"),
            },
            {
                "speaker": "Alex",
                "text": alex_line,
                "audio": base64.b64encode(alex_audio).decode("utf-8"),
            },
        ]
    }


# --------------------------------------------------------------------------- #
# Health / meta
# --------------------------------------------------------------------------- #

@app.get("/")
def root() -> dict[str, Any]:
    return {
        "app": "paper2pod",
        "status": "ok",
        "features": ["streaming-pipeline", "ask-alex", "expertise-levels"],
    }


@app.get("/api/voices")
def voices() -> dict[str, Any]:
    return {
        "Alex": {"voice_id": ALEX_VOICE_ID, "persona": "expert"},
        "Sam": {"voice_id": SAM_VOICE_ID, "persona": "curious co-host"},
        "model": ELEVEN_MODEL,
    }
