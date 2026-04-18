import {
  DragEvent,
  KeyboardEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

// -----------------------------------------------------------------------------
// Paper2Pod, single-file React app.
//
// Three screens:
//   1. Hero / dropzone with an expertise selector.
//   2. Streaming status panel while the SSE pipeline runs.
//   3. Player with a Web Audio visualizer, click-to-jump transcript, and
//      an "Ask Alex" overlay that pauses the episode to record a voiced
//      follow-up grounded in the paper text.
// -----------------------------------------------------------------------------

const API_BASE = "http://localhost:8000";

type SpeakerName = "Alex" | "Sam";
type Expertise = "eli5" | "undergrad" | "expert";

interface Clip {
  speaker: SpeakerName;
  text: string;
  audio: string; // base64 mp3
}

interface DialogueLine {
  speaker: SpeakerName;
  text: string;
}

type Stage =
  | "idle"
  | "extracting"
  | "scripting"
  | "synthesizing"
  | "ready"
  | "error";

interface Progress {
  done: number;
  total: number;
}

interface QA {
  mainIdx: number;         // the main-clip index this Q&A was inserted after
  resumeTime: number;      // where in that main clip we paused
  turns: Clip[];           // Sam's question + Alex's answer
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

function base64ToBlobUrl(b64: string, mime = "audio/mpeg"): string {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], { type: mime }));
}

function formatTime(sec: number): string {
  if (!isFinite(sec) || sec < 0) return "0:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

async function responseErrorMessage(res: Response, fallback: string): Promise<string> {
  try {
    const data = await res.clone().json();
    if (typeof data.detail === "string" && data.detail.trim()) return data.detail;
    if (typeof data.message === "string" && data.message.trim()) return data.message;
  } catch {
    /* use fallback */
  }
  return fallback;
}

const STAGE_COPY: Record<Exclude<Stage, "idle" | "ready" | "error">, string> = {
  extracting: "Reading the paper",
  scripting: "Writing the script",
  synthesizing: "Recording the hosts",
};

const EXPERTISE_OPTIONS: { value: Expertise; label: string; blurb: string }[] = [
  { value: "eli5",     label: "Explain like I'm 15", blurb: "No jargon. All analogies." },
  { value: "undergrad", label: "Curious outsider",    blurb: "Adjacent-field undergrad." },
  { value: "expert",   label: "Grad student",         blurb: "Dive into methodology." },
];

// -----------------------------------------------------------------------------
// Audio visualizer: radial bars driven by an AnalyserNode.
// -----------------------------------------------------------------------------

function useAudioVisualizer(
  audioRef: React.RefObject<HTMLAudioElement>,
  canvasRef: React.RefObject<HTMLCanvasElement>,
  active: boolean,
  speaker: SpeakerName | null,
) {
  const ctxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const speakerRef = useRef<SpeakerName | null>(speaker);
  speakerRef.current = speaker;

  const ensureGraph = useCallback(() => {
    const el = audioRef.current;
    if (!el) return;
    if (!ctxRef.current) {
      const AC = window.AudioContext || (window as any).webkitAudioContext;
      ctxRef.current = new AC();
    }
    const ac = ctxRef.current!;
    if (!sourceRef.current) {
      sourceRef.current = ac.createMediaElementSource(el);
      analyserRef.current = ac.createAnalyser();
      analyserRef.current.fftSize = 256;
      analyserRef.current.smoothingTimeConstant = 0.82;
      sourceRef.current.connect(analyserRef.current);
      analyserRef.current.connect(ac.destination);
    }
    if (ac.state === "suspended") void ac.resume();
  }, [audioRef]);

  useEffect(() => {
    if (!active) return;
    ensureGraph();
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    if (!canvas || !analyser) return;
    const ctx = canvas.getContext("2d")!;
    const buf = new Uint8Array(analyser.frequencyBinCount);

    const render = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth * dpr;
      const h = canvas.clientHeight * dpr;
      if (canvas.width !== w) canvas.width = w;
      if (canvas.height !== h) canvas.height = h;

      analyser.getByteFrequencyData(buf);
      ctx.clearRect(0, 0, w, h);

      const bars = 56;
      const step = Math.floor(buf.length / bars);
      const cx = w / 2;
      const cy = h / 2;
      const baseR = Math.min(w, h) * 0.22;
      const maxLift = Math.min(w, h) * 0.18;
      const tint =
        speakerRef.current === "Sam" ? "124, 198, 255" : "179, 164, 255";

      ctx.save();
      ctx.translate(cx, cy);
      for (let i = 0; i < bars; i++) {
        const v = buf[i * step] / 255;
        const angle = (i / bars) * Math.PI * 2 - Math.PI / 2;
        const lift = Math.pow(v, 1.4) * maxLift + 2;
        const x1 = Math.cos(angle) * baseR;
        const y1 = Math.sin(angle) * baseR;
        const x2 = Math.cos(angle) * (baseR + lift);
        const y2 = Math.sin(angle) * (baseR + lift);
        const alpha = 0.35 + v * 0.65;
        ctx.strokeStyle = `rgba(${tint}, ${alpha.toFixed(3)})`;
        ctx.lineWidth = 2 * dpr;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
      const g = ctx.createRadialGradient(0, 0, 0, 0, 0, baseR);
      g.addColorStop(0, `rgba(${tint}, 0.35)`);
      g.addColorStop(1, "rgba(10,10,15,0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(0, 0, baseR, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      rafRef.current = requestAnimationFrame(render);
    };
    rafRef.current = requestAnimationFrame(render);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, [active, canvasRef, ensureGraph]);

  return { ensureGraph };
}

// -----------------------------------------------------------------------------
// Main component
// -----------------------------------------------------------------------------

export default function App() {
  // Pipeline state
  const [stage, setStage] = useState<Stage>("idle");
  const [progress, setProgress] = useState<Progress>({ done: 0, total: 0 });
  const [dialogue, setDialogue] = useState<DialogueLine[]>([]);
  const [clips, setClips] = useState<(Clip | null)[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [isDragging, setIsDragging] = useState(false);
  const [expertise, setExpertise] = useState<Expertise>("undergrad");
  const [paperText, setPaperText] = useState<string>("");
  const [paperTitle, setPaperTitle] = useState<string>("");

  // Playback state for the main episode
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [awaitingNext, setAwaitingNext] = useState(false);

  // Playback state for the Q&A overlay
  const [qaByMainIdx, setQaByMainIdx] = useState<Record<number, Clip[]>>({});
  const [qaActive, setQaActive] = useState<QA | null>(null);
  const [qaTurnIdx, setQaTurnIdx] = useState<number>(0);

  // Q&A modal
  const [askOpen, setAskOpen] = useState(false);
  const [askText, setAskText] = useState("");
  const [askLoading, setAskLoading] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);

  const audioRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const currentUrlRef = useRef<string | null>(null);
  const askInputRef = useRef<HTMLTextAreaElement>(null);
  const processAbortRef = useRef<AbortController | null>(null);
  const processRequestIdRef = useRef(0);
  const askAbortRef = useRef<AbortController | null>(null);
  const askRequestIdRef = useRef(0);
  const shouldAutoPlayRef = useRef(false);
  const pendingMainSeekRef = useRef<number | null>(null);

  // Derived: who is currently speaking?
  const currentClip: Clip | null = qaActive
    ? qaActive.turns[qaTurnIdx] ?? null
    : clips[currentIndex] ?? null;

  const { ensureGraph } = useAudioVisualizer(
    audioRef,
    canvasRef,
    isPlaying && stage === "ready",
    currentClip?.speaker ?? null,
  );

  // ---------------------------------------------------------------------------
  // Upload + SSE
  // ---------------------------------------------------------------------------

  const startProcessing = useCallback(async (file: File) => {
    resetState();
    const requestId = ++processRequestIdRef.current;
    const controller = new AbortController();
    processAbortRef.current = controller;
    setFileName(file.name);
    setStage("extracting");

    const form = new FormData();
    form.append("file", file);
    form.append("expertise", expertise);

    let res: Response;
    try {
      res = await fetch(`${API_BASE}/api/process`, {
        method: "POST",
        body: form,
        signal: controller.signal,
      });
    } catch (err) {
      if (controller.signal.aborted || (err instanceof DOMException && err.name === "AbortError")) {
        return;
      }
      if (processAbortRef.current === controller) processAbortRef.current = null;
      setError("Could not reach the backend. Is it running on :8000?");
      setStage("error");
      return;
    }
    if (requestId !== processRequestIdRef.current || controller.signal.aborted) return;
    if (!res.ok || !res.body) {
      if (processAbortRef.current === controller) processAbortRef.current = null;
      const message = await responseErrorMessage(res, `Server returned ${res.status}`);
      if (requestId !== processRequestIdRef.current || controller.signal.aborted) return;
      setError(message);
      setStage("error");
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    const applyFrame = (frame: string) => {
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        if (requestId !== processRequestIdRef.current || controller.signal.aborted) return;
        try {
          const ev = JSON.parse(line.slice(6));
          handleEvent(ev);
        } catch {
          /* ignore malformed frame */
        }
      }
    };

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (requestId !== processRequestIdRef.current || controller.signal.aborted) {
          await reader.cancel().catch(() => {});
          return;
        }
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const frames = buf.split("\n\n");
        buf = frames.pop() ?? "";
        for (const frame of frames) applyFrame(frame);
      }
      buf += decoder.decode();
      if (buf.trim()) applyFrame(buf);
    } catch {
      if (controller.signal.aborted || requestId !== processRequestIdRef.current) return;
      setError("The backend connection closed while processing the paper.");
      setStage("error");
    } finally {
      if (processAbortRef.current === controller) {
        processAbortRef.current = null;
      }
    }
  }, [expertise]);

  const isPlayableClip = (value: any): value is Clip => {
    return (
      value &&
      (value.speaker === "Alex" || value.speaker === "Sam") &&
      typeof value.text === "string" &&
      value.text.trim().length > 0 &&
      typeof value.audio === "string" &&
      value.audio.length > 0
    );
  };

  const isClipPayload = (value: any): value is Clip & { index: number; done: number; total: number } => {
    return (
      isPlayableClip(value) &&
      Number.isInteger((value as any).index) &&
      Number.isFinite((value as any).done) &&
      Number.isFinite((value as any).total)
    );
  };

  const handleEvent = useCallback((ev: any) => {
    switch (ev.stage) {
      case "extracting":
        setStage("extracting");
        break;
      case "extracted":
        if (ev.paperText) setPaperText(ev.paperText);
        if (ev.title) setPaperTitle(ev.title);
        break;
      case "scripting":
        setStage("scripting");
        break;
      case "scripted":
        setDialogue(ev.dialogue ?? []);
        setClips(new Array(ev.lineCount).fill(null));
        setProgress({ done: 0, total: ev.lineCount });
        setStage("synthesizing");
        break;
      case "synthesizing":
        setProgress({ done: ev.done ?? 0, total: ev.total ?? 0 });
        break;
      case "clip":
        if (!isClipPayload(ev)) {
          setError(`Missing or invalid audio for line ${(ev.index ?? 0) + 1}.`);
          setStage("error");
          break;
        }
        setClips((prev) => {
          const next = prev.slice();
          next[ev.index] = {
            speaker: ev.speaker,
            text: ev.text,
            audio: ev.audio,
          };
          return next;
        });
        setProgress({ done: ev.done, total: ev.total });
        break;
      case "done":
        setStage("ready");
        break;
      case "warning":
        console.warn("[paper2pod]", ev.message);
        break;
      case "error":
        setError(ev.message || "Something went wrong");
        setStage("error");
        break;
      default:
        break;
    }
  }, []);

  const resetState = () => {
    processRequestIdRef.current += 1;
    if (processAbortRef.current) {
      processAbortRef.current.abort();
      processAbortRef.current = null;
    }
    askRequestIdRef.current += 1;
    if (askAbortRef.current) {
      askAbortRef.current.abort();
      askAbortRef.current = null;
    }
    shouldAutoPlayRef.current = false;
    pendingMainSeekRef.current = null;
    const el = audioRef.current;
    if (el) {
      el.pause();
      el.removeAttribute("src");
      el.load();
    }
    if (currentUrlRef.current) {
      URL.revokeObjectURL(currentUrlRef.current);
      currentUrlRef.current = null;
    }
    setError(null);
    setDialogue([]);
    setClips([]);
    setProgress({ done: 0, total: 0 });
    setCurrentIndex(0);
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    setAwaitingNext(false);
    setPaperText("");
    setPaperTitle("");
    setQaByMainIdx({});
    setQaActive(null);
    setQaTurnIdx(0);
    setAskOpen(false);
    setAskText("");
    setAskError(null);
    setAskLoading(false);
  };

  // ---------------------------------------------------------------------------
  // Dropzone handlers
  // ---------------------------------------------------------------------------

  const onFiles = (files: FileList | null) => {
    if (!files || !files[0]) return;
    const f = files[0];
    if (!f.name.toLowerCase().endsWith(".pdf")) {
      setError("Please drop a PDF file.");
      setStage("error");
      return;
    }
    void startProcessing(f);
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    onFiles(e.dataTransfer.files);
  };

  // ---------------------------------------------------------------------------
  // Unified audio src wiring, driven by qaActive and currentIndex.
  // ---------------------------------------------------------------------------

  const totalClips = clips.length;

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    const clip = currentClip;
    if (!clip) {
      setAwaitingNext(!qaActive && totalClips > 0 && currentIndex < totalClips);
      return;
    }
    if (!clip.audio) {
      setError("Missing audio for the current line.");
      setStage("error");
      return;
    }
    setAwaitingNext(false);
    if (currentUrlRef.current) URL.revokeObjectURL(currentUrlRef.current);
    const url = base64ToBlobUrl(clip.audio);
    currentUrlRef.current = url;

    const shouldPlay = !!qaActive || shouldAutoPlayRef.current;
    const hasPendingSeek = pendingMainSeekRef.current != null;

    const playCurrent = () => {
      ensureGraph();
      void el.play().catch(() => {
        if (!qaActive) shouldAutoPlayRef.current = false;
        setIsPlaying(false);
      });
    };

    const onMeta = () => {
      const seekTo = pendingMainSeekRef.current;
      if (seekTo != null) {
        const max = Number.isFinite(el.duration) ? el.duration : seekTo;
        el.currentTime = Math.max(0, Math.min(max, seekTo));
        pendingMainSeekRef.current = null;
      }
      if (shouldPlay) playCurrent();
    };

    if (hasPendingSeek) el.addEventListener("loadedmetadata", onMeta, { once: true });
    el.src = url;
    if (!hasPendingSeek && shouldPlay) {
      playCurrent();
    }
    return () => {
      if (hasPendingSeek) el.removeEventListener("loadedmetadata", onMeta);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentIndex, currentClip?.audio, qaActive?.mainIdx, qaTurnIdx, totalClips]);

  const togglePlay = () => {
    const el = audioRef.current;
    if (!el) return;
    ensureGraph();
    if (el.paused) {
      shouldAutoPlayRef.current = true;
      void el.play().catch(() => {
        shouldAutoPlayRef.current = false;
        setIsPlaying(false);
      });
    } else {
      shouldAutoPlayRef.current = false;
      el.pause();
    }
  };

  const onEnded = () => {
    if (qaActive) {
      if (qaTurnIdx < qaActive.turns.length - 1) {
        setQaTurnIdx((i) => i + 1);
        return;
      }
      // End of Q&A overlay → resume main episode where we left off.
      const resume = qaActive;
      pendingMainSeekRef.current = resume.resumeTime;
      shouldAutoPlayRef.current = true;
      setQaActive(null);
      setQaTurnIdx(0);
      setCurrentIndex(resume.mainIdx);
      return;
    }
    if (currentIndex < totalClips - 1) {
      shouldAutoPlayRef.current = true;
      setCurrentIndex((i) => i + 1);
    } else {
      shouldAutoPlayRef.current = false;
      setIsPlaying(false);
    }
  };

  const onTimeUpdate = () => {
    const el = audioRef.current;
    if (!el) return;
    setCurrentTime(el.currentTime);
  };

  const onLoadedMetadata = () => {
    const el = audioRef.current;
    if (!el) return;
    setDuration(el.duration);
  };

  const seek = (pct: number) => {
    const el = audioRef.current;
    if (!el || !isFinite(duration)) return;
    el.currentTime = Math.max(0, Math.min(duration, pct * duration));
  };

  const skipNext = () => {
    if (qaActive) return;
    if (currentIndex < totalClips - 1) setCurrentIndex((i) => i + 1);
  };
  const skipPrev = () => {
    if (qaActive) return;
    const el = audioRef.current;
    if (el && el.currentTime > 2) {
      el.currentTime = 0;
      return;
    }
    if (currentIndex > 0) setCurrentIndex((i) => i - 1);
  };

  // ---------------------------------------------------------------------------
  // "Ask Alex" flow
  // ---------------------------------------------------------------------------

  const openAsk = () => {
    if (!paperText || stage !== "ready" || qaActive) return;
    const el = audioRef.current;
    shouldAutoPlayRef.current = false;
    if (el) el.pause();
    setAskOpen(true);
    setTimeout(() => askInputRef.current?.focus(), 80);
  };

  const closeAsk = () => {
    askRequestIdRef.current += 1;
    if (askAbortRef.current) {
      askAbortRef.current.abort();
      askAbortRef.current = null;
    }
    setAskOpen(false);
    setAskError(null);
    setAskText("");
    setAskLoading(false);
  };

  const submitAsk = async () => {
    const question = askText.trim();
    if (!question || askLoading || qaActive) return;
    askAbortRef.current?.abort();
    const controller = new AbortController();
    askAbortRef.current = controller;
    const requestId = ++askRequestIdRef.current;
    setAskLoading(true);
    setAskError(null);

    const el = audioRef.current;
    const resumeTime = el?.currentTime ?? 0;
    const mainIdx = currentIndex;

    // Pass the last few transcript lines for conversational grounding.
    const recent = clips
      .slice(Math.max(0, mainIdx - 4), mainIdx + 1)
      .filter((c): c is Clip => c != null)
      .map((c) => ({ speaker: c.speaker, text: c.text }));

    try {
      const res = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          paper: paperText,
          question,
          recent,
          expertise,
        }),
        signal: controller.signal,
      });
      if (requestId !== askRequestIdRef.current || controller.signal.aborted) return;
      if (!res.ok) {
        const message = await responseErrorMessage(res, `Server returned ${res.status}`);
        if (requestId !== askRequestIdRef.current || controller.signal.aborted) return;
        setAskError(message);
        return;
      }
      const data = await res.json();
      if (requestId !== askRequestIdRef.current || controller.signal.aborted) return;
      const turns = Array.isArray(data.turns) ? data.turns.filter(isPlayableClip) : [];
      if (turns.length === 0 || turns.length !== data.turns?.length) {
        setAskError("No answer came back. Try rephrasing.");
        return;
      }
      setQaByMainIdx((prev) => {
        const next = { ...prev };
        next[mainIdx] = [...(next[mainIdx] ?? []), ...turns];
        return next;
      });
      setQaActive({ mainIdx, resumeTime, turns });
      setQaTurnIdx(0);
      setAskOpen(false);
      setAskText("");
    } catch (err) {
      if (controller.signal.aborted || (err instanceof DOMException && err.name === "AbortError")) {
        return;
      }
      setAskError("Couldn't reach the backend.");
    } finally {
      if (requestId === askRequestIdRef.current) {
        setAskLoading(false);
      }
      if (askAbortRef.current === controller) {
        askAbortRef.current = null;
      }
    }
  };

  // Keyboard shortcut: "A" opens Ask Alex when the player is ready.
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (stage !== "ready") return;
      if (askOpen) return;
      if (qaActive && (e.key === "a" || e.key === "A")) return;
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
      if (e.key === "a" || e.key === "A") {
        e.preventDefault();
        openAsk();
      } else if (e.key === " ") {
        e.preventDefault();
        togglePlay();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stage, askOpen, paperText, qaActive]);

  // ---------------------------------------------------------------------------
  // Download full episode (concatenated mp3, includes Q&A in order)
  // ---------------------------------------------------------------------------

  const canDownload = stage === "ready" && clips.every((c) => c != null && c.audio.length > 0);
  const downloadEpisode = () => {
    if (!canDownload) return;
    const ordered: Clip[] = [];
    for (let i = 0; i < clips.length; i++) {
      ordered.push(clips[i] as Clip);
      const qa = qaByMainIdx[i];
      if (qa) ordered.push(...qa);
    }
    const parts: BlobPart[] = [];
    for (const c of ordered) {
      const bin = atob(c.audio);
      const buffer = new ArrayBuffer(bin.length);
      const bytes = new Uint8Array(buffer);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      parts.push(buffer);
    }
    const blob = new Blob(parts, { type: "audio/mpeg" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const stem = (paperTitle || fileName.replace(/\.pdf$/i, "") || "episode").slice(0, 80);
    a.download = `${stem} - Paper2Pod.mp3`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ---------------------------------------------------------------------------
  // Derived UI state
  // ---------------------------------------------------------------------------

  const progressPct = useMemo(() => {
    if (!progress.total) return 0;
    return Math.min(1, progress.done / progress.total);
  }, [progress]);

  const clipProgressPct = useMemo(() => {
    if (!duration || !isFinite(duration)) return 0;
    return currentTime / duration;
  }, [currentTime, duration]);

  const subStageText = (() => {
    if (stage === "extracting") return "Parsing every page";
    if (stage === "scripting") return "Two hosts. One paper. No filler.";
    if (stage === "synthesizing")
      return progress.total
        ? `${progress.done} of ${progress.total} lines recorded`
        : "Warming up the studio";
    return "";
  })();

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="min-h-screen w-full text-[color:var(--color-ink)]">
      <audio
        ref={audioRef}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={onEnded}
        onTimeUpdate={onTimeUpdate}
        onLoadedMetadata={onLoadedMetadata}
        preload="auto"
      />

      <Header
        onReset={() => { resetState(); setStage("idle"); }}
        canReset={stage !== "idle"}
      />

      <main className="mx-auto max-w-5xl px-6 pb-24">
        {stage === "idle" && (
          <Hero
            isDragging={isDragging}
            setIsDragging={setIsDragging}
            onDrop={onDrop}
            onPick={() => inputRef.current?.click()}
            expertise={expertise}
            setExpertise={setExpertise}
          />
        )}

        {(stage === "extracting" ||
          stage === "scripting" ||
          stage === "synthesizing") && (
          <Processing
            stage={stage}
            fileName={paperTitle || fileName}
            progressPct={progressPct}
            subStageText={subStageText}
          />
        )}

        {stage === "ready" && (
          <Player
            clips={clips as Clip[]}
            qaByMainIdx={qaByMainIdx}
            currentIndex={currentIndex}
            qaActive={qaActive}
            qaTurnIdx={qaTurnIdx}
            currentClip={currentClip}
            isPlaying={isPlaying}
            awaitingNext={awaitingNext}
            currentTime={currentTime}
            duration={duration}
            clipProgressPct={clipProgressPct}
            canvasRef={canvasRef}
            togglePlay={togglePlay}
            skipNext={skipNext}
            skipPrev={skipPrev}
            seek={seek}
            canDownload={canDownload}
            downloadEpisode={downloadEpisode}
            onJump={(i) => {
              if (qaActive) return;
              shouldAutoPlayRef.current = isPlaying;
              pendingMainSeekRef.current = null;
              setQaTurnIdx(0);
              setCurrentIndex(i);
            }}
            onAsk={openAsk}
            canAsk={!!paperText}
            paperTitle={paperTitle}
          />
        )}

        {stage === "error" && (
          <ErrorCard
            message={error ?? "Unknown error"}
            onRetry={() => { resetState(); setStage("idle"); }}
          />
        )}

        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          className="hidden"
          onChange={(e) => {
            onFiles(e.target.files);
            e.currentTarget.value = "";
          }}
        />
      </main>

      {askOpen && (
        <AskModal
          inputRef={askInputRef}
          value={askText}
          onChange={setAskText}
          onCancel={closeAsk}
          onSubmit={submitAsk}
          loading={askLoading}
          error={askError}
          currentLine={currentClip}
        />
      )}

      <Footer />
    </div>
  );
}

// -----------------------------------------------------------------------------
// Subcomponents
// -----------------------------------------------------------------------------

function Header({ onReset, canReset }: { onReset: () => void; canReset: boolean }) {
  return (
    <header className="mx-auto flex max-w-5xl items-center justify-between px-6 pt-8">
      <button
        onClick={onReset}
        className="group flex items-center gap-2 text-sm tracking-wide text-[color:var(--color-ink-dim)] transition hover:text-[color:var(--color-ink)]"
      >
        <span className="inline-block h-2 w-2 rounded-full bg-[color:var(--color-accent)] shadow-[0_0_12px_rgba(179,164,255,0.8)]" />
        <span className="font-semibold text-[color:var(--color-ink)]">Paper2Pod</span>
      </button>
      <div className="flex items-center gap-4 text-xs text-[color:var(--color-ink-dim)]">
        {canReset && (
          <button
            onClick={onReset}
            className="rounded-full border border-white/10 px-3 py-1.5 transition hover:border-white/25 hover:text-white"
          >
            New paper
          </button>
        )}
      </div>
    </header>
  );
}

function Footer() {
  return (
    <footer className="mx-auto max-w-5xl px-6 py-10 text-center text-xs text-[color:var(--color-ink-dim)]">
      <p>Built with Gemini + ElevenLabs · <span className="text-white/50">press A during playback to ask Alex anything</span></p>
    </footer>
  );
}

function Hero({
  isDragging,
  setIsDragging,
  onDrop,
  onPick,
  expertise,
  setExpertise,
}: {
  isDragging: boolean;
  setIsDragging: (v: boolean) => void;
  onDrop: (e: DragEvent<HTMLDivElement>) => void;
  onPick: () => void;
  expertise: Expertise;
  setExpertise: (v: Expertise) => void;
}) {
  return (
    <section className="pt-16 sm:pt-24">
      <div className="mx-auto max-w-3xl text-center">
        <p className="mb-6 text-[11px] uppercase tracking-[0.32em] text-[color:var(--color-ink-dim)]">
          Papers you meant to read
        </p>
        <h1 className="text-5xl font-semibold leading-[1.05] tracking-tight sm:text-7xl">
          Turn a paper into a
          <br />
          <span className="bg-gradient-to-r from-[color:var(--color-accent)] via-[#d9cffd] to-[color:var(--color-accent-2)] bg-clip-text text-transparent">
            conversation
          </span>
          .
        </h1>
        <p className="mx-auto mt-6 max-w-xl text-base font-light leading-relaxed text-[color:var(--color-ink-dim)] sm:text-lg">
          Drop any academic PDF. Two AI hosts will walk you through the argument:
          what it claims, how it proves it, and why it matters. Pause any time and
          ask them a follow-up.
        </p>
      </div>

      <div className="mx-auto mt-10 flex max-w-2xl flex-col items-center gap-3">
        <p className="text-[10px] uppercase tracking-[0.3em] text-[color:var(--color-ink-dim)]">
          Pitch level
        </p>
        <div className="flex w-full flex-wrap justify-center gap-2">
          {EXPERTISE_OPTIONS.map((opt) => {
            const active = opt.value === expertise;
            return (
              <button
                key={opt.value}
                onClick={() => setExpertise(opt.value)}
                className={[
                  "group flex flex-col items-start rounded-2xl border px-4 py-3 text-left transition",
                  active
                    ? "border-white/25 bg-white/[0.06] text-white"
                    : "border-white/10 text-[color:var(--color-ink-dim)] hover:border-white/20 hover:text-white",
                ].join(" ")}
              >
                <span className="text-sm font-medium">{opt.label}</span>
                <span className="text-[11px] text-white/45">{opt.blurb}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        onClick={onPick}
        role="button"
        tabIndex={0}
        onKeyDown={(e: KeyboardEvent<HTMLDivElement>) => e.key === "Enter" && onPick()}
        className={[
          "group relative mx-auto mt-10 flex min-h-[300px] cursor-pointer flex-col items-center justify-center overflow-hidden rounded-[28px] border transition-all",
          "border-white/10 bg-[color:var(--color-surface-1)]/60 backdrop-blur-xl",
          "hover:border-white/25 hover:bg-[color:var(--color-surface-1)]/80",
          isDragging ? "border-[color:var(--color-accent)]/70 bg-[color:var(--color-surface-2)]" : "",
        ].join(" ")}
      >
        <div
          aria-hidden
          className={[
            "pointer-events-none absolute inset-0 opacity-70 transition-opacity duration-500",
            isDragging ? "opacity-100" : "group-hover:opacity-90",
          ].join(" ")}
          style={{
            background:
              "radial-gradient(420px 260px at 50% 40%, rgba(179,164,255,0.18), transparent 70%)",
          }}
        />
        <div className="relative z-10 flex flex-col items-center gap-4 px-6 text-center">
          <UploadGlyph dragging={isDragging} />
          <p className="mt-2 text-xl font-medium tracking-tight">
            {isDragging ? "Release to begin" : "Drop a PDF"}
          </p>
          <p className="text-sm text-[color:var(--color-ink-dim)]">
            or <span className="underline decoration-white/25 underline-offset-4">click to browse</span>
            <span className="mx-2 text-white/20">·</span>
            works best with papers under 40 pages
          </p>
        </div>
      </div>
    </section>
  );
}

function UploadGlyph({ dragging }: { dragging: boolean }) {
  return (
    <div className="relative h-20 w-20">
      <div className="absolute inset-0 rounded-full bg-[radial-gradient(closest-side,rgba(179,164,255,0.6),rgba(179,164,255,0)_70%)] animate-breathe" />
      <div className="absolute inset-0 flex items-center justify-center">
        <svg
          viewBox="0 0 24 24"
          className={[
            "h-8 w-8 transition-transform",
            dragging ? "-translate-y-1 text-white" : "text-[color:var(--color-ink)]/85",
          ].join(" ")}
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M12 16V4" />
          <path d="M7 9l5-5 5 5" />
          <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
        </svg>
      </div>
    </div>
  );
}

function Processing({
  stage,
  fileName,
  progressPct,
  subStageText,
}: {
  stage: Exclude<Stage, "idle" | "ready" | "error">;
  fileName: string;
  progressPct: number;
  subStageText: string;
}) {
  const stages: Array<[Stage, string]> = [
    ["extracting", "Reading"],
    ["scripting", "Writing"],
    ["synthesizing", "Recording"],
  ];
  const activeIdx = stages.findIndex(([s]) => s === stage);
  return (
    <section className="mx-auto max-w-2xl pt-24 text-center">
      <div className="relative mx-auto mb-10 h-28 w-28">
        <div aria-hidden className="absolute inset-0 rounded-full bg-[radial-gradient(closest-side,rgba(179,164,255,0.5),rgba(179,164,255,0)_70%)] animate-breathe" />
        <div className="absolute inset-[22%] rounded-full bg-[radial-gradient(closest-side,rgba(255,255,255,0.85),rgba(255,255,255,0)_70%)]" />
      </div>
      <p className="text-[11px] uppercase tracking-[0.32em] text-[color:var(--color-ink-dim)]">
        {fileName || "paper.pdf"}
      </p>
      <h2 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl">
        {STAGE_COPY[stage]}<span className="animate-softblink">.</span>
      </h2>
      <p className="mx-auto mt-4 max-w-md text-sm font-light text-[color:var(--color-ink-dim)]">
        {subStageText}
      </p>
      <div className="mx-auto mt-12 h-[3px] w-full max-w-md overflow-hidden rounded-full bg-white/5">
        <div
          className="h-full rounded-full bg-gradient-to-r from-[color:var(--color-accent)] to-[color:var(--color-accent-2)] transition-[width] duration-500"
          style={{
            width:
              stage === "synthesizing" ? `${Math.max(5, progressPct * 100)}%`
                : stage === "scripting" ? "33%"
                : "8%",
          }}
        />
      </div>
      <ol className="mx-auto mt-10 flex max-w-md items-center justify-between gap-3 text-xs uppercase tracking-[0.25em]">
        {stages.map(([s, label], i) => (
          <li
            key={s}
            className={[
              "flex flex-1 items-center justify-center rounded-full border px-3 py-2 transition",
              i < activeIdx ? "border-white/15 text-white/70"
                : i === activeIdx ? "border-[color:var(--color-accent)]/60 text-white"
                : "border-white/5 text-white/30",
            ].join(" ")}
          >
            {label}
          </li>
        ))}
      </ol>
    </section>
  );
}

function Player({
  clips,
  qaByMainIdx,
  currentIndex,
  qaActive,
  qaTurnIdx,
  currentClip,
  isPlaying,
  awaitingNext,
  currentTime,
  duration,
  clipProgressPct,
  canvasRef,
  togglePlay,
  skipNext,
  skipPrev,
  seek,
  canDownload,
  downloadEpisode,
  onJump,
  onAsk,
  canAsk,
  paperTitle,
}: {
  clips: Clip[];
  qaByMainIdx: Record<number, Clip[]>;
  currentIndex: number;
  qaActive: QA | null;
  qaTurnIdx: number;
  currentClip: Clip | null;
  isPlaying: boolean;
  awaitingNext: boolean;
  currentTime: number;
  duration: number;
  clipProgressPct: number;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  togglePlay: () => void;
  skipNext: () => void;
  skipPrev: () => void;
  seek: (pct: number) => void;
  canDownload: boolean;
  downloadEpisode: () => void;
  onJump: (i: number) => void;
  onAsk: () => void;
  canAsk: boolean;
  paperTitle: string;
}) {
  return (
    <section className="pt-8">
      {paperTitle && (
        <div className="mx-auto mb-6 max-w-3xl text-center">
          <p className="text-[10px] uppercase tracking-[0.32em] text-[color:var(--color-ink-dim)]">
            Now playing
          </p>
          <h2 className="mt-1 line-clamp-2 text-2xl font-semibold tracking-tight sm:text-3xl">
            {paperTitle}
          </h2>
        </div>
      )}

      <div className="grid gap-10 lg:grid-cols-[1.05fr_0.95fr]">
        <div className="relative flex flex-col items-center justify-center rounded-[28px] border border-white/10 bg-[color:var(--color-surface-1)]/60 px-6 py-12 backdrop-blur-xl">
          {qaActive && (
            <div className="absolute left-4 top-4 rounded-full border border-white/15 bg-white/[0.04] px-3 py-1 text-[10px] uppercase tracking-[0.26em] text-white/70">
              Listener question
            </div>
          )}

          <div className="relative h-[340px] w-full">
            <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <p className="text-[11px] uppercase tracking-[0.32em] text-[color:var(--color-ink-dim)]">
                Now speaking
              </p>
              <p
                className={[
                  "mt-2 text-5xl font-semibold tracking-tight transition-colors duration-500",
                  currentClip?.speaker === "Alex"
                    ? "bg-gradient-to-r from-[#c7bdff] to-[#e8e1ff] bg-clip-text text-transparent"
                    : "bg-gradient-to-r from-[#9fdcff] to-[#d7efff] bg-clip-text text-transparent",
                ].join(" ")}
              >
                {currentClip?.speaker ?? "..."}
              </p>
              {awaitingNext && (
                <p className="mt-3 text-xs text-[color:var(--color-ink-dim)]">
                  Waiting for the next line to finish rendering…
                </p>
              )}
              {currentClip && (
                <p className="mt-5 max-w-sm px-6 text-center text-sm leading-relaxed text-white/75">
                  {currentClip.text}
                </p>
              )}
            </div>
          </div>

          <div className="mt-6 w-full">
            <Scrubber pct={clipProgressPct} currentTime={currentTime} duration={duration} onSeek={seek} />

            <div className="mt-5 flex items-center justify-center gap-6">
              <IconButton label="Previous" onClick={skipPrev} disabled={!!qaActive}>
                <path d="M19 20L9 12l10-8v16z" />
                <path d="M5 19V5" />
              </IconButton>
              <button
                onClick={togglePlay}
                className="flex h-14 w-14 items-center justify-center rounded-full bg-white text-black transition hover:scale-[1.04] active:scale-95"
                aria-label={isPlaying ? "Pause" : "Play"}
              >
                {isPlaying ? (
                  <svg viewBox="0 0 24 24" className="h-6 w-6" fill="currentColor">
                    <rect x="6" y="5" width="4" height="14" rx="1" />
                    <rect x="14" y="5" width="4" height="14" rx="1" />
                  </svg>
                ) : (
                  <svg viewBox="0 0 24 24" className="h-6 w-6" fill="currentColor">
                    <path d="M7 5l12 7-12 7V5z" />
                  </svg>
                )}
              </button>
              <IconButton label="Next" onClick={skipNext} disabled={!!qaActive}>
                <path d="M5 4l10 8-10 8V4z" />
                <path d="M19 5v14" />
              </IconButton>
            </div>

            <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
              <button
                onClick={onAsk}
                disabled={!canAsk || !!qaActive}
                className={[
                  "group flex items-center gap-2 rounded-full px-4 py-2 text-xs uppercase tracking-[0.22em] transition",
                  "border",
                  !canAsk || qaActive
                    ? "cursor-not-allowed border-white/5 text-white/25"
                    : "border-[color:var(--color-accent)]/60 bg-[color:var(--color-accent)]/[0.08] text-white hover:bg-[color:var(--color-accent)]/[0.16]",
                ].join(" ")}
              >
                <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.6">
                  <path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z" />
                </svg>
                Ask Alex
                <span className="ml-1 rounded bg-white/10 px-1.5 py-0.5 text-[9px] font-mono tracking-normal text-white/60 group-hover:text-white/80">
                  A
                </span>
              </button>
              {canDownload && (
                <button
                  onClick={downloadEpisode}
                  className="flex items-center gap-2 rounded-full border border-white/10 px-4 py-2 text-xs uppercase tracking-[0.24em] text-white/70 transition hover:border-white/25 hover:text-white"
                >
                  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M12 4v12" />
                    <path d="M7 11l5 5 5-5" />
                    <path d="M5 20h14" />
                  </svg>
                  Download .mp3
                </button>
              )}
            </div>
          </div>
        </div>

        <div className="rounded-[28px] border border-white/10 bg-[color:var(--color-surface-1)]/40 px-6 py-10 backdrop-blur-xl">
          <p className="mb-6 text-[11px] uppercase tracking-[0.32em] text-[color:var(--color-ink-dim)]">
            Transcript · {clips.length} lines
          </p>
          <ol className="mask-y max-h-[460px] space-y-3 overflow-auto pr-2">
            {clips.map((c, i) => {
              const active = !qaActive && i === currentIndex;
              const qa = qaByMainIdx[i];
              return (
                <li key={i} className="space-y-3">
                  <button
                    onClick={() => onJump(i)}
                    className={[
                      "w-full cursor-pointer rounded-2xl border px-4 py-3 text-left transition",
                      active
                        ? "border-white/20 bg-white/[0.04]"
                        : "border-transparent opacity-60 hover:opacity-100",
                    ].join(" ")}
                  >
                    <div className="mb-1 flex items-center justify-between">
                      <span
                        className={[
                          "text-[10px] uppercase tracking-[0.26em]",
                          c.speaker === "Alex" ? "text-[#c7bdff]" : "text-[#9fdcff]",
                        ].join(" ")}
                      >
                        {c.speaker}
                      </span>
                      <span className="text-[10px] text-white/30">
                        {String(i + 1).padStart(2, "0")}
                      </span>
                    </div>
                    <p className={["text-[15px] leading-relaxed", active ? "text-white" : "text-white/75"].join(" ")}>
                      {c.text}
                    </p>
                  </button>
                  {qa && qa.map((q, qi) => {
                    const isActiveQA = qaActive?.mainIdx === i && qaTurnIdx === qi;
                    return (
                      <div
                        key={`${i}-qa-${qi}`}
                        className={[
                          "ml-6 rounded-2xl border border-[color:var(--color-accent)]/25 bg-[color:var(--color-accent)]/[0.04] px-4 py-3",
                          isActiveQA ? "ring-1 ring-[color:var(--color-accent)]/60" : "",
                        ].join(" ")}
                      >
                        <div className="mb-1 flex items-center gap-2">
                          <span className="text-[9px] uppercase tracking-[0.28em] text-[color:var(--color-accent)]">
                            Listener Q
                          </span>
                          <span
                            className={[
                              "text-[10px] uppercase tracking-[0.26em]",
                              q.speaker === "Alex" ? "text-[#c7bdff]" : "text-[#9fdcff]",
                            ].join(" ")}
                          >
                            {q.speaker}
                          </span>
                        </div>
                        <p className="text-[14px] leading-relaxed text-white/85">{q.text}</p>
                      </div>
                    );
                  })}
                </li>
              );
            })}
          </ol>
        </div>
      </div>
    </section>
  );
}

function Scrubber({
  pct, currentTime, duration, onSeek,
}: {
  pct: number; currentTime: number; duration: number; onSeek: (pct: number) => void;
}) {
  const trackRef = useRef<HTMLDivElement>(null);
  const handleClick = (e: React.MouseEvent) => {
    const el = trackRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    onSeek(x / rect.width);
  };
  return (
    <div>
      <div ref={trackRef} onClick={handleClick} className="relative h-1 w-full cursor-pointer rounded-full bg-white/10">
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-[color:var(--color-accent)] to-[color:var(--color-accent-2)]"
          style={{ width: `${Math.max(0, Math.min(1, pct)) * 100}%` }}
        />
        <div
          className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow-[0_0_0_4px_rgba(255,255,255,0.08)]"
          style={{ left: `${Math.max(0, Math.min(1, pct)) * 100}%` }}
        />
      </div>
      <div className="mt-2 flex justify-between font-mono text-[11px] tabular-nums text-white/40">
        <span>{formatTime(currentTime)}</span>
        <span>{formatTime(duration)}</span>
      </div>
    </div>
  );
}

function IconButton({
  children, label, onClick, disabled,
}: {
  children: React.ReactNode; label: string; onClick: () => void; disabled?: boolean;
}) {
  return (
    <button
      aria-label={label}
      onClick={onClick}
      disabled={disabled}
      className={[
        "flex h-10 w-10 items-center justify-center rounded-full border transition",
        disabled
          ? "border-white/5 text-white/25"
          : "border-white/10 text-white/70 hover:border-white/25 hover:text-white",
      ].join(" ")}
    >
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        {children}
      </svg>
    </button>
  );
}

function AskModal({
  inputRef, value, onChange, onCancel, onSubmit, loading, error, currentLine,
}: {
  inputRef: React.RefObject<HTMLTextAreaElement>;
  value: string;
  onChange: (v: string) => void;
  onCancel: () => void;
  onSubmit: () => void;
  loading: boolean;
  error: string | null;
  currentLine: Clip | null;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-md" onClick={onCancel} />
      <div className="relative w-full max-w-xl rounded-3xl border border-white/10 bg-[color:var(--color-surface-1)]/95 px-8 py-8 shadow-2xl">
        <p className="text-[11px] uppercase tracking-[0.32em] text-[color:var(--color-accent)]">Ask Alex</p>
        <h3 className="mt-2 text-2xl font-semibold tracking-tight">What do you want to know?</h3>
        {currentLine && (
          <p className="mt-3 text-xs text-[color:var(--color-ink-dim)]">
            Paused right as <span className="text-white">{currentLine.speaker}</span> said: <span className="italic text-white/70">"{currentLine.text.slice(0, 120)}{currentLine.text.length > 120 ? "…" : ""}"</span>
          </p>
        )}
        <textarea
          ref={inputRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              onSubmit();
            }
            if (e.key === "Escape") onCancel();
          }}
          placeholder="e.g. What dataset did they evaluate on? Or, is this actually new?"
          rows={3}
          className="mt-5 w-full resize-none rounded-2xl border border-white/10 bg-black/30 px-4 py-3 text-sm leading-relaxed text-white placeholder:text-white/30 focus:border-[color:var(--color-accent)]/60 focus:outline-none"
        />
        {error && (
          <p className="mt-2 text-xs text-red-300/80">{error}</p>
        )}
        <div className="mt-5 flex items-center justify-between">
          <p className="text-[11px] text-white/35">
            <kbd className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-[10px]">⌘</kbd>
            <span className="mx-1">+</span>
            <kbd className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-[10px]">Enter</kbd>
            <span className="ml-2">to send</span>
          </p>
          <div className="flex gap-2">
            <button
              onClick={onCancel}
              className="rounded-full border border-white/10 px-4 py-2 text-xs uppercase tracking-[0.22em] text-white/70 hover:border-white/25 hover:text-white"
            >
              Cancel
            </button>
            <button
              onClick={onSubmit}
              disabled={loading || !value.trim()}
              className={[
                "rounded-full px-5 py-2 text-xs uppercase tracking-[0.22em] transition",
                loading || !value.trim()
                  ? "cursor-not-allowed bg-white/10 text-white/40"
                  : "bg-white text-black hover:bg-white/90",
              ].join(" ")}
            >
              {loading ? "Thinking…" : "Ask"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function ErrorCard({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <section className="mx-auto max-w-lg pt-24 text-center">
      <h2 className="text-3xl font-semibold tracking-tight">Something went sideways.</h2>
      <p className="mx-auto mt-4 max-w-sm text-sm text-[color:var(--color-ink-dim)]">{message}</p>
      <button
        onClick={onRetry}
        className="mt-8 rounded-full bg-white px-5 py-2.5 text-sm font-medium text-black transition hover:bg-white/90"
      >
        Try another paper
      </button>
    </section>
  );
}
