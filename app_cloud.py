"""LiveSub AI Cloud demo v0.9 — Android PCM16/WebSocket -> Groq STT -> EN/JA/ZH to Vietnamese.

Designed for a *limited, opt-in demo* on a free Render Web Service and
Groq's rate-limited free API tier. Original Android WebSocket protocol unchanged.
No audio is deliberately persisted. This is not a production multi-user backend.
"""
from __future__ import annotations

import asyncio
import hmac
import io
import json
import logging
import os
import re
import time
import wave
from array import array
from collections import deque

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI(title="LiveSub AI Cloud", version="0.9.0")
log = logging.getLogger("uvicorn.error")

RATE = 16_000                   # Android sends 16kHz, signed PCM16, mono
BYTES_PER_SECOND = RATE * 2
# Four seconds between STT requests = at most 15 requests/minute, below the
# common 20 RPM Whisper free-tier limit before accounting for other sessions.
SEGMENT_SECONDS = max(4, min(12, int(os.getenv("SEGMENT_SECONDS", "4"))))
SEGMENT_BYTES = BYTES_PER_SECOND * SEGMENT_SECONDS
# Append the end of the preceding chunk to the next chunk, without increasing
# the number of STT calls. A 4s cadence with 1s overlap -> 4s, 5s, 5s WAVs.
AUDIO_OVERLAP_SECONDS = max(0, min(2, int(os.getenv("AUDIO_OVERLAP_SECONDS", "1"))))
OVERLAP_BYTES = BYTES_PER_SECOND * AUDIO_OVERLAP_SECONDS
# Bounded backpressure; do not silently drop old audio when inference is slow.
MAX_AUDIO_QUEUE = max(1, min(6, int(os.getenv("MAX_AUDIO_QUEUE", "3"))))
MAX_TRANSCRIPT_QUEUE = max(1, min(8, int(os.getenv("MAX_TRANSCRIPT_QUEUE", "4"))))
MAX_SESSION_SECONDS = max(60, min(1800, int(os.getenv("MAX_SESSION_SECONDS", "300"))))
MAX_ACTIVE_SESSIONS = max(1, min(4, int(os.getenv("MAX_ACTIVE_SESSIONS", "1"))))
MAX_TOTAL_AUDIO_SECONDS = max(60, int(os.getenv("MAX_TOTAL_AUDIO_SECONDS", "1800")))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
DEMO_TOKEN = os.getenv("DEMO_TOKEN", "").strip()
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "openai/gpt-oss-120b")
API_BASE = "https://api.groq.com/openai/v1"

_active_sessions = 0
_total_audio_seconds = 0.0
_slots_lock = asyncio.Lock()


@app.get("/")
def index() -> dict:
    return {"service": "LiveSub AI Cloud v0.9", "status": "online", "health_url": "/health"}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "LiveSub AI Cloud v0.9",
            "groq_configured": bool(GROQ_API_KEY),
            "translation": "English/Japanese/Chinese speech -> Vietnamese subtitles",
            "source_languages": ["en", "ja", "zh"],
            "segment_seconds": SEGMENT_SECONDS,
            "audio_overlap_seconds": AUDIO_OVERLAP_SECONDS,
            "pipeline": "capture -> transcribe -> translate"}


def make_wav(pcm: bytes) -> bytes:
    """Wrap the raw Android PCM frames into a valid RIFF/WAVE stream."""
    output = io.BytesIO()
    with wave.open(output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return output.getvalue()


def has_voice_energy(pcm: bytes) -> bool:
    """Cheap silence check to avoid pointless cloud API calls."""
    if len(pcm) < BYTES_PER_SECOND or len(pcm) % 2:
        return False
    samples = array("h")
    samples.frombytes(pcm)
    sampled = samples[::32]
    if not sampled:
        return False
    power = sum(float(s) ** 2 for s in sampled) / len(sampled)
    return power ** 0.5 > 130


def clean_source(text: str) -> str:
    return " ".join(text.strip().split())[:750]


def deduplicate_transcript(previous: str, current: str, language: str) -> str:
    """Remove a *clearly repeated* prefix produced by overlapping WAV chunks.

    Conservative by design: keeping an occasional duplicate is better than
    erasing a legitimately repeated word. Punctuation and case are ignored for
    overlap comparisons but the original spelling is preserved in the result.
    """
    current = clean_source(current)
    if not previous or not current:
        return current
    if language == "en":
        pattern = r"[\w]+(?:['’][\w]+)?"
        past = list(re.finditer(pattern, previous, re.UNICODE))
        now = list(re.finditer(pattern, current, re.UNICODE))
        p = [m.group().casefold() for m in past]
        n = [m.group().casefold() for m in now]
        if p == n:
            return ""
        for amount in range(min(len(p), len(n)), 1, -1):
            if p[-amount:] == n[:amount]:
                return current[now[amount - 1].end():].lstrip(" ,.:;!?…—-\t")
        return current

    # Chinese and Japanese do not reliably use spaces between words. Trim only
    # a *long* shared phrase (>=4 visible characters), not common syllables.
    def visible_chars(value: str):
        return [(m.group(), m.end()) for m in re.finditer(r"[^\W_]", value, re.UNICODE)]

    past = visible_chars(previous)
    now = visible_chars(current)
    p = [c for c, _ in past]
    n = [c for c, _ in now]
    if p == n:
        return ""
    for amount in range(min(len(p), len(n)), 3, -1):
        if p[-amount:] == n[:amount]:
            return current[now[amount - 1][1]:].lstrip(" 、，。！？!?,.:;…\t")
    return current


async def groq_transcribe(client: httpx.AsyncClient, pcm: bytes, source_language: str = "en") -> str:
    """Only the server has GROQ_API_KEY. Never send this key to Android."""
    response = await client.post(
        f"{API_BASE}/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": ("segment.wav", make_wav(pcm), "audio/wav")},
        data={"model": STT_MODEL, "language": source_language, "response_format": "json", "temperature": "0"},
    )
    response.raise_for_status()
    return clean_source(response.json().get("text", ""))


async def groq_translate(client: httpx.AsyncClient, text: str, history: list[str], source_language: str = "en") -> str:
    history_str = " / ".join(history[-2:])
    source_name = {"en": "English", "ja": "Japanese", "zh": "Mandarin Chinese"}.get(source_language, "English")
    # GPT-OSS tends to reason before answering. 'low' reduces latency for short captions.
    body = {
        "model": TRANSLATION_MODEL,
        "temperature": 0,
        "max_tokens": 220,
        "messages": [
            {"role": "system", "content": (
                f"You translate {source_name} speech into natural, accurate Vietnamese video subtitles. "
                "Translate only the CURRENT source-language text. Retain names, numbers and meaning. "
                "Previous source-language excerpts are context only, DO NOT translate them again. "
                "Be concise. Preserve normal Vietnamese punctuation. "
                "Place each COMPLETE Vietnamese sentence on its own line; do not use bullets or labels. "
                "Output ONLY Vietnamese subtitle text, no explanation."
            )},
            {"role": "user", "content": f"Previous excerpts: {history_str or '(none)'}\nCURRENT: {text}"},
        ],
    }
    if TRANSLATION_MODEL in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
        body["reasoning_effort"] = "low"
        body["include_reasoning"] = False
    response = await client.post(
        f"{API_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json=body,
    )
    response.raise_for_status()
    data = response.json()
    return (data["choices"][0]["message"].get("content") or "").strip()[:600]


def error_message(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            return "Đã chạm giới hạn Groq miễn phí. Hãy thử lại sau."
        if status in (401, 403):
            return "API key Groq không hợp lệ hoặc chưa được cấp quyền."
        if status == 400:
            return "Yêu cầu AI không hợp lệ. Kiểm tra STT_MODEL và TRANSLATION_MODEL trong Render."
        return f"Dịch vụ AI đang lỗi HTTP {status}."
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError)):
        return "Máy chủ cloud tạm thời không liên lạc được với Groq."
    return "Lỗi AI. Xem Render Logs để biết chi tiết."


@app.websocket("/ws")
async def ws_subtitles(ws: WebSocket) -> None:
    global _active_sessions, _total_audio_seconds
    # The token is optional ONLY to simplify initial setup; set one for demo.
    if DEMO_TOKEN and not hmac.compare_digest(ws.query_params.get("token", ""), DEMO_TOKEN):
        await ws.close(code=1008, reason="Demo token required")
        return

    async with _slots_lock:
        if _active_sessions >= MAX_ACTIVE_SESSIONS:
            await ws.close(code=1013, reason="Demo busy")
            return
        _active_sessions += 1

    await ws.accept()
    send_lock = asyncio.Lock()
    started = time.monotonic()
    # The transcription worker may continue while the previous transcription
    # is being translated. This greatly reduces avoidable sequential latency.
    audio_queue: asyncio.Queue[tuple[bytes, str, int]] = asyncio.Queue(maxsize=MAX_AUDIO_QUEUE)
    transcript_queue: asyncio.Queue[tuple[str, str, int]] = asyncio.Queue(maxsize=MAX_TRANSCRIPT_QUEUE)
    previous_sources: deque[str] = deque(maxlen=2)
    previous_recognized = ""
    source_language = "en"
    config_generation = 0
    configured = False
    buffer = bytearray()
    overlap_tail = b""
    last_warning = 0.0

    async def send(kind: str, value: str, extra: dict | None = None) -> None:
        key = "text" if kind == "subtitle" else "message"
        async with send_lock:
            payload = {"type": kind, key: value}
            if extra:
                payload.update(extra)
            await ws.send_text(json.dumps(payload, ensure_ascii=False))

    async def transcribe_worker(client: httpx.AsyncClient) -> None:
        nonlocal previous_recognized
        while True:
            pcm_chunk, lang, generation = await audio_queue.get()
            try:
                if generation != config_generation or not has_voice_energy(pcm_chunk):
                    continue
                recognized = await groq_transcribe(client, pcm_chunk, lang)
                if generation != config_generation:
                    continue
                if not recognized or recognized.casefold().strip(".,! ?") in {
                    "thank you for watching", "you", "subtitles by the amara.org community"
                }:
                    continue
                fresh = deduplicate_transcript(previous_recognized, recognized, lang)
                previous_recognized = recognized
                if fresh:
                    # Backpressure instead of dropping recognized text.
                    await transcript_queue.put((fresh, lang, generation))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("Transcription failed: %s", type(exc).__name__)
                try:
                    await send("error", "Nhận diện tiếng nói: " + error_message(exc))
                except Exception:
                    return
            finally:
                audio_queue.task_done()

    async def translate_worker(client: httpx.AsyncClient) -> None:
        while True:
            speech, lang, generation = await transcript_queue.get()
            try:
                if generation != config_generation:
                    continue
                vi = await groq_translate(client, speech, list(previous_sources), lang)
                if generation != config_generation:
                    continue
                previous_sources.append(speech)
                if vi:
                    await send("subtitle", vi, {"source_text": speech, "source_language": lang})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("Translation failed: %s", type(exc).__name__)
                try:
                    await send("error", "Dịch phụ đề: " + error_message(exc))
                except Exception:
                    return
            finally:
                transcript_queue.task_done()

    client = httpx.AsyncClient(timeout=45, follow_redirects=False)
    workers = [asyncio.create_task(transcribe_worker(client)),
               asyncio.create_task(translate_worker(client))]
    try:
        if not GROQ_API_KEY:
            await send("error", "Máy chủ chưa có GROQ_API_KEY. Hãy đặt Environment Variable trên Render.")
            return
        await send("status", "Đã nối Cloud; chờ cài đặt ngôn ngữ.")
        while True:
            if time.monotonic() - started >= MAX_SESSION_SECONDS:
                await send("error", "Đã hết thời lượng demo. Hãy bấm Dừng và Bắt đầu lại.")
                return
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("text") is not None:
                try:
                    msg = json.loads(message["text"])
                except (TypeError, ValueError):
                    await send("error", "Cấu hình không phải JSON hợp lệ.")
                    continue
                if msg.get("type") == "config":
                    if msg.get("target_language", "vi") != "vi":
                        await send("error", "Bản demo chỉ hỗ trợ ngôn ngữ đích là tiếng Việt.")
                    else:
                        selected = msg.get("source_language", "en")
                        if selected not in ("en", "ja", "zh"):
                            await send("error", "Ngôn ngữ nguồn không hợp lệ (chọn en/ja/zh).")
                            continue
                        source_language = selected
                        config_generation += 1
                        previous_sources.clear()
                        previous_recognized = ""
                        buffer.clear()
                        overlap_tail = b""
                        configured = True
                        label = {"en": "Anh", "ja": "Nhật", "zh": "Trung"}[selected]
                        await send("status", f"Đã kết nối Cloud AI. Mở video tiếng {label} để thử.")
                continue
            pcm = message.get("bytes")
            if not pcm or not configured:
                continue
            if len(pcm) > 128_000:
                await send("error", "Gói âm thanh gửi lên quá lớn.")
                continue
            buffer.extend(pcm)
            while len(buffer) >= SEGMENT_BYTES:
                raw_chunk = bytes(buffer[:SEGMENT_BYTES])
                del buffer[:SEGMENT_BYTES]
                # Exact non-overlapping duration is used for quota accounting.
                # Prepending the previous tail helps Whisper recognize cut words.
                stt_chunk = overlap_tail + raw_chunk
                overlap_tail = raw_chunk[-OVERLAP_BYTES:] if OVERLAP_BYTES else b""
                async with _slots_lock:
                    if _total_audio_seconds + SEGMENT_SECONDS > MAX_TOTAL_AUDIO_SECONDS:
                        await send("error", "Đã hết hạn mức âm thanh demo của máy chủ.")
                        return
                    _total_audio_seconds += SEGMENT_SECONDS
                if audio_queue.full() and time.monotonic() - last_warning > 10:
                    await send("status", "Groq đang xử lý chậm; giữ âm thanh trong hàng đợi để tránh mất từ.")
                    last_warning = time.monotonic()
                # Preserve audio in order. Backpressure may cause delay during
                # overload; cannot promise both zero loss and zero latency.
                await audio_queue.put((stt_chunk, source_language, config_generation))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.info("WebSocket disconnected: %s", type(exc).__name__)
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await client.aclose()
        async with _slots_lock:
            _active_sessions = max(0, _active_sessions - 1)
        try:
            await ws.close()
        except Exception:
            pass
