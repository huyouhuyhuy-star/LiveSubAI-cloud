"""LiveSub AI Cloud demo v0.3 — Android PCM16/WebSocket -> Groq STT -> EN/VI.

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
import time
import wave
from array import array
from collections import deque

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI(title="LiveSub AI Cloud", version="0.3.0")
log = logging.getLogger("uvicorn.error")

RATE = 16_000                   # Android sends 16kHz, signed PCM16, mono
BYTES_PER_SECOND = RATE * 2
SEGMENT_SECONDS = max(4, min(12, int(os.getenv("SEGMENT_SECONDS", "6"))))
SEGMENT_BYTES = BYTES_PER_SECOND * SEGMENT_SECONDS
MAX_SESSION_SECONDS = max(60, min(1800, int(os.getenv("MAX_SESSION_SECONDS", "300"))))
MAX_ACTIVE_SESSIONS = max(1, min(4, int(os.getenv("MAX_ACTIVE_SESSIONS", "1"))))
MAX_TOTAL_AUDIO_SECONDS = max(60, int(os.getenv("MAX_TOTAL_AUDIO_SECONDS", "1800")))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
DEMO_TOKEN = os.getenv("DEMO_TOKEN", "").strip()
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "llama-3.3-70b-versatile")
API_BASE = "https://api.groq.com/openai/v1"

_active_sessions = 0
_total_audio_seconds = 0.0
_slots_lock = asyncio.Lock()


@app.get("/")
def index() -> dict:
    return {"service": "LiveSub AI Cloud v0.3", "status": "online", "health_url": "/health"}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "LiveSub AI Cloud v0.3",
            "groq_configured": bool(GROQ_API_KEY),
            "translation": "English voice -> Vietnamese subtitles",
            "segment_seconds": SEGMENT_SECONDS}


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
    return power ** 0.5 > 170


def clean_source(text: str) -> str:
    return " ".join(text.strip().split())[:750]


async def groq_transcribe(client: httpx.AsyncClient, pcm: bytes) -> str:
    """Only the server has GROQ_API_KEY. Never send this key to Android."""
    response = await client.post(
        f"{API_BASE}/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": ("segment.wav", make_wav(pcm), "audio/wav")},
        data={"model": STT_MODEL, "language": "en", "response_format": "json", "temperature": "0"},
    )
    response.raise_for_status()
    return clean_source(response.json().get("text", ""))


async def groq_translate(client: httpx.AsyncClient, text: str, history: list[str]) -> str:
    history_str = " / ".join(history[-2:])
    response = await client.post(
        f"{API_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": TRANSLATION_MODEL,
            "temperature": 0,
            "max_tokens": 220,
            "messages": [
                {"role": "system", "content": (
                    "You translate English speech into natural, accurate Vietnamese video subtitles. "
                    "Translate only the CURRENT English text. Retain names, numbers and meaning. "
                    "Previous English excerpts are context only, DO NOT translate them again. "
                    "Be concise. Output ONLY Vietnamese subtitle text, no explanation or formatting."
                )},
                {"role": "user", "content": f"Previous excerpts: {history_str or '(none)'}\nCURRENT: {text}"},
            ],
        },
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
            return "Yêu cầu AI không hợp lệ. Kiểm tra model trong Render."
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
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
    previous_sources: deque[str] = deque(maxlen=2)
    previous_text = ""
    configured = False
    buffer = bytearray()
    segments_received = 0
    last_warning = 0.0

    async def send(kind: str, value: str) -> None:
        key = "text" if kind == "subtitle" else "message"
        async with send_lock:
            await ws.send_text(json.dumps({"type": kind, key: value}, ensure_ascii=False))

    async def translate_worker() -> None:
        nonlocal previous_text
        async with httpx.AsyncClient(timeout=45, follow_redirects=False) as client:
            while True:
                chunk = await audio_queue.get()
                try:
                    # Not forwarding silent audio saves requests on the free plan.
                    if not has_voice_energy(chunk):
                        continue
                    speech = await groq_transcribe(client, chunk)
                    if not speech or speech.lower().strip(".,! ?") in {
                        "thank you for watching", "you", "subtitles by the amara.org community"
                    } or speech == previous_text:
                        continue
                    vi = await groq_translate(client, speech, list(previous_sources))
                    previous_sources.append(speech)
                    previous_text = speech
                    if vi:
                        await send("subtitle", vi)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.warning("Translation request failed: %s", type(exc).__name__)
                    try:
                        await send("error", error_message(exc))
                    except Exception:
                        return
                finally:
                    audio_queue.task_done()

    worker = asyncio.create_task(translate_worker())
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
                        await send("error", "Bản demo hiện hỗ trợ tiếng Việt (dịch từ tiếng Anh).")
                    else:
                        configured = True
                        await send("status", "Đã kết nối Cloud AI. Mở video tiếng Anh để thử.")
                continue
            pcm = message.get("bytes")
            if not pcm or not configured:
                continue
            if len(pcm) > 128_000:
                await send("error", "Gói âm thanh gửi lên quá lớn.")
                continue
            buffer.extend(pcm)
            while len(buffer) >= SEGMENT_BYTES:
                chunk = bytes(buffer[:SEGMENT_BYTES]); del buffer[:SEGMENT_BYTES]
                async with _slots_lock:
                    if _total_audio_seconds + SEGMENT_SECONDS > MAX_TOTAL_AUDIO_SECONDS:
                        await send("error", "Đã hết hạn mức âm thanh demo của máy chủ.")
                        return
                    _total_audio_seconds += SEGMENT_SECONDS
                segments_received += 1
                if audio_queue.full():
                    # Drop stale audio rather than progressively lagging behind the video.
                    try:
                        audio_queue.get_nowait()
                        audio_queue.task_done()
                    except asyncio.QueueEmpty:
                        pass
                    if time.monotonic() - last_warning > 20:
                        await send("status", "AI xử lý chưa kịp; đang bỏ đoạn cũ để giảm trễ.")
                        last_warning = time.monotonic()
                audio_queue.put_nowait(chunk)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.info("WebSocket disconnected: %s", type(exc).__name__)
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        async with _slots_lock:
            _active_sessions = max(0, _active_sessions - 1)
        try:
            await ws.close()
        except Exception:
            pass
