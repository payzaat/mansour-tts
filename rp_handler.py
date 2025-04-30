"""
RunPod Serverless handler for Fish-Speech.

• Uses checkpoint cloned into /app/checkpoints/fish-speech-1.5
• Starts one server on port 3000, guarded by a semaphore (1 req / GPU).
• Accepts OpenAI-compatible JSON ({ "input": "...", ... }).
• Returns base-64 WAV or streams base-64 OGG chunks.
"""

import os, time, asyncio, base64, subprocess, traceback
import runpod, aiohttp, httpx

# ───────────── Launch Fish-Speech ─────────────
FISH_PORT   = int(os.getenv("FISH_PORT", 3000))
VOICE_DIR   = "/app/voices"
CKPT_DIR    = "/app/checkpoints/fish-speech-1.5"

server = subprocess.Popen([
    "fish-speech",
    "--port", str(FISH_PORT),
    "--voice-dir", VOICE_DIR,
    "--checkpoint", CKPT_DIR
])

# Wait until /health returns 200
for _ in range(60):
    try:
        if httpx.get(f"http://127.0.0.1:{FISH_PORT}/health", timeout=2).status_code == 200:
            break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError("fish-speech failed to start")

FISH_URL = f"http://127.0.0.1:{FISH_PORT}/v1/audio/speech"
GPU_SEMAPHORE = asyncio.Semaphore(1)

# ───────────── Handler ─────────────
async def handler(job):
    inp   = job.get("input", {})
    text  = inp.get("input", "")
    voice = inp.get("voice", "default")
    fmt   = inp.get("response_format", "wav").lower()

    if not text:
        return {"error": "input must be non-empty"}
    if fmt not in ("wav", "ogg"):
        return {"error": "response_format must be 'wav' or 'ogg'"}

    payload = {"input": text, "voice": voice, "response_format": fmt}

    async with GPU_SEMAPHORE:
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(FISH_URL, json=payload) as resp:
                    resp.raise_for_status()

                    if fmt == "wav":
                        wav = await resp.read()
                        return {
                            "audio_format": "wav",
                            "audio_base64": base64.b64encode(wav).decode()
                        }

                    async def ogg_stream():
                        async for chunk in resp.content.iter_chunked(4096):
                            if chunk:
                                yield {"chunk_base64": base64.b64encode(chunk).decode()}
                    return ogg_stream()

        except Exception as e:
            traceback.print_exc()
            return {"error": f"handler exception: {str(e)}"}

# ───────────── Start RunPod loop ─────────────
runpod.serverless.start({"handler": handler})