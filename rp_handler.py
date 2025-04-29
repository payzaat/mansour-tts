"""
RunPod Serverless handler for Fish-Speech
──────────────────────────────────────────
• Spins up the Rust TTS server once per worker (default port 3000).  
• Guarantees single-request-per-GPU with an asyncio.Semaphore.  
• Accepts **OpenAI-compatible** jobs:
      { "input": { "input": "<text>", "voice": "...", "response_format": "wav|ogg", "model": "tts-1" } }
  – `model` is accepted but **ignored** when forwarding to Fish-Speech.  
• Returns one base-64 WAV blob **or** streams base-64 OGG chunks.
"""

import os, time, asyncio, base64, subprocess, traceback
import runpod, aiohttp, httpx

# ─────────────────── Launch the Rust server ───────────────────
FISH_PORT = int(os.getenv("FISH_PORT", 3000))
VOICE_DIR = os.getenv("VOICE_DIR", "/app/voices")

server = subprocess.Popen([
    "fish-speech", "--port", str(FISH_PORT), "--voice-dir", VOICE_DIR
])

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

# ─────────────────────── Handler function ─────────────────────
async def handler(job):
    inp   = job.get("input", {})
    text  = inp.get("input", "")          
    voice = inp.get("voice", "default")
    fmt   = inp.get("response_format", "wav").lower()
    _     = inp.get("model")               

    if not text:
        return {"error": "input must be non-empty"}
    if fmt not in ("wav", "ogg"):
        return {"error": "response_format must be 'wav' or 'ogg'"}

    payload = {                           
        "input": text,
        "voice": voice,
        "response_format": fmt
    }

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

                    # --- stream OGG ---
                    async def ogg_stream():
                        async for chunk in resp.content.iter_chunked(4096):
                            if chunk:
                                yield {"chunk_base64": base64.b64encode(chunk).decode()}
                    return ogg_stream()

        except Exception as e:
            traceback.print_exc()
            return {"error": f"handler exception: {str(e)}"}

# ─────────────────────── Start RunPod loop ────────────────────
runpod.serverless.start({"handler": handler})