"""
RunPod Serverless handler for Fish-Speech (lazy checkpoint download).

• Downloads all required Fish-Speech-1.5 files into /tmp on first boot
  (~1 GB, ~50 s). Subsequent cold starts reuse RunPod’s cached layer.
• Starts one Fish-Speech server on port 3000.
• Accepts OpenAI-compatible JSON ("input", "voice", "response_format").
• Returns WAV (base-64) or streams OGG chunks (base-64).
"""

import os, time, asyncio, base64, subprocess, traceback, shutil, urllib.request, pathlib, sys
import runpod, aiohttp, httpx

# ────────── 0. Lazy-download checkpoint ──────────
CKPT_DIR = "/tmp/fish-speech-1.5"
FILES = {
    "model.safetensors":
        "https://huggingface.co/fishaudio/fish-speech-1.5/resolve/main/fish-speech-1.5.safetensors",
    "config.json":
        "https://huggingface.co/fishaudio/fish-speech-1.5/resolve/main/config.json",
    "tokenizer.json":
        "https://huggingface.co/fishaudio/fish-speech-1.5/resolve/main/tokenizer.json",
    "special_tokens_map.json":
        "https://huggingface.co/fishaudio/fish-speech-1.5/resolve/main/special_tokens_map.json",
    "dual_ar.bin":
        "https://huggingface.co/fishaudio/fish-speech-1.5/resolve/main/dual_ar.bin"
}
pathlib.Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
for name, url in FILES.items():
    dest = f"{CKPT_DIR}/{name}"
    if not os.path.exists(dest):
        print(f"[handler] Downloading {name} …", file=sys.stderr, flush=True)
        with urllib.request.urlopen(url) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)

# ────────── 1. Launch Fish-Speech server ──────────
FISH_PORT = int(os.getenv("FISH_PORT", 3000))
VOICE_DIR = "/app/voices"

server = subprocess.Popen([
    "fish-speech",
    "--port", str(FISH_PORT),
    "--voice-dir", VOICE_DIR,
    "--checkpoint", CKPT_DIR
])

# wait for /health
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

# ────────── 2. Handler ──────────
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

# ────────── 3. Start RunPod loop ──────────
runpod.serverless.start({"handler": handler})