# Qwen3-1.7B Hosting: Options and Recommendation

## Model
- Qwen3-1.7B
- Open source, Apache 2.0 license
- Self-hostable, runs locally on modest hardware

## Goal
Move model from local machine to a "virtual" (cloud-accessible) setup, exposed as an API for use in a project.

## Options Considered

### 1. Fully Local
- Zero cost
- Only reachable on same machine/network
- Not viable for shared/production use

### 2. Local + Tunnel (ngrok / Cloudflare Tunnel)
- Free, gives local model a public URL
- Machine must stay on 24/7
- No SLA, breaks on reboot/ISP drop
- OK for demos or low-stakes internal use only

### 3. Cloud GPU (Rented)
- RunPod, Vast.ai, etc.
- Reliable uptime, scales with traffic
- Costs money (per-hour or per-second billing)
- Best fit for production

### 4. Free-Tier Cloud (Google Colab, Kaggle)
- Free GPU
- Sessions expire, no persistent public URL
- Testing/dev only, not production

### 5. On-Device / Edge
- Quantized model runs directly on user's device (llama.cpp, Ollama)
- No server cost
- Only works if project is a local app, not a shared backend
- Depends on user's hardware

### 6. Self-Owned Physical Server
- One-time hardware cost instead of recurring cloud bill
- Cost-effective only at high, sustained, long-term usage
- Upfront cost + maintenance burden on you

## Cost Comparison (Cloud GPU)

| Option | Cost | Best For |
|---|---|---|
| RunPod Serverless | ~$1-5/month (spiky/low traffic) | Unpredictable, low-volume traffic |
| Vast.ai spot GPU (24/7) | ~$70-110/month | Continuous, steady traffic |
| RunPod On-Demand (paused when idle) | ~$50-100/month | Bursty, work-hours-only usage |
| Google Colab | Free | Testing only, not production |

## Recommendation for Production

**RunPod Serverless + vLLM**

- Pay only for actual inference time, scales to zero when idle
- vLLM serves the model with an OpenAI-compatible API endpoint (`/v1/chat/completions`)
- Project calls this endpoint like any OpenAI API call
- Tradeoff: 5-10s cold start after idle periods (acceptable unless sub-second first response is required)
- Switch to dedicated always-on GPU only once traffic becomes steady and high enough that 24/7 cost beats per-request billing

## Non-Cloud Fallbacks (if budget is zero)
- Local + tunnel: acceptable for low-stakes/internal use
- On-device/edge: acceptable if project is a user-run app, not a shared backend
- Neither matches cloud reliability for a shared, always-available production API