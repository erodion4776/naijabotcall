# ============================================
# main.py - NAIJASHOP AISHA BOT
# ============================================

import asyncio
import os
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client

from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams
)
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMAssistantAggregator
)
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

app = FastAPI()

PUBLIC_URL = os.getenv("RENDER_EXTERNAL_URL", "")

SYSTEM_PROMPT = """You are Aisha, a sharp Nigerian female sales rep 
calling for Naijashop.com.ng.

BACKGROUND:
Naijashop helps Nigerian shop owners track their 
sales and stock on their phone instead of using 
paper notebooks. First 30 days are completely FREE.

YOUR PERSONALITY:
- You sound like a real Lagos professional woman
- Warm, friendly and persuasive
- Speak simply, no big grammar
- Confident but not pushy
- You understand Nigerian shop problems deeply

STRICT RULES:
- Reply in MAXIMUM 1 short sentence only
- Never say I am an AI or I am a bot
- Never repeat the same thing twice
- Always respond to exactly what they say
- Use Nigerian expressions naturally:
  Ehen, No wahala, E easy, Na so
- If they speak Pidgin reply in Pidgin
- If they speak Yoruba or Igbo greet them 
  in that language then continue in English
- Keep pushing towards your goal

YOUR GOAL:
Get the shop owner to agree to try 
Naijashop FREE for 30 days then 
tell them to visit naijashop.com.ng

CONVERSATION STEPS:
1. Confirm you are speaking with shop owner
2. Ask ONE pain point about their current 
   way of tracking stock and sales
3. Show how Naijashop solves that pain
4. Ask if they want to try FREE for 30 days
5. If yes tell them naijashop.com.ng
6. If no ask what their concern is
7. Handle objection and try to close again

HANDLE OBJECTIONS LIKE THIS:
- No time → E go save you time, e fast to set up
- No data → E work offline, no need data always  
- Not interested → Nothing to lose, na free
- How much → First 30 days free, after that small
- I use notebook → Notebook fit get lost or tear,
                   Naijashop keep everything safe
- I use Google → Naijashop better for shop, 
                 e show your profit automatically
- I am busy → E no go take long, just 2 minutes"""

GREETING = "Hello! Na Aisha from Naijashop. Please, na shop owner I dey speak with?"

@app.get("/")
async def root():
    return {
        "status": "✅ Naijashop Aisha Bot is running!",
        "version": "2.0",
        "voice": "Tyler - Friendly Salesman"
    }

@app.post("/voice")
async def voice(request: Request):
    host = request.headers.get("host")
    res = VoiceResponse()
    connect = Connect()
    connect.stream(url=f"wss://{host}/ws")
    res.append(connect)
    return HTMLResponse(
        content=str(res),
        media_type="application/xml"
    )

@app.get("/make-call")
async def make_call(phone_number: str):
    try:
        client = Client(
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN")
        )
        call = client.calls.create(
            to=phone_number,
            from_=os.getenv("TWILIO_PHONE_NUMBER"),
            url=f"{PUBLIC_URL}/voice"
        )
        return {
            "status": "✅ Calling...",
            "calling": phone_number,
            "call_sid": call.sid
        }
    except Exception as e:
        return {
            "status": "❌ Error",
            "detail": str(e)
        }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    stream_sid = None
    call_sid = None

    try:
        data = await websocket.receive_json()
        if data.get("event") == "connected":
            data = await websocket.receive_json()
        if data.get("event") == "start":
            stream_sid = data["start"]["streamSid"]
            call_sid = data["start"]["callSid"]
            print(f"✅ Stream SID: {stream_sid}")
            print(f"✅ Call SID  : {call_sid}")
    except Exception as e:
        print(f"⚠️ Error: {e}")

    # ✅ TRANSPORT WITH FAST VAD
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_out_enabled=True,
            audio_out_sample_rate=8000,
            audio_out_channels=1,
            audio_out_10ms_chunks=1,
            audio_in_enabled=True,
            audio_in_sample_rate=8000,
            audio_in_channels=1,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.3,
                    start_secs=0.1,
                    confidence=0.4,
                    min_volume=0.3,
                )
            ),
            serializer=TwilioFrameSerializer(
                stream_sid=stream_sid,
                call_sid=call_sid,
                account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
                auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
                params=TwilioFrameSerializer.InputParams(
                    twilio_sample_rate=8000,
                    auto_hang_up=False
                )
            ),
        ),
    )

    # ✅ FAST DEEPGRAM STT
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model="nova-2",
            language="en",
            smart_format=True,
            endpointing=200,
            utterance_end_ms=500,
            interim_results=True,
            punctuate=True,
        )
    )

    # ✅ FASTEST GROQ MODEL
    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant"
    )

    # ✅ TYLER - FRIENDLY SALESMAN VOICE
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="820a3788-2b37-4d21-847a-b65d8a68c99a",
        sample_rate=8000,
        encoding="pcm_s16le",
        container="raw",
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = LLMContext(messages=messages)
    user_aggregator = LLMUserAggregator(context)
    assistant_aggregator = LLMAssistantAggregator(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True)
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        print("✅ Client connected - Aisha is calling!")
        await task.queue_frames([
            TTSSpeakFrame(text=GREETING)
        ])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        print("📵 Call ended")
        await task.cancel()

    runner = PipelineRunner()
    await runner.run(task)
