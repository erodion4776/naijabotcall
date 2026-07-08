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

SYSTEM_PROMPT = """You are 'Aisha,' a friendly female voice assistant calling
for Naijashop.com.ng - a Nigerian business management app.

YOUR GOAL:
Convince the shop owner to try Naijashop to track their sales and stock
instead of using paper or notebooks.

STRICT RULES:
- Maximum 1 to 2 SHORT sentences per reply
- Be natural and conversational
- Speak like a Nigerian professional
- Never repeat yourself
- Answer exactly what they ask directly
- Do not use big English grammar
- Do not say things like "I am an AI"

CONVERSATION FLOW:
1. Greet and confirm you are speaking with the shop owner
2. Introduce Naijashop briefly
3. Ask how they currently track their stock and sales
4. Explain how Naijashop solves that problem
5. Ask if they want to try it free for 30 days"""

GREETING = "Good day! Am I speaking with the shop owner?"

@app.get("/")
async def root():
    return {"status": "✅ Naijashop Aisha Bot is running!"}

@app.post("/voice")
async def voice(request: Request):
    host = request.headers.get("host")
    res = VoiceResponse()
    connect = Connect()
    connect.stream(url=f"wss://{host}/ws")
    res.append(connect)
    return HTMLResponse(content=str(res), media_type="application/xml")

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
        return {"status": "❌ Error", "detail": str(e)}

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
        print(f"⚠️ Could not get stream_sid: {e}")

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_out_enabled=True,
            audio_out_sample_rate=8000,
            audio_out_channels=1,
            audio_out_10ms_chunks=2,
            audio_in_enabled=True,
            audio_in_sample_rate=8000,
            audio_in_channels=1,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.5,
                    start_secs=0.2,
                    confidence=0.5,
                    min_volume=0.4,
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

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model="nova-2",
            language="en",
            smart_format=True,
            endpointing=300,
            utterance_end_ms=1000,
            interim_results=True,
            punctuate=True,
        )
    )

    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="f9836c6e-a0bd-460e-9d3c-f7299fa60f94",
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
        print("✅ Client connected - sending greeting")
        await task.queue_frames([TTSSpeakFrame(text=GREETING)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        print("📵 Client disconnected")
        await task.cancel()

    runner = PipelineRunner()
    await runner.run(task)
