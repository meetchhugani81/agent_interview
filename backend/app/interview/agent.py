import asyncio
import os
import json
import httpx
import re

from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, llm, AgentStateChangedEvent
from livekit.plugins import silero, groq
from livekit.agents import inference

from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger("interviewer")


def parse_json_object(content: str) -> dict:
    """Parse an object even when a model wraps JSON in a Markdown fence."""
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content.strip(), flags=re.IGNORECASE)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("The model response did not contain a JSON object")
    value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise ValueError("The model response JSON was not an object")
    return value


def safe_score(value) -> int:
    """Keep model-provided scores in the report's expected 0–100 range."""
    try:
        return max(0, min(100, round(float(value))))
    except (TypeError, ValueError):
        return 0


def is_substantive_answer(answer: str) -> bool:
    """Only a genuinely empty STT result is unusable for evaluation.

    Do not perform substring-based refusal detection here: a valid answer can
    contain words such as "passages", which previously matched "pass" and
    bypassed the evaluator despite being visible in the candidate transcript.
    The evaluator receives every non-empty final transcript and determines its
    relevance and quality.
    """
    return bool((answer or "").strip())


def missing_answer_evaluation() -> dict:
    return {
        "score": 0,
        "correctness": "No substantive answer provided",
        "relevance": 0,
        "completeness": 0,
        "technical_depth": 0,
        "feedback": "No substantive answer was provided for this question.",
        "strengths": [],
        "weaknesses": ["Provide a complete, topic-relevant answer instead of skipping the question."],
    }


async def fetch_session(session_id: str):
    logger.info("FETCHING SESSION: %s", session_id)
    async with httpx.AsyncClient() as client:
        r = await client.get(f"http://localhost:8000/api/interview/session/{session_id}")
    logger.info("SESSION RESPONSE STATUS: %s", r.status_code)
    data = r.json()
    if r.status_code != 200:
        raise RuntimeError(
            f"Session {session_id} not found on FastAPI (HTTP {r.status_code}): {data}"
        )
    if "setup" not in data:
        raise RuntimeError(
            f"Session response missing 'setup' key. Got keys: {list(data.keys())}"
        )
    return data


async def complete_session(session_id: str, report: dict):
    async with httpx.AsyncClient() as client:
        await client.post(
            f"http://localhost:8000/api/interview/session/{session_id}/complete",
            json=report,
        )


server = AgentServer()

# Pre-load the VAD model once at module/worker startup.
# silero.VAD.load() is synchronous and loads a PyTorch/ONNX model.
# Calling it inside the async entrypoint (after ctx.connect()) blocks the
# event loop long enough for the LiveKit 10-second RoomIO watchdog to fire
# and the FFI layer to panic with ReadyForRoomEventRequest timeout.
logger.info("[STARTUP] Pre-loading Silero VAD model...")
_preloaded_vad = silero.VAD.load()
logger.info("[STARTUP] Silero VAD model ready")


@server.rtc_session(agent_name="interviewer")
async def entrypoint(ctx: JobContext):
    logger.info("[ROOM] ENTRYPOINT START")
    logger.info("[ROOM] JOB ROOM = %s", ctx.room.name)

    logger.info("[ROOM] CALLING CTX.CONNECT")
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
    logger.info("[ROOM] CTX.CONNECT RETURNED")
    logger.info("[ROOM] ROOM CONNECTED participants=%d", len(ctx.room.remote_participants))

    session_id = ctx.room.name

    # ── Load setup ────────────────────────────────────────────────────────────
    setup = None
    raw_metadata = getattr(ctx.job, "metadata", None)
    if raw_metadata:
        try:
            setup = json.loads(raw_metadata)
            logger.info("SETUP LOADED FROM DISPATCH METADATA: %s", setup)
        except Exception as e:
            logger.warning("Failed to parse dispatch metadata: %s", e)

    if not setup:
        logger.info("No dispatch metadata - falling back to HTTP fetch")
        try:
            session_data = await fetch_session(session_id)
            setup = session_data["setup"]
            logger.info("SETUP LOADED FROM API FALLBACK: %s", setup)
        except Exception as e:
            logger.error(f"Could not fetch session {session_id}: {e}")
            return

    # ── State variables ───────────────────────────────────────────────────────
    #
    # interview_phase is the single source of truth for the state machine.
    #
    # Legal transitions:
    #   "INIT"
    #     └─► generate_interview_question("initial") ─► "GENERATING_QUESTION"
    #           └─► session.say()                   ─► "AI_SPEAKING"
    #                 └─► say() resolves             ─► "WAITING_FOR_ANSWER"
    #                       └─► STOP pressed / STT  ─► "EVALUATING_ANSWER"
    #                             ├─► current_q < 5 ─► "GENERATING_QUESTION" (loop)
    #                             └─► current_q == 5─► "WAITING_FOR_COMPLETION"
    #                                   └─► button   ─► "GENERATING_REPORT"
    #                                         └─►      "COMPLETE"
    # This is supplied by the setup screen and validated by the API.
    MAX_INTERVIEW_QUESTIONS = setup.get("num_questions", 5)
    question_count          = MAX_INTERVIEW_QUESTIONS
    current_q               = 0
    current_question_text   = ""
    interview_results       = []

    # interview_phase — only generate_interview_question and handle_user_speech write this
    interview_phase = "INIT"

    # is_processing_turn — owned exclusively by handle_user_speech
    # True while handle_user_speech is executing; prevents duplicate STOP events
    is_processing_turn = False

    # report_generation_started — prevents do_complete_interview running twice
    report_generation_started = False

    # ── Interview config ──────────────────────────────────────────────────────
    role        = setup.get("role",       "Unknown Role")
    domain      = setup.get("domain",     "Unknown Domain")
    difficulty  = setup.get("difficulty", "Unknown Difficulty")
    topics_list = setup.get("topics", [])
    topics_str  = ", ".join(topics_list)
    groq_model  = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-20b")

    logger.info(
        "[INTERVIEW CONFIG] role=%s domain=%s difficulty=%s topics=%s",
        role, domain, difficulty, topics_list,
    )

    async def request_json(prompt: str, source: str) -> dict:
        """Use the configured LiveKit Groq plugin for JSON-only LLM work.

        Keeping all Groq calls on this client avoids requiring a second,
        separately-installed ``groq`` SDK just for scoring and reports.
        """
        logger.info("[LLM CALL] source=%s", source)
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(
            role="system",
            content="You are an expert technical interviewer evaluator. Output ONLY valid JSON matching the requested schema.",
        )
        chat_ctx.add_message(role="user", content=prompt)
        stream = groq.LLM(model=groq_model).chat(chat_ctx=chat_ctx)
        parts = []
        async with stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    parts.append(chunk.delta.content)
        return parse_json_object("".join(parts))

    # ── Data-channel helpers ──────────────────────────────────────────────────
    async def publish(msg: dict):
        logger.info("[DATA OUT] %s", msg)
        data = json.dumps(msg).encode("utf-8")
        if ctx.room and ctx.room.local_participant:
            await ctx.room.local_participant.publish_data(data, reliable=True)

    async def broadcast_state(state: str, q_num: int = None):
        msg = {"type": "interview_state", "state": state}
        if state == "READY_TO_ANSWER" and q_num is not None:
            msg = {
                "type":            "question_state",
                "state":           state,
                "question_number": q_num,
                "question_count":  question_count,
            }
        logger.info("STATE: %s", state)
        await publish(msg)

    async def send_question_started():
        await publish({
            "type":            "question_started",
            "question_number": current_q,
            "question_count":  question_count,
        })

    # ── Question generation ───────────────────────────────────────────────────
    #
    # ONLY TWO LEGITIMATE CALLERS:
    #   1. entrypoint   → generate_interview_question("initial")
    #   2. handle_user_speech (after evaluation) → generate_interview_question("after_answer")
    #
    async def generate_interview_question(source: str):
        nonlocal current_q, current_question_text, interview_phase

        # ── Hard guards: refuse if state machine is wrong ─────────────────────
        if interview_phase not in ("INIT", "GENERATING_QUESTION"):
            logger.warning(
                "[FLOW] QUESTION GENERATION BLOCKED reason=wrong_phase "
                "phase=%s source=%s current_q=%s evaluated=%d",
                interview_phase, source, current_q, len(interview_results),
            )
            return None

        if current_q >= MAX_INTERVIEW_QUESTIONS:
            logger.info(
                "[FLOW] QUESTION LIMIT REACHED - NO NEW QUESTION current_q=%s max=%s",
                current_q, MAX_INTERVIEW_QUESTIONS,
            )
            return None

        # Answers must be caught up: before Q(N) we need N-1 evaluated answers
        if current_q > 0 and len(interview_results) < current_q:
            logger.warning(
                "[FLOW] QUESTION GENERATION BLOCKED reason=answer_not_evaluated "
                "current_q=%s evaluated=%d",
                current_q, len(interview_results),
            )
            return None

        # ── Increment counter and generate ────────────────────────────────────
        interview_phase = "GENERATING_QUESTION"
        current_q += 1

        logger.info(
            "[FLOW] GENERATING QUESTION current_q=%s evaluated_answers=%d "
            "phase=%s source=%s",
            current_q, len(interview_results), interview_phase, source,
        )

        try:
            await send_question_started()

            language_instruction = "Ask and respond in clear professional English only."

            prompt = (
                f"You are a technical interviewer.\n"
                f"ROLE: {role}\n"
                f"DOMAIN: {domain}\n"
                f"DIFFICULTY: {difficulty}\n"
                f"TOPICS: {topics_str}\n"
                f"LANGUAGE: {language_instruction}\n\n"
                "Output EXACTLY ONE interview question as plain text.\n"
                "Rules:\n"
                "- One question only, nothing else.\n"
                "- No bullets, numbers, or labels.\n"
                "- No notes, explanations, or meta-commentary.\n"
                "- End with exactly one question mark.\n"
            )

            llm_client = groq.LLM(model=groq_model)

            async def ask_llm() -> str:
                logger.info("[LLM CALL] source=INTERVIEW_QUESTION")
                ctx_llm = llm.ChatContext()
                ctx_llm.add_message(role="system", content=prompt)
                stream = llm_client.chat(chat_ctx=ctx_llm)
                parts = []
                logger.info("[LLM] STREAM START")
                async with stream:
                    async for chunk in stream:
                        if chunk.delta and chunk.delta.content:
                            parts.append(chunk.delta.content)
                logger.info("[LLM] STREAM COMPLETE")
                return "".join(parts).strip()

            def is_valid(q: str) -> bool:
                if not q:
                    return False
                # Must end with a question mark
                if not q.endswith("?"):
                    return False
                # Must be a single line
                if "\n" in q:
                    return False
                # Must not contain multiple questions
                if q.count("?") > 1:
                    return False
                # Reject list formatting or meta-commentary
                bad = ("1.", "2.", "3.", "- ", "Note:", "(Note", "Question:",
                       "candidate has not", "previous question", "already asked")
                if any(b in q for b in bad):
                    return False
                return True

            question_text = await ask_llm()
            logger.info("[LLM] QUESTION TEXT READY: %s", question_text)

            if not is_valid(question_text):
                logger.warning("[VALIDATION FAILED] Retrying. Got: %r", question_text)
                question_text = await ask_llm()
                logger.info("[LLM] RETRY TEXT READY: %s", question_text)
                if not is_valid(question_text):
                    logger.warning("[VALIDATION FAILED TWICE] Using fallback.")
                    fallback_topic = topics_list[0] if topics_list else "your field"
                    question_text = f"Can you explain a core concept related to {fallback_topic}?"

            current_question_text = question_text

            # ── Speak the question via TTS ─────────────────────────────────────
            async def speak_interview_question(question_text: str):
                nonlocal interview_phase
                if interview_phase != "AI_SPEAKING": # Phase was set to AI_SPEAKING just before
                    logger.warning("[FLOW] SPOKEN RESPONSE BLOCKED: phase=%s", interview_phase)
                    return False
                
                logger.info("[FLOW] BEFORE SESSION.SAY question_number=%s phase=%s", current_q, interview_phase)
                logger.info("[FLOW] SESSION.SAY START")
                
                await session.say(text=question_text)
    
                logger.info("[FLOW] SESSION.SAY RETURNED question_number=%s", current_q)
    
                # session.say() resolves when TTS finishes playing
                interview_phase = "WAITING_FOR_ANSWER"
                logger.info("[FLOW] SETTING WAITING_FOR_ANSWER question_number=%s phase=%s", current_q, interview_phase)
                
                await publish({
                    "type": "waiting_for_answer",
                    "question_number": current_q,
                    "question_count": question_count,
                })
                return True

            interview_phase = "AI_SPEAKING"
            await speak_interview_question(question_text)
            
            return question_text

        except Exception:
            logger.exception("[QUESTION GENERATION FAILED] Full traceback:")
            # Roll back so this slot can potentially be retried
            current_q -= 1
            interview_phase = "INIT" if current_q == 0 else "WAITING_FOR_ANSWER"
            raise

    # ── Agent / Session setup ─────────────────────────────────────────────────
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Configure it in backend/.env")

    logger.info(f"ENV CHECK - LIVEKIT_URL: {'OK' if os.getenv('LIVEKIT_URL') else 'MISSING'}")
    logger.info(f"ENV CHECK - LIVEKIT_API_KEY: {'OK' if os.getenv('LIVEKIT_API_KEY') else 'MISSING'}")
    logger.info(f"ENV CHECK - LIVEKIT_API_SECRET: {'OK' if os.getenv('LIVEKIT_API_SECRET') else 'MISSING'}")
    logger.info(f"ENV CHECK - GROQ_API_KEY: {'OK' if groq_api_key else 'MISSING'}")

    system_prompt = (
        f"You are a technical interviewer conducting a one-on-one interview.\n"
        f"ROLE: {role}\nDOMAIN: {domain}\nDIFFICULTY: {difficulty}\n"
        f"FOCUS TOPICS: {topics_str}\n\n"
        "Ask exactly one focused technical question per turn. "
        "Never list multiple questions. Never evaluate or comment on answers. "
        "Understand and respond to candidate answers in English only. If the candidate uses another language, politely ask them in English to answer in English."
    )

    interviewer_agent = Agent(instructions=system_prompt)

    logger.info("[VOICE] Using English-only STT/TTS")
    session_stt = groq.STT()
    session_tts = inference.TTS(model="deepgram/aura-2", voice="athena", language="en")

    session = AgentSession(
        stt=session_stt,
        tts=session_tts,
        vad=_preloaded_vad,
        turn_detection="manual",
    )

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev: AgentStateChangedEvent):
        logger.info("[AGENT STATE] %s", ev.new_state)
        new_state = ev.new_state
        if new_state == "speaking":
            asyncio.create_task(broadcast_state("AI_SPEAKING"))
        elif new_state == "listening":
            # Only broadcast READY_TO_ANSWER when we are genuinely waiting.
            # DO NOT touch is_processing_turn here — it is owned by handle_user_speech.
            if interview_phase == "WAITING_FOR_ANSWER" and current_q > 0:
                asyncio.create_task(broadcast_state("READY_TO_ANSWER", current_q))

    candidate_transcript_buffer = []
    final_transcript_event = asyncio.Event()
    # Groq STT can return a final segment shortly after its first request times
    # out. Keep this flag so that late result is still evaluated instead of
    # leaving the candidate at the same question.
    awaiting_late_transcript = False

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev):
        nonlocal is_processing_turn, interview_phase, awaiting_late_transcript
        if getattr(ev, "is_final", False) and getattr(ev, "transcript", ""):
            text = ev.transcript.strip()
            if text:
                logger.info("[DEBUG-STT] TRANSCRIPT EVENT RECEIVED")
                logger.info("[DEBUG-STT] IS FINAL = True")
                logger.info("[DEBUG-STT] TRANSCRIPT TEXT = %s", text)
                candidate_transcript_buffer.append(text)
                final_transcript_event.set()
                if awaiting_late_transcript and interview_phase == "WAITING_FOR_ANSWER" and not is_processing_turn:
                    logger.info("[DEBUG-STT] LATE FINAL TRANSCRIPT — resuming answer evaluation")
                    awaiting_late_transcript = False
                    is_processing_turn = True
                    interview_phase = "PROCESSING"
                    asyncio.create_task(process_answer_task())

    async def repeat_current_question():
        """Replay the current question without advancing the interview state."""
        nonlocal interview_phase
        if interview_phase != "WAITING_FOR_ANSWER" or not current_question_text:
            logger.info("[REPEAT] Ignored phase=%s question_available=%s", interview_phase, bool(current_question_text))
            return

        logger.info("[REPEAT] Replaying question_number=%s", current_q)
        interview_phase = "AI_SPEAKING"
        await broadcast_state("AI_SPEAKING")
        try:
            await session.say(text=current_question_text)
        except Exception:
            logger.exception("[REPEAT] Unable to replay question")
        finally:
            interview_phase = "WAITING_FOR_ANSWER"
            await publish({
                "type": "waiting_for_answer",
                "question_number": current_q,
                "question_count": question_count,
            })

    @ctx.room.on("data_received")
    def on_data_received(data_packet):
        nonlocal is_processing_turn, interview_phase, awaiting_late_transcript
        try:
            payload_text = data_packet.data.decode("utf-8")
            logger.info("DATA RECEIVED: %s", payload_text)
            payload = json.loads(payload_text)

            if payload.get("type") == "start_answering":
                logger.info("START ANSWERING RECEIVED phase=%s", interview_phase)
                candidate_transcript_buffer.clear()
                final_transcript_event.clear()
                awaiting_late_transcript = False
                asyncio.ensure_future(broadcast_state("RECORDING"))

            elif payload.get("type") == "stop_answering":
                if is_processing_turn:
                    return
                if interview_phase != "WAITING_FOR_ANSWER":
                    return
                is_processing_turn = True
                awaiting_late_transcript = False
                interview_phase = "PROCESSING"
                logger.info("[PROCESSING LOCK] is_processing_turn = True")
                logger.info("[PTT] STOP ANSWERING RECEIVED")
                logger.info("[DEBUG-STT] STOP RECEIVED")
                logger.info("[DEBUG-STT] COMMIT USER TURN")
                logger.info("STATE: PROCESSING")
                asyncio.ensure_future(broadcast_state("PROCESSING"))
                try:
                    session.commit_user_turn()
                except Exception as e:
                    logger.warning("commit_user_turn err: %s", e)
                asyncio.create_task(process_answer_task())

            elif payload.get("type") == "complete_interview":
                logger.info("[DATA IN] complete_interview received")
                asyncio.ensure_future(do_complete_interview())

            elif payload.get("type") == "repeat_question":
                logger.info("[DATA IN] repeat_question received")
                asyncio.create_task(repeat_current_question())

        except Exception:
            logger.exception("ERROR HANDLING DATA RECEIVED")

    # ── Start session and fire Q1 ─────────────────────────────────────────────
    logger.info("[ROOM] STARTING AGENT SESSION")
    await session.start(room=ctx.room, agent=interviewer_agent)
    logger.info("[ROOM] AGENT SESSION STARTED")
    logger.info("[ROOM] ROOMIO ACTIVE")

    # THE ONLY TOP-LEVEL CALL TO generate_interview_question
    await generate_interview_question("initial")

    # ── Answer Evaluation ──
    async def evaluate_answer(question: str, answer: str) -> dict:
        logger.info("[EVAL DEBUG] question = %s", question)
        logger.info("[EVAL DEBUG] candidate_answer = %s", answer)
        logger.info("[EVAL DEBUG] answer_length = %d", len(answer))
        prompt = f"""Evaluate the candidate's answer to the following interview question.
Domain: {domain}
Role: {role}
Difficulty: {difficulty}

Question: {question}
Candidate Answer: {answer}

Provide a JSON response with the following fields:
- score (0-100)
- correctness (string describing if it was correct)
- relevance (0-10)
- completeness (0-10)
- technical_depth (0-10)
- feedback (constructive feedback for the candidate)
- strengths (list of strings)
- weaknesses (list of strings)
        """
        try:
            return await request_json(prompt, "ANSWER_EVALUATION")
        except Exception as e:
            logger.exception("[LLM] Failed to generate or parse answer evaluation JSON")
            return {
                "score": 0, "correctness": "Evaluation unavailable", 
                "relevance": 0, "completeness": 0, "technical_depth": 0, 
                "feedback": "Unable to evaluate answer due to parsing error.", 
                "strengths": [], "weaknesses": []
            }
    # ── Candidate answer handler ─────────────────────────────────────────────
    async def process_answer_task():
        nonlocal is_processing_turn, interview_phase, awaiting_late_transcript
        try:
            text = ""
            if candidate_transcript_buffer:
                text = " ".join(candidate_transcript_buffer).strip()

            if not text:
                try:
                    await asyncio.wait_for(final_transcript_event.wait(), timeout=10.0)
                    text = " ".join(candidate_transcript_buffer).strip()
                except asyncio.TimeoutError:
                    pass

            if not text:
                logger.info("[DEBUG-STT] EMPTY FINAL TRANSCRIPT")
                awaiting_late_transcript = True
                interview_phase = "WAITING_FOR_ANSWER"
                await publish({"type": "waiting_for_answer", "question_number": current_q, "question_count": MAX_INTERVIEW_QUESTIONS})
                return

            logger.info("[DEBUG-STT] FINAL ANSWER ACCEPTED")
            awaiting_late_transcript = False
            logger.info("[DEBUG-STT] STARTING EVALUATION")

            interview_phase = "EVALUATING_ANSWER"
            substantive_answer = is_substantive_answer(text)
            logger.info("[EVAL DEBUG] transcript_buffer = %s", candidate_transcript_buffer)
            logger.info("[EVAL DEBUG] resolved_candidate_answer = %s", text)
            logger.info("[EVAL DEBUG] resolved_answer_length = %d", len(text))
            eval_data = (
                await evaluate_answer(current_question_text, text)
                if substantive_answer
                else missing_answer_evaluation()
            )
            logger.info("[DEBUG-STT] EVALUATION COMPLETED")

            result = {
                "question_number":  current_q,
                "question":         current_question_text,
                "candidate_answer": text,
                "answer":           text,
                "is_substantive_answer": substantive_answer,
                "evaluation":       eval_data,
                "score":            eval_data.get("score", 0),
                "correctness":      eval_data.get("correctness", ""),
                "feedback":         eval_data.get("feedback", ""),
                "strengths":        eval_data.get("strengths", []),
                "weaknesses":       eval_data.get("weaknesses", []),
            }
            interview_results.append(result)
            logger.info("[DEBUG-STT] RESULT STORED")

            if current_q < MAX_INTERVIEW_QUESTIONS:
                logger.info(
                    "[FLOW] NEXT QUESTION AUTHORIZED current_q=%s evaluated_answers=%d",
                    current_q, len(interview_results),
                )
                interview_phase = "GENERATING_QUESTION"
                await generate_interview_question("after_answer")
            else:
                logger.info("[FLOW] Final question evaluated - waiting for completion.")
                interview_phase = "WAITING_FOR_COMPLETION"
                await publish({"type": "interview_ready_to_complete"})

        except Exception:
            logger.exception("Error in process_answer_task")
            interview_phase = "WAITING_FOR_ANSWER"
            await publish({
                "type": "waiting_for_answer",
                "question_number": current_q,
                "question_count":  MAX_INTERVIEW_QUESTIONS,
            })
        finally:
            is_processing_turn = False
            logger.info("[PROCESSING LOCK] is_processing_turn = False")

            return

    # ── Complete interview (frontend button) ──────────────────────────────────
    async def do_complete_interview():
        nonlocal report_generation_started, interview_phase

        if report_generation_started:
            logger.info("[COMPLETE_INTERVIEW] Ignoring duplicate")
            return
        report_generation_started = True
        interview_phase = "GENERATING_REPORT"

        logger.info("[COMPLETE_INTERVIEW] Received")

        if len(interview_results) < MAX_INTERVIEW_QUESTIONS:
            logger.error(
                "[COMPLETE_INTERVIEW] REJECTED: only %d results, need %d",
                len(interview_results), MAX_INTERVIEW_QUESTIONS,
            )
            report_generation_started = False
            return

        logger.info("[REPORT] RESULTS COUNT = %d", len(interview_results))
        logger.info(
            "[REPORT] Final interview_results: %s",
            json.dumps(interview_results, indent=2),
        )

        scores = [
            safe_score(result.get("score"))
            if result.get("is_substantive_answer", is_substantive_answer(result.get("answer", "")))
            else 0
            for result in interview_results
        ]
        overall = round(sum(scores) / len(scores)) if scores else 0
        has_substantive_answer = any(
            result.get("is_substantive_answer", is_substantive_answer(result.get("answer", "")))
            for result in interview_results
        )

        logger.info("[LLM CALL] source=FINAL_REPORT")
        try:
            prompt = f"""Generate a final interview report based on the following candidate evaluations:
Domain: {domain}
Role: {role}
Difficulty: {difficulty}

Evaluations:
{json.dumps(interview_results, indent=2)}

Provide a JSON response matching this schema exactly:
- overall_score (0-100)
- technical_score (0-100)
- communication_score (0-100)
- strengths (list of strings)
- weaknesses (list of strings)
- final_feedback (string summarizing performance)
- recommendation (string, e.g. "STRONG HIRE", "HIRE", "NO HIRE")
"""
            report_data = await request_json(prompt, "FINAL_REPORT")
            
            report = {
                "overall_score":       report_data.get("overall_score", overall),
                "technical_score":     report_data.get("technical_score", overall),
                "communication_score": report_data.get("communication_score", overall),
                "strengths":           report_data.get("strengths", []),
                "weaknesses":          report_data.get("weaknesses", []),
                "final_feedback":      report_data.get("final_feedback", "Overall solid performance."),
                "recommendation":      report_data.get("recommendation", "CONSIDER"),
                "question_results":    interview_results,
            }
        except Exception:
            logger.exception("[LLM] Failed to generate or parse final report JSON")
            strengths = []
            weaknesses = []
            for result in interview_results:
                strengths.extend(result.get("strengths") or [])
                weaknesses.extend(result.get("weaknesses") or [])
            strengths = list(dict.fromkeys(strengths))[:5]
            weaknesses = list(dict.fromkeys(weaknesses))[:5]
            if not strengths:
                strengths = ["Completed the interview"]
            if not weaknesses:
                weaknesses = ["Review the topics covered in the interview"]
            recommendation = "STRONG HIRE" if overall >= 85 else "HIRE" if overall >= 70 else "CONSIDER" if overall >= 50 else "NO HIRE"
            report = {
                "overall_score":       overall,
                "technical_score":     overall,
                "communication_score": overall,
                "strengths":           strengths,
                "weaknesses":          weaknesses,
                "final_feedback":      "The automated narrative report was unavailable, so this summary is based on the per-question evaluations.",
                "recommendation":      recommendation,
                "question_results":    interview_results,
            }

        # A report model must not manufacture positives from an interview with
        # no answer. This policy takes precedence over model-generated prose.
        if not has_substantive_answer:
            report.update({
                "overall_score": 0,
                "technical_score": 0,
                "communication_score": 0,
                "strengths": [],
                "weaknesses": [
                    "Provide a substantive answer to each question.",
                    "Explain your reasoning with relevant technical concepts and examples.",
                ],
                "final_feedback": "No substantive answers were recorded, so technical and communication skills could not be assessed.",
                "recommendation": "NO HIRE",
            })

        await complete_session(session_id, report)
        logger.info("[REPORT] Stored via complete_session")

        interview_phase = "COMPLETE"
        await publish({"type": "interview_completed"})

        await asyncio.sleep(1)
        await ctx.room.disconnect()
        logger.info("[INTERVIEW] COMPLETE — disconnected")

    # user_speech_committed is intentionally NOT wired: answer processing
    # is driven by process_answer_task launched from on_data_received.



if __name__ == "__main__":
    agents.cli.run_app(server)
