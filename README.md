# InterviewOS

InterviewOS is an AI-powered technical candidate-screening platform. A recruiter configures a role-specific voice assessment, the candidate completes it with an AI interviewer, and the application produces a structured screening report with scores, feedback, strengths, and areas for review.

> This project is designed for first-round screening support. Its report is decision-support information for a recruiter; it does not make an automatic hiring or rejection decision.

## Key capabilities

- Role, domain, difficulty, focus-area, and question-count configuration.
- Real-time voice interview using LiveKit.
- One question is generated at a time, so the system does not pre-generate unused questions.
- English-only voice interview questions and responses.
- Push-to-talk candidate responses and a **Repeat question** action.
- Per-question evaluation plus a final candidate screening report.
- Responsive React UI with live interview states and report visualizations.

## Tech stack

| Area | Technology |
| --- | --- |
| Frontend | React, Vite, Tailwind CSS, LiveKit Components |
| Backend API | FastAPI, Uvicorn |
| Voice transport | LiveKit |
| Agent orchestration | LiveKit Agents |
| LLM / evaluation | Groq |
| Speech recognition | Groq STT |
| Speech synthesis | Deepgram Aura 2 (`athena`, English) |

## Prerequisites

Install these before running the project:

- Windows 10/11
- [Python 3.11](https://www.python.org/downloads/release/python-3119/)
- Node.js 20+ and npm
- A LiveKit project with a URL, API key, and API secret
- A Groq API key

## 1. Clone and prepare the project

```powershell
git clone <YOUR_PRIVATE_REPOSITORY_URL>
cd "Agent Interview"
```

Create a Python virtual environment:

```powershell
& "C:\Users\<YOUR_WINDOWS_USER>\AppData\Local\Programs\Python\Python311\python.exe" -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in the current terminal and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Install backend dependencies:

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r .\backend\requirements.txt
```

Install frontend dependencies:

```powershell
cd frontend
npm install
cd ..
```

## 2. Configure environment variables

Copy the template file:

```powershell
Copy-Item .env.example .env
```

Open `.env` and provide real values. **Never commit this file.**

```env
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

GROQ_API_KEY=your_groq_api_key

```

## 3. Run the application

Open **three PowerShell terminals**. Keep all three running while testing.

### Terminal 1 — FastAPI backend

```powershell
cd "C:\path\to\Agent Interview"
.\venv\Scripts\Activate.ps1
cd backend
python -m uvicorn app.main:app --reload
```

FastAPI starts at `http://localhost:8000`.

### Terminal 2 — LiveKit AI interviewer agent

```powershell
cd "C:\path\to\Agent Interview"
.\venv\Scripts\Activate.ps1
cd backend
python -m app.interview.agent start
```

Confirm this log appears before beginning an interview:

```text
[VOICE] Using English-only STT/TTS
```

If it fails to initialize speech services, check that the LiveKit and Groq configuration is present in the root `.env` file and restart the agent.

### Terminal 3 — React frontend

```powershell
cd "C:\path\to\Agent Interview\frontend"
npm run dev
```

Open the URL printed by Vite (normally `http://localhost:5173` or `http://localhost:5174`).

## 4. Use the application

1. Open the frontend URL.
2. Configure the screening domain, target role, difficulty, focus areas, and question count.
3. Select **Create AI screening**.
4. Allow microphone access in the browser.
5. Listen to each question and select **Start Answering**.
6. Select **Stop Answering** after speaking. The agent transcribes, evaluates, and asks the next question.
7. Select **Repeat question** if the candidate needs to hear the current question again.
8. When all questions are complete, generate and review the **Candidate Screening Report**.

## Validation commands

Run these before sharing changes:

```powershell
# From the project root
.\venv\Scripts\python.exe -m py_compile .\backend\app\main.py .\backend\app\interview\agent.py

# From the frontend folder
cd frontend
npm run build
```

## Troubleshooting

### `Unable to create process` / broken virtual environment

Delete only the `venv` folder, then recreate it using Python 3.11:

```powershell
cd "C:\path\to\Agent Interview"
Remove-Item -Recurse -Force .\venv
& "C:\Users\<YOUR_WINDOWS_USER>\AppData\Local\Programs\Python\Python311\python.exe" -m venv venv
.\venv\Scripts\python.exe -m pip install -r .\backend\requirements.txt
```

### English voice is unclear or no voice plays

- Ensure the agent terminal shows the English-only STT/TTS startup log above.
- Confirm the configured LiveKit and Groq credentials are valid, then restart the agent.
- Stop the agent with `Ctrl + C`, then start it again after changing `.env`.
- Start a **new** screening session after restarting the agent.

### The UI stays on “Connecting to secure screening”

- Confirm Terminal 1 is running and `http://localhost:8000/docs` opens.
- Confirm Terminal 2 is running and has connected to LiveKit.
- Create a new session after restarting either the API or agent.

### Report disappears after backend restart

Current sessions and reports are stored in FastAPI process memory. Restarting the API clears them. This is expected in the current prototype; a database is not included.

## Safe GitHub sharing

Before pushing, verify no keys are staged:

```powershell
git status
git add .
git status
```

The included `.gitignore` excludes `.env`, `venv`, Node dependencies, build output, and logs. Commit only `.env.example`, never `.env`.

To create a private GitHub repository:

1. On GitHub, select **New repository**.
2. Choose a repository name, for example `interviewos`.
3. Select **Private**.
4. Do not add GitHub’s generated README because this project already has one.
5. Add your company reviewer as a collaborator with the least access they need.
6. Initialize and push from the project root:

```powershell
git init
git add .
git commit -m "Initial InterviewOS project"
git branch -M main
git remote add origin <YOUR_PRIVATE_REPOSITORY_URL>
git push -u origin main
```

## Important project constraints

- Do not commit API keys, `.env`, access tokens, audio recordings, or candidate data.
- Do not expose the LiveKit API secret to the frontend.
- The frontend uses the existing FastAPI API contracts and LiveKit token flow.
- The agent creates questions one at a time and uses the configured actual question count.
