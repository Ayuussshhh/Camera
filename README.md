# FaceTrace Attendance System

Production-oriented college attendance platform built with:

- `Next.js` + `MUI` frontend
- `Next.js` API routes for server-side orchestration
- `PostgreSQL` for persistent academic, attendance, and audit data
- `FastAPI` AI service using `MediaPipe` face embeddings

## Architecture

- `frontend/`: dashboard UI, API routes, Postgres-backed server services
- `python-ai/`: face enrollment, embedding generation, live recognition
- `database/`: schema and seed scripts for PostgreSQL

## Frontend Setup

```powershell
cd frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

Required environment values in `frontend/.env.local` (or `frontend/.env`):

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:3000
NEXT_PUBLIC_AI_SERVICE_URL=http://127.0.0.1:8000
AI_SERVICE_URL=http://127.0.0.1:8000
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/attendance
```

You can also use split PostgreSQL settings instead of `DATABASE_URL`:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=attendance
DB_USER=postgres
DB_PASSWORD=your_password
```

Frontend URL:

```text
http://localhost:3000
``

## Database Setup

```powershell
createdb attendance
psql -d attendance -f database/schema.sql
psql -d attendance -f database/seed.sql
```

## Python AI Service Setup

```powershell
cd python-ai
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

AI service URL:

```text
http://127.0.0.1:8000
```

## Recognition Pipeline

The AI service now uses one supported recognition path:

- MediaPipe face landmarks for detection and alignment
- OpenCV preprocessing for normalization, CLAHE, gradients, and DCT features
- Weighted face embedding profiles stored per student under `python-ai/embeddings/mediapipe/`

Enrollment works by saving multiple student images, extracting the best-quality face samples, and creating a stable embedding profile. This is an embedding-based recognition workflow, not deep neural network re-training.

## Runtime Notes

- Mock attendance repositories have been removed from the active runtime path.
- A valid `DATABASE_URL` or the split `DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD` values are required for the frontend server services.
- The Python AI service must be running for enrollment and live recognition.
- Student face embeddings are saved in per-student JSON records and a generated gallery index.
