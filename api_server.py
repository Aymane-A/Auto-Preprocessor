"""Loopback FastAPI service with bounded uploads and isolated dataset workers."""

import asyncio
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shutil
import tempfile
from threading import Lock
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from isolated_processing import ProcessLimits, run_isolated
import web_processing as work

LOG = logging.getLogger(__name__)
MIB = 1024 ** 2
SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".parquet"}
ORIGIN_DEV = "http://localhost:5173"
ORIGINS = {ORIGIN_DEV, "http://127.0.0.1:8000", "http://localhost:8000"}


class SessionRequest(BaseModel):
    """Identify a single server-owned dataset."""
    model_config = ConfigDict(extra="forbid")
    session_id: str


class Configure(SessionRequest):
    """Describe the learning mode and optional supervised target."""
    mode: Literal["supervised", "unsupervised"]
    target: str | None = None
    task: Literal["auto", "classification", "regression"] = "auto"


class Preview(SessionRequest):
    """Describe one feature preview."""
    column: str
    choice: str = "recommended"


class WebOptions(BaseModel):
    """Allow only supported desktop options with bounded values."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    test_size_pct: int = Field(20, ge=1, le=50)
    scaler_type: Literal["standard", "robust", "minmax", "none"] = "standard"
    outlier_method: Literal["winsorize", "iqr", "none"] = "winsorize"
    cardinality_threshold: int = Field(10, ge=2, le=1000)
    missing_threshold: float = Field(.90, gt=0, le=1)
    correlation_threshold: float = Field(.95, gt=0, le=1)
    rare_category_threshold: float = Field(.01, ge=0, lt=1)
    skew_threshold: float = Field(1.0, ge=0, le=100)
    transform_target: bool = True
    drop_leaky_columns: bool = False
    encode_discrete_as_categorical: bool = False
    random_state: int = Field(42, ge=0, le=2**32 - 1)
    sparse_output: bool = False
    model_family: Literal["linear", "logistic", "decision_tree", "random_forest", "gradient_boosting",
                          "xgboost", "lightgbm", "knn", "svm", "neural_network"] | None = None


class Execute(SessionRequest):
    """Describe transformation choices and fit options."""
    options: WebOptions = Field(default_factory=WebOptions)
    choices: dict[str, str] = Field(default_factory=dict)


@dataclass
class Session:
    """Keep only metadata and server-owned file references in the API process."""
    folder: Path
    path: Path
    name: str
    expires: float
    target: str | None = None
    task: str = "auto"
    configured: bool = False
    output: Path | None = None
    lock: Lock = field(default_factory=Lock)


class SessionStore:
    """Expire idle sessions and serialize work on each dataset."""

    def __init__(self, root, ttl):
        """Initialize the private directory and session registry."""
        self.root = root
        self.ttl = ttl
        self.sessions = {}
        self.lock = Lock()

    def clean(self):
        """Remove expired directories when they have no active reader or writer."""
        with self.lock:
            expired = []
            for key, session in list(self.sessions.items()):
                if session.expires <= time.monotonic() and session.lock.acquire(blocking=False):
                    expired.append(self.sessions.pop(key))
        for session in expired:
            shutil.rmtree(session.folder, ignore_errors=True)

    @contextmanager
    def use(self, key, *, configured=True):
        """Pin a session for the duration of an operation."""
        self.clean()
        with self.lock:
            session = self.sessions.get(key)
            if session is None:
                raise HTTPException(404, "Session introuvable ou expirée. Importez à nouveau le fichier.")
            if not session.lock.acquire(blocking=False):
                raise HTTPException(409, "Une opération est déjà en cours pour cette session.")
        try:
            if configured and not session.configured:
                raise HTTPException(409, "Configurez la session avant de continuer.")
            yield session
        finally:
            session.expires = time.monotonic() + self.ttl
            session.lock.release()


def safe_name(value):
    """Sanitize the display name and validate its tabular extension."""
    name = (value or "").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^\w. ()-]", "_", name)[:180] if len(name) <= 180 else (
        re.sub(r"[^\w. ()-]", "_", Path(name).stem)[:160] + Path(name).suffix[:16])
    suffix = Path(name).suffix.lower()
    if suffix not in SUFFIXES:
        raise HTTPException(415, "Format non pris en charge.")
    return name, suffix


async def stream_upload(request, folder, limit):
    """Stream exactly one multipart file into a server-generated filename."""
    media, params = parse_options_header(request.headers.get("content-type", ""))
    boundary = params.get(b"boundary", b"")
    if media != b"multipart/form-data" or not 1 <= len(boundary) <= 200:
        raise HTTPException(415, "Envoyez un fichier multipart dans le champ file.")
    total = size = 0
    header_name = bytearray()
    header_value = bytearray()
    headers = {}
    name = suffix = None
    finished = False

    def header_field(data, start, end):
        """Collect the current multipart header name."""
        header_name.extend(data[start:end])

    def header_data(data, start, end):
        """Collect the current multipart header value."""
        header_value.extend(data[start:end])

    def header_end():
        """Store one parsed multipart header."""
        headers[bytes(header_name).lower()] = bytes(header_value)
        header_name.clear()
        header_value.clear()

    def part_begin():
        """Reject additional multipart parts."""
        if name is not None:
            raise HTTPException(422, "Un seul fichier est autorisé.")

    def headers_finished():
        """Validate the multipart field name and sanitize the submitted filename."""
        nonlocal name, suffix
        disposition, fields = parse_options_header(headers.get(b"content-disposition", b""))
        if disposition != b"form-data" or fields.get(b"name") != b"file" or b"filename" not in fields:
            raise HTTPException(422, "Le champ file est obligatoire.")
        name, suffix = safe_name(fields[b"filename"].decode("utf-8", errors="replace"))

    def part_data(data, start, end):
        """Write a bounded chunk without buffering the dataset in the API."""
        nonlocal size
        size += end - start
        if size > limit:
            raise HTTPException(413, "Fichier trop volumineux pour la limite configurée.")
        handle.write(memoryview(data)[start:end])

    def end():
        """Record that the closing multipart boundary was parsed."""
        nonlocal finished
        finished = True

    raw = folder / "upload"
    try:
        with raw.open("wb") as handle:
            parser = MultipartParser(boundary, {
                "on_part_begin": part_begin, "on_header_field": header_field, "on_header_value": header_data,
                "on_header_end": header_end, "on_headers_finished": headers_finished, "on_part_data": part_data,
                "on_end": end,
            })
            async for chunk in request.stream():
                total += len(chunk)
                if total > limit + 64 * 1024:
                    raise HTTPException(413, "Fichier trop volumineux pour la limite configurée.")
                await run_in_threadpool(parser.write, chunk)
            parser.finalize()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, "Envoi de fichier incomplet ou illisible.") from exc
    if not finished or not size or name is None:
        raise HTTPException(422, "Fichier vide ou incomplet.")
    path = raw.rename(folder / ("dataset" + suffix))
    return path, name


def worker_call(function, args):
    """Return controlled input failures without leaking parser internals."""
    try:
        return function(*args), None
    except work.InputError as exc:
        return None, str(exc)


def isolated(function, *args, limits):
    """Run a trusted job and map worker failures to user-facing HTTP statuses."""
    try:
        (result, error), usage = run_isolated(worker_call, function, args, limits=limits, with_usage=True)
    except TimeoutError as exc:
        raise HTTPException(504, "Délai de traitement dépassé.") from exc
    except (MemoryError, RuntimeError, ValueError, OSError) as exc:
        message = str(exc)
        if re.search("mémoire|memory|allocat|budget", message, re.I):
            raise HTTPException(507, "Limite mémoire atteinte pendant le traitement.") from exc
        LOG.warning("Échec du traitement isolé (%s)", type(exc).__name__)
        raise HTTPException(422, "Traitement impossible pour ces données ou ces options.") from exc
    if error:
        raise HTTPException(422, error)
    return result, {"elapsed_seconds": round(usage.elapsed_seconds, 3), "peak_rss_bytes": usage.peak_rss_bytes}


def create_app(*, max_table_bytes=None, ttl=1800, limits=None, frontend=None):
    """Build the local API with configurable storage and worker budgets."""
    limit = max_table_bytes if max_table_bytes is not None else int(os.getenv("AUTOPREP_MAX_TABLE_MB", "100")) * MIB
    limits = limits or ProcessLimits(int(os.getenv("AUTOPREP_WORKER_MB", "1024")) * MIB,
                                     float(os.getenv("AUTOPREP_TIMEOUT_SECONDS", "120")))
    if limit <= 0 or ttl <= 0:
        raise ValueError("Les limites d'import et de session doivent être positives.")

    @asynccontextmanager
    async def lifespan(app):
        """Own all dataset storage and run periodic expiry cleanup."""
        with tempfile.TemporaryDirectory(prefix="autoprep-web-") as root:
            app.state.store = SessionStore(Path(root), ttl)

            async def sweep():
                """Remove idle uploads even when there is no incoming traffic."""
                while True:
                    await asyncio.sleep(min(30, ttl))
                    await run_in_threadpool(app.state.store.clean)

            task = asyncio.create_task(sweep())
            try:
                yield
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(title="Auto-Préprocesseur local", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])
    app.add_middleware(CORSMiddleware, allow_origins=[ORIGIN_DEV], allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])

    @app.middleware("http")
    async def local_origin(request, call_next):
        """Reject API access initiated by an unexpected browser origin."""
        origin = request.headers.get("origin")
        if request.url.path.startswith("/api/") and origin and origin not in ORIGINS:
            return JSONResponse({"detail": "Origine non autorisée."}, status_code=403)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        """Return a readable validation error without echoing submitted data."""
        fields = ", ".join(".".join(str(p) for p in error["loc"][1:]) for error in exc.errors())
        return JSONResponse({"detail": f"Paramètres invalides : {fields or 'requête'}."}, status_code=422)

    @app.post("/api/upload")
    async def upload(request: Request):
        """Stream, validate and register a dataset in a random session."""
        store = app.state.store
        await run_in_threadpool(store.clean)
        key = secrets.token_urlsafe(24)
        folder = store.root / key
        folder.mkdir()
        try:
            path, name = await stream_upload(request, folder, limit)
            result, usage = await run_in_threadpool(lambda: isolated(work.upload_summary, path, limit, limits=limits))
            with store.lock:
                store.sessions[key] = Session(folder, path, name, time.monotonic() + ttl)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return {"session_id": key, "filename": name, **result, "worker": usage}

    @app.post("/api/config")
    def configure(body: Configure):
        """Validate the requested configuration and discard stale results."""
        if (body.mode == "supervised") != bool(body.target):
            raise HTTPException(422, "Sélectionnez une cible uniquement en mode supervisé.")
        if body.mode == "unsupervised" and body.task != "auto":
            raise HTTPException(422, "La tâche doit être automatique en mode non supervisé.")
        with app.state.store.use(body.session_id, configured=False) as session:
            isolated(work.validate_config, session.path, limit, body.target, body.task, limits=limits)
            if (session.target, session.task) != (body.target, body.task) and session.output:
                shutil.rmtree(session.output, ignore_errors=True)
                session.output = None
            session.target, session.task, session.configured = body.target, body.task, True
        return {"mode": body.mode, "target": body.target, "task": body.task}

    @app.get("/api/eda/{session_id}")
    def dataset_eda(session_id: str):
        """Return descriptive statistics and bounded association matrices."""
        with app.state.store.use(session_id) as session:
            return isolated(work.eda, session.path, limit, session.target, session.task, limits=limits)[0]

    @app.get("/api/eda/{session_id}/feature/{column:path}")
    def column_eda(session_id: str, column: str):
        """Return distribution and boxplot data for one dataset column."""
        with app.state.store.use(session_id) as session:
            return isolated(work.feature_eda, session.path, limit, column, limits=limits)[0]

    @app.post("/api/analyze")
    def analyze(body: SessionRequest):
        """Return justified preprocessing recommendations."""
        with app.state.store.use(body.session_id) as session:
            return isolated(work.analyze, session.path, limit, session.target, session.task, limits=limits)[0]

    @app.post("/api/preview")
    def preview(body: Preview):
        """Return the before/after preview for one selected transformation."""
        with app.state.store.use(body.session_id) as session:
            return isolated(work.preview, session.path, limit, session.target, session.task, body.column, body.choice,
                            limits=limits)[0]

    @app.post("/api/preprocess")
    def preprocess(body: Execute):
        """Fit and atomically publish a new set of session exports."""
        with app.state.store.use(body.session_id) as session:
            output = Path(tempfile.mkdtemp(prefix="export-", dir=session.folder))
            try:
                result, usage = isolated(work.execute, session.path, limit, session.target, session.task,
                                         body.options.model_dump(), body.choices, output, session.name, limits=limits)
            except BaseException:
                shutil.rmtree(output, ignore_errors=True)
                raise
            if session.output:
                shutil.rmtree(session.output, ignore_errors=True)
            session.output = output
            return {**result, "worker": usage}

    @app.get("/api/report/{session_id}")
    def report(session_id: str):
        """Return the most recent transformation report."""
        with app.state.store.use(session_id) as session:
            if session.output is None:
                raise HTTPException(404, "Lancez le prétraitement pour générer le rapport.")
            return json.loads((session.output / "report.json").read_text(encoding="utf-8"))

    @app.get("/api/download/{session_id}/{kind}")
    def download(session_id: str, kind: Literal["train", "test", "full", "report_xlsx"]):
        """Stream a fixed export name and keep the session pinned while reading."""
        lease = app.state.store.use(session_id)
        session = lease.__enter__()
        try:
            if session.output is None:
                raise HTTPException(404, "Aucun fichier exporté pour cette session.")
            filename = work.DOWNLOADS[kind]
            path = session.output / filename
            media = ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                     if kind == "report_xlsx" else "text/csv")

            async def chunks():
                """Release the session even if the download disconnects."""
                try:
                    with path.open("rb") as handle:
                        while chunk := await run_in_threadpool(handle.read, MIB):
                            yield chunk
                finally:
                    lease.__exit__(None, None, None)

            return StreamingResponse(chunks(), media_type=media,
                                     headers={"Content-Disposition": f'attachment; filename="{filename}"'})
        except BaseException:
            lease.__exit__(None, None, None)
            raise

    dist = Path(frontend) if frontend is not None else Path(__file__).parent / "frontend" / "dist"
    if (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app


app = create_app()
