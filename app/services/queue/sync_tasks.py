"""
Sync Background Tasks
=====================

Celery tasks that offload S3 data sync and sales statistics sync from the
main FastAPI process.  Previously these ran as asyncio background loops inside
the app, tying up the uvicorn event-loop and restarting from scratch on every
deploy.  Moving them to Celery lets:

  - the web process stay lightweight (no background threads pulling from S3),
  - multiple workers share the load,
  - beat schedule the cadence centrally,
  - retries / dead-letter handled by the broker.

Tasks:
  sync.s3_data_sync        – full S3 → local sync (templates, datasets, resumes)
  sync.sales_stats_sync    – map sales reps + recalculate stats
  sync.templates_sync_s3   – sync only HTML templates from S3 (on-demand)
"""

import time
from pathlib import Path
from typing import Any, Dict

from app.core.logger import get_logger

logger = get_logger(__name__)

try:
    from app.core.celery_app import celery_app
except ImportError:
    celery_app = None


# ---------------------------------------------------------------------------
# S3 full data sync (templates + datasets + resumes)
# ---------------------------------------------------------------------------

def s3_data_sync() -> Dict[str, Any]:
    """
    Sync templates, datasets, and resumes from S3 to local app/data.
    Reuses the existing _run_full_s3_data_sync() from app.py so behaviour
    is identical to the old asyncio loop.  Runs every ~8s via Celery Beat.
    """
    _start = time.time()
    logger.debug("[SyncTask] S3 data sync START")
    try:
        from pathlib import Path
        from app.app import _run_full_s3_data_sync

        app_dir = Path(__file__).resolve().parent.parent.parent
        if not (app_dir / "data").exists():
            alt = app_dir / "app"
            if (alt / "data").exists():
                app_dir = alt

        result = _run_full_s3_data_sync(app_dir)
        elapsed = time.time() - _start
        t = result.get("templates", {})
        d = result.get("datasets", {})
        r = result.get("resumes", {})
        total_synced = t.get("synced", 0) + d.get("synced", 0) + r.get("synced", 0)
        total_failed = t.get("failed", 0) + d.get("failed", 0) + r.get("failed", 0)
        if total_synced > 0 or total_failed > 0:
            logger.info(
                "[SyncTask] S3 data sync DONE in %.1fs — "
                "templates %d/%d, datasets %d/%d, resumes %d/%d",
                elapsed,
                t.get("synced", 0), t.get("failed", 0),
                d.get("synced", 0), d.get("failed", 0),
                r.get("synced", 0), r.get("failed", 0),
            )
        return {
            "status": "success",
            "elapsed_seconds": round(elapsed, 2),
            **result,
        }
    except Exception as e:
        elapsed = time.time() - _start
        logger.error("[SyncTask] S3 data sync FAILED in %.1fs: %s", elapsed, e, exc_info=True)
        return {"status": "error", "error": str(e), "elapsed_seconds": round(elapsed, 2)}


# ---------------------------------------------------------------------------
# Sales statistics sync
# ---------------------------------------------------------------------------

def sales_stats_sync(verbose: bool = False) -> Dict[str, Any]:
    """
    Map sales rep names → IDs and recalculate per-rep statistics.
    Wraps the existing sync_sales_stats() function.
    """
    _start = time.time()
    logger.info("[SyncTask] Sales stats sync START")
    try:
        from app.services.sales.sales_stats_sync import sync_sales_stats
        result = sync_sales_stats(verbose=verbose)
        elapsed = time.time() - _start
        logger.info(
            "[SyncTask] Sales stats sync DONE in %.1fs — "
            "leads_mapped=%s reps_updated=%s success=%s",
            elapsed,
            result.get("leads_mapped", 0),
            result.get("reps_updated", 0),
            result.get("success"),
        )
        return {
            "status": "success" if result.get("success") else "error",
            "elapsed_seconds": round(elapsed, 2),
            **result,
        }
    except Exception as e:
        elapsed = time.time() - _start
        logger.error("[SyncTask] Sales stats sync FAILED in %.1fs: %s", elapsed, e, exc_info=True)
        return {"status": "error", "error": str(e), "elapsed_seconds": round(elapsed, 2)}


# ---------------------------------------------------------------------------
# Template-only S3 sync (on-demand via API)
# ---------------------------------------------------------------------------

def templates_sync_s3() -> Dict[str, Any]:
    """
    Sync only HTML templates from S3 (template_*.html).
    Designed for on-demand use from the /templates/sync-from-s3 endpoint.
    """
    _start = time.time()
    logger.info("[SyncTask] Template S3 sync START")
    try:
        from pathlib import Path as PathLib
        from app.services.storage.s3_functions import list_files_in_s3, download_file_from_s3
        from datetime import datetime

        app_dir = PathLib(__file__).resolve().parent.parent.parent
        if not (app_dir / "data").exists():
            alt = app_dir / "app"
            if (alt / "data").exists():
                app_dir = alt

        templates_dir = app_dir / "data" / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)

        s3_result = list_files_in_s3(prefix="templates/", file_extensions=[".html"])

        synced = []
        failed = []

        for file_info in s3_result.get("files", []):
            s3_key = file_info.get("key", "")
            filename = PathLib(s3_key).name
            if not filename.startswith("template_") or not filename.endswith(".html"):
                continue
            try:
                local_path = templates_dir / filename
                download_file_from_s3(s3_key=s3_key, local_file_path=str(local_path), create_directories=True)
                synced.append({"name": filename, "s3_key": s3_key, "synced_at": datetime.now().isoformat()})
            except Exception as e:
                logger.error("[SyncTask] Template download failed %s: %s", s3_key, e)
                failed.append({"name": filename, "s3_key": s3_key, "error": str(e)})

        elapsed = time.time() - _start
        logger.info(
            "[SyncTask] Template S3 sync DONE in %.1fs — %d synced, %d failed",
            elapsed, len(synced), len(failed),
        )
        return {
            "status": "success",
            "elapsed_seconds": round(elapsed, 2),
            "synced": len(synced),
            "failed": len(failed),
            "templates_synced": synced,
            "templates_failed": failed,
        }
    except Exception as e:
        elapsed = time.time() - _start
        logger.error("[SyncTask] Template S3 sync FAILED in %.1fs: %s", elapsed, e, exc_info=True)
        return {"status": "error", "error": str(e), "elapsed_seconds": round(elapsed, 2)}


# ---------------------------------------------------------------------------
# Newsletter provider reports sync (Mailjet + SendGrid)
# ---------------------------------------------------------------------------

def provider_reports_sync(days: int = 1) -> Dict[str, Any]:
    """
    Fetch email performance reports from Mailjet and SendGrid and save to DB.
    Wraps the sync logic from newsletter_provider_reports.py.
    Called daily by Celery Beat, or on-demand from the API.
    """
    _start = time.time()
    logger.info("[SyncTask] Provider reports sync START (days=%d)", days)
    try:
        from datetime import datetime, timedelta
        from app.core.db import get_db_connection
        from app.services.sales.newsletter_provider_reports import (
            process_mailjet_reports,
            process_sendgrid_reports,
            init_email_reports_table,
        )

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        conn = get_db_connection()
        if not conn:
            raise RuntimeError("Database connection failed")

        try:
            init_email_reports_table(conn)

            total = 0
            results = {}
            for provider_key, processor in [
                ('mailjet', lambda: process_mailjet_reports('mailjet', start_date, end_date, conn)),
                ('mailjet2', lambda: process_mailjet_reports('mailjet2', start_date, end_date, conn)),
                ('sendgrid', lambda: process_sendgrid_reports('sendgrid', start_date, end_date, conn)),
                ('sendgrid2', lambda: process_sendgrid_reports('sendgrid2', start_date, end_date, conn)),
            ]:
                try:
                    count = processor()
                    results[provider_key] = count
                    total += count
                except Exception as provider_err:
                    logger.warning("[SyncTask] %s sync failed (non-fatal): %s", provider_key, provider_err)
                    results[provider_key] = 0
        finally:
            conn.close()

        elapsed = time.time() - _start
        logger.info(
            "[SyncTask] Provider reports sync DONE in %.1fs — %d total reports synced %s",
            elapsed, total, results,
        )
        return {
            "status": "success",
            "elapsed_seconds": round(elapsed, 2),
            "total_synced": total,
            "start_date": start_date,
            "end_date": end_date,
            "by_provider": results,
        }
    except Exception as e:
        elapsed = time.time() - _start
        logger.error("[SyncTask] Provider reports sync FAILED in %.1fs: %s", elapsed, e, exc_info=True)
        return {"status": "error", "error": str(e), "elapsed_seconds": round(elapsed, 2)}


# ---------------------------------------------------------------------------
# S3 bidirectional storage sync (pull + push — identical copies)
# ---------------------------------------------------------------------------

def s3_storage_sync(mode: str = "sync") -> Dict[str, Any]:
    """
    Full bidirectional S3 storage sync ensuring local and remote are identical.
    Wraps scripts/storage/s3_storage_sync.py (run_cycle(mode)); scheduled by
    Celery Beat every 45 seconds (sync = pull + push), not in the web app.

    Modes:
      pull  — S3 → local  (templates, datasets, resumes)
      push  — local → S3  (models, vectorstore, backups, pictures, resumes)
      sync  — bidirectional: pull where remote has more, push where local has more
    """
    _start = time.time()
    logger.debug("[SyncTask] S3 storage sync START (mode=%s)", mode)
    try:
        script_dir = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "storage"
        import importlib.util
        spec = importlib.util.spec_from_file_location("s3_storage_sync", str(script_dir / "s3_storage_sync.py"))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot find s3_storage_sync.py in {script_dir}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        result = mod.run_cycle(mode)
        elapsed = time.time() - _start

        total_synced = sum(
            v.get("synced", 0) for v in result.values() if isinstance(v, dict)
        )
        total_failed = sum(
            v.get("failed", 0) for v in result.values() if isinstance(v, dict)
        )
        if total_synced > 0 or total_failed > 0:
            logger.info(
                "[SyncTask] S3 storage sync DONE in %.1fs — mode=%s synced=%d failed=%d details=%s",
                elapsed, mode, total_synced, total_failed, result,
            )
        else:
            logger.debug(
                "[SyncTask] S3 storage sync DONE in %.1fs — nothing to sync (mode=%s)",
                elapsed, mode,
            )
        return {
            "status": "success",
            "mode": mode,
            "elapsed_seconds": round(elapsed, 2),
            "total_synced": total_synced,
            "total_failed": total_failed,
            "details": result,
        }
    except Exception as e:
        elapsed = time.time() - _start
        logger.error("[SyncTask] S3 storage sync FAILED in %.1fs: %s", elapsed, e, exc_info=True)
        return {"status": "error", "mode": mode, "error": str(e), "elapsed_seconds": round(elapsed, 2)}


# ---------------------------------------------------------------------------
# Register with Celery
# ---------------------------------------------------------------------------

if celery_app is not None:
    # DISABLED: S3 data sync no longer runs via Celery beat — app pulls once on startup.
    # The function still exists for manual invocation if needed.
    # s3_data_sync = celery_app.task(
    #     name="sync.s3_data_sync",
    #     soft_time_limit=600,
    #     time_limit=900,
    # )(s3_data_sync)

    sales_stats_sync = celery_app.task(
        name="sync.sales_stats_sync",
        soft_time_limit=300,
        time_limit=600,
    )(sales_stats_sync)

    templates_sync_s3 = celery_app.task(
        name="sync.templates_sync_s3",
        soft_time_limit=300,
        time_limit=600,
    )(templates_sync_s3)

    provider_reports_sync = celery_app.task(
        name="sync.provider_reports_sync",
        soft_time_limit=300,
        time_limit=600,
    )(provider_reports_sync)

    # REMOVED: s3_storage_sync was a duplicate of s3_data_sync (same pull logic).
    # Use scripts/storage/s3_model_sync.py for manual model/vectorstore sync.
    # s3_storage_sync = celery_app.task(
    #     name="sync.s3_storage_sync",
    #     soft_time_limit=120,
    #     time_limit=180,
    # )(s3_storage_sync)
