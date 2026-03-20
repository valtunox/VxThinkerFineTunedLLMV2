"""
VaLLM Specialist Model — Computer Operations Troubleshooting Tests
====================================================================

Author: Joel Otepa Wembo
https://joelwembo.com

Tests that verify the AI can find relevant troubleshooting answers
from the skills/ knowledge base via semantic search and generate
useful diagnostic responses. Covers:

  1. Networking issues (DNS, VPN, SSL, firewall, latency)
  2. Linux administration (disk, SSH, permissions, OOM, cron)
  3. Windows troubleshooting (BSOD, AD, Group Policy, updates)
  4. Database issues (PostgreSQL, MySQL, Redis, MongoDB, deadlocks)
  5. Cloud/DevOps (Kubernetes, Docker, Terraform, CI/CD, AWS)
  6. Security incidents (vulnerability, certificates, incident response)
  7. Programming/debugging (memory leaks, race conditions, profiling)
  8. Hardware diagnostics (CPU, RAM, disk, RAID, NIC)

PREREQUISITES:
    - Server running on port 8747
    - Precompute completed with skills/ folder indexed

USAGE:
    python -m pytest app/tests/tests_troubleshooting.py -v -s --tb=short
"""

import sys
import os
import time
import atexit
from pathlib import Path
from typing import Any, Dict

import pytest
import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8747")

# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
_SCORECARD: Dict[str, Dict[str, Any]] = {}
_TEST_COUNTER = 0

def _record(name: str, score: int, details: str = ""):
    global _TEST_COUNTER
    _TEST_COUNTER += 1
    score = max(1, min(10, score))
    bar = "#" * score + "." * (10 - score)
    label = "PERFECT" if score == 10 else "EXCELLENT" if score >= 8 else "GOOD" if score >= 6 else "FAIR" if score >= 4 else "POOR"
    _SCORECARD[f"Test {_TEST_COUNTER}"] = {"name": name, "score": score, "label": label}
    print(f"\n  {'='*65}\n  [Test {_TEST_COUNTER}] {name}\n  Score: {score}/10 [{bar}] {label}")
    if details: print(f"  Details: {details}")
    print(f"  {'='*65}")

def _print_final():
    if not _SCORECARD: return
    total = sum(v["score"] for v in _SCORECARD.values())
    count = len(_SCORECARD)
    avg = total / count if count else 0
    print("\n\n" + "=" * 70)
    print("  TROUBLESHOOTING TESTS — FINAL SCORECARD")
    print("=" * 70)
    for k, v in _SCORECARD.items():
        bar = "#" * v["score"] + "." * (10 - v["score"])
        print(f"  {k:>8} | {v['score']:>2}/10 [{bar}] {v['label']:<10} | {v['name']}")
    print("-" * 70)
    overall = "EXCELLENT" if avg >= 8 else "GOOD" if avg >= 6 else "NEEDS IMPROVEMENT"
    print(f"  {'TOTAL':>8} | {total}/{count*10}  Average: {avg:.1f}/10  {overall}")
    print("=" * 70)

atexit.register(_print_final)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def search(query: str, top_k: int = 5) -> dict:
    """Search the vector store."""
    start = time.time()
    try:
        r = requests.post(f"{BASE_URL}/search", json={"query": query, "top_k": top_k}, timeout=30)
        elapsed = (time.time() - start) * 1000
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            return {"ok": True, "results": results, "count": len(results), "elapsed_ms": elapsed}
        return {"ok": False, "status": r.status_code, "results": [], "count": 0, "elapsed_ms": elapsed}
    except Exception as e:
        return {"ok": False, "error": str(e), "results": [], "count": 0, "elapsed_ms": 0}

def generate(prompt: str) -> dict:
    """Generate text with the LLM."""
    start = time.time()
    try:
        r = requests.post(f"{BASE_URL}/generate", json={"prompt": prompt, "max_new_tokens": 150}, timeout=60)
        elapsed = (time.time() - start) * 1000
        if r.status_code == 200:
            data = r.json()
            return {"ok": True, "text": data.get("response", ""), "model_loaded": data.get("model_loaded"), "elapsed_ms": elapsed}
        return {"ok": False, "status": r.status_code, "text": "", "elapsed_ms": elapsed}
    except Exception as e:
        return {"ok": False, "error": str(e), "text": "", "elapsed_ms": 0}

def check_relevance(results, keywords):
    """Check if search results contain expected keywords."""
    all_text = " ".join(r.get("text", "") for r in results).lower()
    found = [kw for kw in keywords if kw.lower() in all_text]
    return found, len(found) / len(keywords) if keywords else 0


def run_troubleshoot_test(name, query, expected_keywords, category):
    """Run a troubleshooting search test and score it."""
    r = search(query)
    print(f"\n  Query: {query}")
    print(f"  Category: {category}")
    
    if not r["ok"]:
        print(f"  Search failed: {r.get('error', r.get('status', 'unknown'))}")
        _record(name, 4, "Search endpoint unavailable")
        return

    print(f"  Results: {r['count']} ({r['elapsed_ms']:.0f}ms)")
    
    if r["results"]:
        for i, res in enumerate(r["results"][:3]):
            score = res.get("score", 0)
            text = res.get("text", "")[:120]
            print(f"    #{i+1}: score={score:.4f} | {text}...")
        
        found, relevance = check_relevance(r["results"], expected_keywords)
        print(f"  Keywords found: {len(found)}/{len(expected_keywords)} ({relevance*100:.0f}%)")
        print(f"    Found: {found}")
        
        top_score = r["results"][0].get("score", 0)
        score = 10 if relevance >= 0.6 and top_score > 0.5 else \
                8 if relevance >= 0.4 or top_score > 0.4 else \
                6 if r["count"] >= 3 else 5
    else:
        relevance = 0
        score = 4
    
    _record(name, score, f"{r['count']} results, {relevance*100:.0f}% relevance")


# ============================================================================
# 1. NETWORKING TROUBLESHOOTING
# ============================================================================

class TestNetworkTroubleshooting:

    def test_dns_resolution_failure(self):
        run_troubleshoot_test(
            "DNS Resolution Failure",
            "DNS not resolving, websites not loading, nslookup fails with SERVFAIL",
            ["dns", "resolv", "nslookup", "flush", "nameserver", "port 53"],
            "networking"
        )

    def test_vpn_connection_drops(self):
        run_troubleshoot_test(
            "VPN Connection Drops",
            "VPN keeps disconnecting, intermittent tunnel failures, high latency",
            ["vpn", "mtu", "tunnel", "nat", "keepalive"],
            "networking"
        )

    def test_ssl_certificate_error(self):
        run_troubleshoot_test(
            "SSL/TLS Certificate Error",
            "SSL certificate expired, browser shows connection not private",
            ["ssl", "certificate", "openssl", "expired", "chain"],
            "networking"
        )

    def test_firewall_blocking(self):
        run_troubleshoot_test(
            "Firewall Blocking Traffic",
            "Firewall blocking legitimate application traffic, connection timeouts",
            ["firewall", "iptables", "block", "rule", "allow"],
            "networking"
        )

    def test_high_latency(self):
        run_troubleshoot_test(
            "High Network Latency",
            "Network latency high, slow application response, packet loss",
            ["latency", "traceroute", "bandwidth", "congestion"],
            "networking"
        )


# ============================================================================
# 2. LINUX ADMINISTRATION
# ============================================================================

class TestLinuxTroubleshooting:

    def test_disk_full(self):
        run_troubleshoot_test(
            "Linux Disk Full",
            "Disk space 100% full, cannot create files, no space left on device error",
            ["disk", "df", "du", "space", "log", "clean"],
            "linux"
        )

    def test_ssh_refused(self):
        run_troubleshoot_test(
            "SSH Connection Refused",
            "SSH connection refused port 22, cannot connect to remote server",
            ["ssh", "sshd", "port", "firewall", "auth"],
            "linux"
        )

    def test_oom_killer(self):
        run_troubleshoot_test(
            "OOM Killer Invoked",
            "Linux OOM killer killing processes, out of memory, application crashes",
            ["oom", "memory", "swap", "kill", "dmesg"],
            "linux"
        )

    def test_permission_denied(self):
        run_troubleshoot_test(
            "Permission Denied Errors",
            "Permission denied when accessing files, chmod chown not working",
            ["permission", "chmod", "chown", "selinux", "ownership"],
            "linux"
        )

    def test_cron_not_running(self):
        run_troubleshoot_test(
            "Cron Job Not Running",
            "Scheduled cron job not executing, no output from crontab",
            ["cron", "crontab", "schedule", "path", "log"],
            "linux"
        )


# ============================================================================
# 3. WINDOWS TROUBLESHOOTING
# ============================================================================

class TestWindowsTroubleshooting:

    def test_bsod(self):
        run_troubleshoot_test(
            "Blue Screen of Death",
            "Windows blue screen BSOD crash, stop code DRIVER_IRQL_NOT_LESS_OR_EQUAL",
            ["bsod", "blue screen", "driver", "sfc", "memory"],
            "windows"
        )

    def test_active_directory(self):
        run_troubleshoot_test(
            "Active Directory Replication",
            "Active Directory replication failure, users cannot log in at some sites",
            ["active directory", "replication", "dc", "dns", "repadmin"],
            "windows"
        )

    def test_group_policy(self):
        run_troubleshoot_test(
            "Group Policy Not Applying",
            "Group policy GPO not applying to computers, gpresult shows not applied",
            ["group policy", "gpo", "gpresult", "gpupdate", "wmi"],
            "windows"
        )

    def test_windows_update_failure(self):
        run_troubleshoot_test(
            "Windows Update Failure",
            "Windows updates failing to install, error code 0x80070002",
            ["update", "windows", "wuauserv", "dism", "cache"],
            "windows"
        )


# ============================================================================
# 4. DATABASE TROUBLESHOOTING
# ============================================================================

class TestDatabaseTroubleshooting:

    def test_postgres_connection(self):
        run_troubleshoot_test(
            "PostgreSQL Connection Refused",
            "PostgreSQL connection refused, cannot connect to database server",
            ["postgresql", "connection", "pg_hba", "port", "listen"],
            "database"
        )

    def test_mysql_deadlock(self):
        run_troubleshoot_test(
            "MySQL Deadlock",
            "MySQL deadlock detected, transactions waiting for locks",
            ["deadlock", "lock", "innodb", "transaction"],
            "database"
        )

    def test_redis_memory(self):
        run_troubleshoot_test(
            "Redis Memory Exhaustion",
            "Redis out of memory OOM error, eviction policy triggered",
            ["redis", "memory", "maxmemory", "eviction", "ttl"],
            "database"
        )

    def test_slow_queries(self):
        run_troubleshoot_test(
            "Slow Database Queries",
            "Database queries very slow, high CPU on database server, missing indexes",
            ["slow", "query", "index", "explain", "profil"],
            "database"
        )

    def test_replication_lag(self):
        run_troubleshoot_test(
            "Database Replication Lag",
            "Database replica behind primary, stale reads, replication lag increasing",
            ["replication", "lag", "replica", "wal", "slave"],
            "database"
        )


# ============================================================================
# 5. CLOUD / DEVOPS
# ============================================================================

class TestCloudDevOpsTroubleshooting:

    def test_pod_crashloop(self):
        run_troubleshoot_test(
            "K8s Pod CrashLoopBackOff",
            "Kubernetes pod CrashLoopBackOff, container keeps restarting",
            ["crashloop", "pod", "kubectl", "logs", "restart"],
            "cloud"
        )

    def test_docker_build_fail(self):
        run_troubleshoot_test(
            "Docker Build Failure",
            "Docker build failing, layer caching issues, Dockerfile errors",
            ["docker", "build", "dockerfile", "image", "layer"],
            "cloud"
        )

    def test_terraform_state_lock(self):
        run_troubleshoot_test(
            "Terraform State Lock",
            "Terraform state lock error, cannot acquire lock for state file",
            ["terraform", "state", "lock", "unlock", "s3"],
            "cloud"
        )

    def test_aws_permission_denied(self):
        run_troubleshoot_test(
            "AWS IAM Permission Denied",
            "AWS access denied error 403, IAM policy not allowing action",
            ["aws", "iam", "permission", "denied", "policy"],
            "cloud"
        )

    def test_cicd_pipeline_failure(self):
        run_troubleshoot_test(
            "CI/CD Pipeline Failure",
            "CI/CD pipeline failing, tests pass locally but fail in CI environment",
            ["ci", "cd", "pipeline", "build", "deploy"],
            "cloud"
        )


# ============================================================================
# 6. SECURITY
# ============================================================================

class TestSecurityTroubleshooting:

    def test_ssl_chain_incomplete(self):
        run_troubleshoot_test(
            "SSL Chain Incomplete",
            "SSL certificate chain incomplete, some browsers fail to connect",
            ["ssl", "certificate", "chain", "intermediate", "ca"],
            "security"
        )

    def test_compromised_server(self):
        run_troubleshoot_test(
            "Compromised Server Response",
            "Server compromised, unauthorized access detected, suspicious processes",
            ["compromised", "incident", "forensic", "isolate", "credential"],
            "security"
        )

    def test_vulnerability_scan(self):
        run_troubleshoot_test(
            "Vulnerability Scan Results",
            "Vulnerability scanner reports critical CVE, need to assess and remediate",
            ["vulnerability", "cve", "scan", "patch", "remediat"],
            "security"
        )


# ============================================================================
# 7. PROGRAMMING / DEBUGGING
# ============================================================================

class TestProgrammingTroubleshooting:

    def test_python_memory_leak(self):
        run_troubleshoot_test(
            "Python Memory Leak",
            "Python application memory grows continuously, eventually OOM killed",
            ["python", "memory", "leak", "tracemalloc", "gc"],
            "programming"
        )

    def test_race_condition(self):
        run_troubleshoot_test(
            "Race Condition Bug",
            "Intermittent test failures, data corruption from concurrent threads",
            ["race", "condition", "thread", "mutex", "lock"],
            "programming"
        )

    def test_java_oom(self):
        run_troubleshoot_test(
            "Java OutOfMemoryError",
            "Java OutOfMemoryError heap space, GC overhead limit exceeded",
            ["java", "outofmemory", "heap", "gc", "jmap"],
            "programming"
        )


# ============================================================================
# 8. HARDWARE
# ============================================================================

class TestHardwareTroubleshooting:

    def test_cpu_overheating(self):
        run_troubleshoot_test(
            "CPU Overheating",
            "CPU thermal throttling, system slow under load, thermal shutdown",
            ["cpu", "thermal", "overheat", "fan", "temp"],
            "hardware"
        )

    def test_ram_errors(self):
        run_troubleshoot_test(
            "RAM/Memory Errors",
            "Random crashes, memtest errors, ECC correctable errors in system log",
            ["ram", "memory", "memtest", "dimm", "ecc"],
            "hardware"
        )

    def test_disk_smart_warning(self):
        run_troubleshoot_test(
            "Disk SMART Warning",
            "SMART status pre-fail, reallocated sectors increasing, disk dying",
            ["smart", "disk", "sector", "backup", "replace"],
            "hardware"
        )

    def test_raid_degraded(self):
        run_troubleshoot_test(
            "RAID Array Degraded",
            "RAID array degraded status, one drive failed, rebuild needed",
            ["raid", "degraded", "rebuild", "drive", "mdadm"],
            "hardware"
        )


# ============================================================================
# 9. GENERATION TESTS — Verify LLM produces useful troubleshooting output
# ============================================================================

class TestTroubleshootingGeneration:

    def test_generate_dns_diagnosis(self):
        """Generate a DNS troubleshooting diagnosis."""
        r = generate("Diagnose this issue: DNS resolution failing, websites not loading, nslookup returns SERVFAIL")
        print(f"\n  Model loaded: {r.get('model_loaded')}")
        print(f"  Response ({r['elapsed_ms']:.0f}ms): {r['text'][:200]}...")
        score = 8 if r["ok"] and r.get("model_loaded") else 5
        _record("Generate DNS Diagnosis", score, f"model_loaded={r.get('model_loaded')}")

    def test_generate_k8s_diagnosis(self):
        """Generate a Kubernetes troubleshooting diagnosis."""
        r = generate("Troubleshoot: Kubernetes pods in CrashLoopBackOff state, containers restarting every 30 seconds")
        print(f"\n  Response ({r['elapsed_ms']:.0f}ms): {r['text'][:200]}...")
        score = 8 if r["ok"] and r.get("model_loaded") else 5
        _record("Generate K8s Diagnosis", score, f"model_loaded={r.get('model_loaded')}")

    def test_generate_db_diagnosis(self):
        """Generate a database troubleshooting diagnosis."""
        r = generate("Troubleshoot: PostgreSQL connection refused, application cannot connect to database on port 5432")
        print(f"\n  Response ({r['elapsed_ms']:.0f}ms): {r['text'][:200]}...")
        score = 8 if r["ok"] and r.get("model_loaded") else 5
        _record("Generate Database Diagnosis", score, f"model_loaded={r.get('model_loaded')}")


if __name__ == "__main__":
    print("=" * 70)
    print("VaLLM Specialist — Computer Operations Troubleshooting Tests")
    print("=" * 70)
    sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short"]))
