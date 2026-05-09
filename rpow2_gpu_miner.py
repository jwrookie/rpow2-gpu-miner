#!/usr/bin/env python3
"""rpow2 GPU miner.

Mines rpow2 SHA-256 proof-of-work on any Vulkan-capable GPU (AMD, NVIDIA,
Intel, integrated). Uses Taichi to JIT-compile a SPIR-V compute shader from
the Python source — no separate driver toolchain required.

Quick start:

    pip install -r requirements.txt
    export RPOW_COOKIE='rpow_session=...'   # paste from your browser DevTools
    python rpow2_gpu_miner.py

Run forever, or stop after N tokens with `--rounds N`. See `--help` for all
flags.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import taichi as ti

# --------------------------------------------------------------------------
# rpow2 API constants. Override via env if you're testing against a fork.
# --------------------------------------------------------------------------
API_BASE = os.environ.get("RPOW_API_BASE", "https://api.rpow2.com")
ORIGIN = os.environ.get("RPOW_ORIGIN", "https://rpow2.com")
USER_AGENT = "rpow2-gpu-miner/1.0 (+https://github.com/ImMike/rpow2-gpu-miner)"


# --------------------------------------------------------------------------
# HTTP helper. Stdlib only, no `requests` dependency.
# --------------------------------------------------------------------------
class ApiError(Exception):
    def __init__(self, status, body):
        super().__init__(f"http {status}: {body}")
        self.status = status
        self.body = body


def http(method: str, path: str, cookie: str, body=None, timeout: float = 60.0):
    headers = {
        "cookie": cookie,
        "origin": ORIGIN,
        "user-agent": USER_AGENT,
        "accept": "application/json",
    }
    data = None
    if body is not None:
        headers["content-type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}{path}", data=data, method=method, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"raw": raw}
        raise ApiError(e.code, parsed) from None


# --------------------------------------------------------------------------
# Taichi / Vulkan kernel.
#
# rpow2 PoW spec (matches the on-site browser worker):
#   preimage = nonce_prefix_bytes || little-endian uint64 nonce
#   accept iff trailing_zero_bits(SHA-256(preimage)) >= difficulty_bits
#
# `nonce_prefix` is 16 bytes (32 hex chars) issued by POST /challenge.
# Total preimage is 24 bytes; the SHA-256 padded message fits in one block.
# --------------------------------------------------------------------------
ti.init(arch=ti.vulkan, log_level=ti.WARN)

_K_TABLE = (
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
)

@ti.func
def _rotr(x: ti.u32, n: ti.i32) -> ti.u32:
    return (x >> n) | (x << (32 - n))


@ti.func
def _bswap32(x: ti.u32) -> ti.u32:
    return ((x & ti.u32(0xff)) << 24) \
        | (((x >> 8) & ti.u32(0xff)) << 16) \
        | (((x >> 16) & ti.u32(0xff)) << 8) \
        | ((x >> 24) & ti.u32(0xff))


@ti.func
def _pow_match(
    p0: ti.u32, p1: ti.u32, p2: ti.u32, p3: ti.u32,
    nonce_lo: ti.u32, nonce_hi: ti.u32,
    bit_mask: ti.u32,
) -> ti.u32:
    W = ti.Vector.zero(ti.u32, 16)
    W[0] = p0
    W[1] = p1
    W[2] = p2
    W[3] = p3
    W[4] = _bswap32(nonce_lo)
    W[5] = _bswap32(nonce_hi)
    W[6] = ti.u32(0x80000000)  # SHA-256 padding marker
    W[15] = ti.u32(192)        # bit length: 24 bytes = 192 bits

    a = ti.u32(0x6A09E667); b = ti.u32(0xBB67AE85)
    c = ti.u32(0x3C6EF372); d = ti.u32(0xA54FF53A)
    e = ti.u32(0x510E527F); f = ti.u32(0x9B05688C)
    g = ti.u32(0x1F83D9AB); h = ti.u32(0x5BE0CD19)

    for t in ti.static(range(64)):
        wt = ti.u32(0)
        if ti.static(t < 16):
            wt = W[t]
        else:
            s0 = _rotr(W[(t - 15) & 15], 7) \
                ^ _rotr(W[(t - 15) & 15], 18) \
                ^ (W[(t - 15) & 15] >> 3)
            s1 = _rotr(W[(t - 2) & 15], 17) \
                ^ _rotr(W[(t - 2) & 15], 19) \
                ^ (W[(t - 2) & 15] >> 10)
            wt = W[(t - 16) & 15] + s0 + W[(t - 7) & 15] + s1
            W[t & 15] = wt

        S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
        ch = (e & f) ^ ((~e) & g)
        temp1 = h + S1 + ch + ti.u32(_K_TABLE[t]) + wt
        S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
        mj = (a & b) ^ (a & c) ^ (b & c)
        temp2 = S0 + mj
        h = g; g = f; f = e
        e = d + temp1
        d = c; c = b; b = a
        a = temp1 + temp2

    h7 = ti.u32(0x5BE0CD19) + h
    return ti.cast((h7 & bit_mask) == ti.u32(0), ti.u32)


@ti.kernel
def mine_kernel(
    n_threads: ti.i32,
    iters: ti.i32,
    base_nonce_lo: ti.u32,
    base_nonce_hi: ti.u32,
    p0: ti.u32, p1: ti.u32, p2: ti.u32, p3: ti.u32,
    bit_mask: ti.u32,
    result: ti.types.ndarray(dtype=ti.u32, ndim=1),  # [found, nonce_lo, nonce_hi]
):
    for gid in range(n_threads):
        local_base = base_nonce_lo + ti.u32(gid) * ti.u32(iters)
        local_carry = ti.cast(local_base < base_nonce_lo, ti.u32)
        for k in range(iters):
            nonce_lo = local_base + ti.u32(k)
            nonce_hi = base_nonce_hi + local_carry \
                + ti.cast(nonce_lo < local_base, ti.u32)
            if _pow_match(p0, p1, p2, p3, nonce_lo, nonce_hi, bit_mask) == ti.u32(1):
                prev = ti.atomic_or(result[0], ti.u32(1))
                if prev == ti.u32(0):
                    result[1] = nonce_lo
                    result[2] = nonce_hi


@ti.kernel
def mine_batch_kernel(
    n_jobs: ti.i32,
    n_threads: ti.i32,
    iters: ti.i32,
    base_nonce_lo: ti.u32,
    base_nonce_hi: ti.u32,
    p0s: ti.types.ndarray(dtype=ti.u32, ndim=1),
    p1s: ti.types.ndarray(dtype=ti.u32, ndim=1),
    p2s: ti.types.ndarray(dtype=ti.u32, ndim=1),
    p3s: ti.types.ndarray(dtype=ti.u32, ndim=1),
    bit_masks: ti.types.ndarray(dtype=ti.u32, ndim=1),
    result: ti.types.ndarray(dtype=ti.u32, ndim=1),  # per job: [found, lo, hi]
):
    for global_gid in range(n_jobs * n_threads):
        job = global_gid // n_threads
        result_offset = job * 3
        if result[result_offset] == ti.u32(0):
            gid = global_gid - job * n_threads
            local_base = base_nonce_lo + ti.u32(gid) * ti.u32(iters)
            local_carry = ti.cast(local_base < base_nonce_lo, ti.u32)
            for k in range(iters):
                nonce_lo = local_base + ti.u32(k)
                nonce_hi = base_nonce_hi + local_carry \
                    + ti.cast(nonce_lo < local_base, ti.u32)
                if _pow_match(
                    p0s[job], p1s[job], p2s[job], p3s[job],
                    nonce_lo, nonce_hi, bit_masks[job],
                ) == ti.u32(1):
                    prev = ti.atomic_or(result[result_offset], ti.u32(1))
                    if prev == ti.u32(0):
                        result[result_offset + 1] = nonce_lo
                        result[result_offset + 2] = nonce_hi


def _hex_prefix_to_uint32_be(prefix_hex: str):
    pb = bytes.fromhex(prefix_hex)
    if len(pb) != 16:
        raise ValueError(f"expected 16-byte prefix, got {len(pb)}")
    return tuple(int.from_bytes(pb[i:i + 4], "big") for i in range(0, 16, 4))


def _trailing_zero_bits(d: bytes) -> int:
    z = 0
    for byte in reversed(d):
        if byte == 0:
            z += 8
            continue
        c = 0
        while not (byte & (1 << c)):
            c += 1
        return z + c
    return z


def verify(prefix_hex: str, nonce: int, target_bits: int) -> bool:
    """Sanity-check that a (prefix, nonce) really meets the target."""
    msg = bytes.fromhex(prefix_hex) + nonce.to_bytes(8, "little")
    return _trailing_zero_bits(hashlib.sha256(msg).digest()) >= target_bits


def _nonce_parts(nonce: int):
    return np.uint32(nonce & 0xFFFFFFFF), np.uint32((nonce >> 32) & 0xFFFFFFFF)


def _nonce_from_parts(lo, hi) -> int:
    return int(lo) | (int(hi) << 32)


def _batch_attempts(n_threads: int, iters: int) -> int:
    attempts = n_threads * iters
    if attempts > (1 << 32):
        raise ValueError(
            "threads * iters must be <= 2^32 so one kernel launch does not "
            "wrap and repeat nonce offsets"
        )
    return attempts


def solve(prefix_hex, target_bits, n_threads, iters, attempt_cap):
    """Return (winning_nonce, total_attempts). Raises RuntimeError on cap."""
    if not (1 <= target_bits <= 32):
        raise ValueError(f"target_bits must be 1..32, got {target_bits}")
    p0, p1, p2, p3 = _hex_prefix_to_uint32_be(prefix_hex)
    bit_mask = np.uint32((1 << target_bits) - 1)
    batch_attempts = _batch_attempts(n_threads, iters)
    base = 0
    total = 0
    while total < attempt_cap:
        base_lo, base_hi = _nonce_parts(base)
        result_buf = np.zeros(3, dtype=np.uint32)
        mine_kernel(
            n_threads, iters, base_lo, base_hi,
            np.uint32(p0), np.uint32(p1), np.uint32(p2), np.uint32(p3),
            bit_mask, result_buf,
        )
        ti.sync()
        total += batch_attempts
        if int(result_buf[0]) == 1:
            nonce = _nonce_from_parts(result_buf[1], result_buf[2])
            if not verify(prefix_hex, nonce, target_bits):
                raise RuntimeError(
                    f"kernel returned nonce={nonce} that does not verify on CPU; "
                    "this should never happen — please file a bug"
                )
            return nonce, total
        base = (base + batch_attempts) & 0xFFFFFFFFFFFFFFFF
    raise RuntimeError(f"attempt_cap reached after {total} hashes")


def solve_batch(challenges, n_threads, iters, attempt_cap):
    """Solve multiple challenge dicts in GPU batches.

    Returns a list of (winning_nonce, total_attempts) matching input order.
    """
    if not challenges:
        return []

    p_words = []
    masks = []
    for ch in challenges:
        bits = int(ch["difficulty_bits"])
        if not (1 <= bits <= 32):
            raise ValueError(f"target_bits must be 1..32, got {bits}")
        p_words.append(_hex_prefix_to_uint32_be(ch["nonce_prefix"]))
        masks.append(np.uint32((1 << bits) - 1))

    base = 0
    batch_attempts = _batch_attempts(n_threads, iters)
    attempts = [0] * len(challenges)
    solutions = [None] * len(challenges)
    active = list(range(len(challenges)))

    while active:
        base_lo, base_hi = _nonce_parts(base)
        p0s = np.array([p_words[i][0] for i in active], dtype=np.uint32)
        p1s = np.array([p_words[i][1] for i in active], dtype=np.uint32)
        p2s = np.array([p_words[i][2] for i in active], dtype=np.uint32)
        p3s = np.array([p_words[i][3] for i in active], dtype=np.uint32)
        bit_masks = np.array([masks[i] for i in active], dtype=np.uint32)
        result_buf = np.zeros(len(active) * 3, dtype=np.uint32)

        mine_batch_kernel(
            len(active), n_threads, iters, base_lo, base_hi,
            p0s, p1s, p2s, p3s, bit_masks, result_buf,
        )
        ti.sync()

        next_active = []
        for pos, original_idx in enumerate(active):
            attempts[original_idx] += batch_attempts
            offset = pos * 3
            if int(result_buf[offset]) == 1:
                nonce = _nonce_from_parts(
                    result_buf[offset + 1], result_buf[offset + 2],
                )
                ch = challenges[original_idx]
                if not verify(ch["nonce_prefix"], nonce, int(ch["difficulty_bits"])):
                    raise RuntimeError(
                        f"kernel returned nonce={nonce} that does not verify "
                        "on CPU; this should never happen — please file a bug"
                    )
                solutions[original_idx] = (nonce, attempts[original_idx])
            elif attempts[original_idx] >= attempt_cap:
                cid = challenges[original_idx].get("challenge_id", original_idx)
                raise RuntimeError(
                    f"attempt_cap reached for challenge {cid} "
                    f"after {attempts[original_idx]} hashes"
                )
            else:
                next_active.append(original_idx)

        active = next_active
        base = (base + batch_attempts) & 0xFFFFFFFFFFFFFFFF

    return solutions


def fetch_challenge(cookie: str):
    _, ch = http("POST", "/challenge", cookie)
    return ch


def mint_solution(cookie: str, challenge, nonce: int):
    _, minted = http(
        "POST", "/mint", cookie,
        {
            "challenge_id": challenge["challenge_id"],
            "solution_nonce": str(nonce),
        },
    )
    return minted


def warm_up(n_threads: int, iters: int, pipeline: int = 1):
    if pipeline > 1:
        warm_result = np.zeros(3, dtype=np.uint32)
        warm_words = np.zeros(1, dtype=np.uint32)
        warm_mask = np.array([0xFFFFFFFF], dtype=np.uint32)
        mine_batch_kernel(
            1, n_threads, iters, np.uint32(0), np.uint32(0),
            warm_words, warm_words, warm_words, warm_words,
            warm_mask, warm_result,
        )
    else:
        warm_result = np.zeros(3, dtype=np.uint32)
        mine_kernel(
            n_threads, iters, np.uint32(0), np.uint32(0),
            np.uint32(0), np.uint32(0), np.uint32(0), np.uint32(0),
            np.uint32(0xFFFFFFFF), warm_result,
        )
    ti.sync()


def benchmark(n_threads: int, iters: int, seconds: float):
    print("compiling SPIR-V kernel (first launch only)...", file=sys.stderr, flush=True)
    warm_up(n_threads, iters)
    print("kernel ready.\n", file=sys.stderr, flush=True)

    batch_attempts = _batch_attempts(n_threads, iters)
    base = 0
    hashes = 0
    launches = 0
    result_buf = np.zeros(3, dtype=np.uint32)
    started = time.perf_counter()
    deadline = started + seconds
    while time.perf_counter() < deadline:
        result_buf.fill(0)
        base_lo, base_hi = _nonce_parts(base)
        mine_kernel(
            n_threads, iters, base_lo, base_hi,
            np.uint32(0), np.uint32(0), np.uint32(0), np.uint32(0),
            np.uint32(0xFFFFFFFF), result_buf,
        )
        ti.sync()
        launches += 1
        hashes += batch_attempts
        base = (base + batch_attempts) & 0xFFFFFFFFFFFFFFFF

    elapsed = time.perf_counter() - started
    rate = hashes / elapsed if elapsed > 0 else 0.0
    print(f"benchmark: {hashes:,} hashes in {elapsed:.2f}s")
    print(f"throughput: {rate/1e6:.1f} MH/s  ({rate/1e9:.3f} GH/s)")
    print(f"launches: {launches}  batch={batch_attempts:,} hashes")


# --------------------------------------------------------------------------
# Live mining loop.
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="GPU-accelerated rpow2 miner (Vulkan/Taichi).",
    )
    p.add_argument(
        "--cookie",
        default=os.environ.get("RPOW_COOKIE"),
        help="rpow2 session cookie. Format: 'rpow_session=<value>'. "
             "Defaults to $RPOW_COOKIE.",
    )
    p.add_argument("--rounds", type=int, default=0,
                   help="stop after N successful mints (0 = run forever)")
    p.add_argument("--threads", type=int, default=1 << 20,
                   help="GPU threads per kernel launch (default: 1048576)")
    p.add_argument("--iters", type=int, default=64,
                   help="nonces per thread per launch (default: 64)")
    p.add_argument("--pipeline", type=int, default=1,
                   help="solve this many outstanding challenges per GPU batch "
                        "(default: 1; try 4-16 if HTTP latency leaves the GPU idle)")
    p.add_argument("--attempt-cap", type=int, default=1 << 36,
                   help="abort a single PoW after this many attempts (safety)")
    p.add_argument("--benchmark", type=float, default=0.0,
                   help="run a local GPU benchmark for N seconds and exit "
                        "(does not need a cookie)")
    p.add_argument("--quiet", action="store_true",
                   help="only print summary, no per-mint lines")
    args = p.parse_args()

    if args.threads <= 0 or args.iters <= 0:
        sys.exit("--threads and --iters must be positive")
    if args.pipeline <= 0:
        sys.exit("--pipeline must be positive")
    try:
        _batch_attempts(args.threads, args.iters)
    except ValueError as e:
        sys.exit(str(e))

    if args.benchmark > 0:
        benchmark(args.threads, args.iters, args.benchmark)
        return

    if not args.cookie:
        sys.exit(
            "no cookie supplied. set $RPOW_COOKIE or pass --cookie. "
            "tip: in your browser, DevTools → Network → click any /api request "
            "→ copy the 'cookie' header (starts with 'rpow_session=')."
        )
    if not args.cookie.startswith("rpow_session="):
        sys.exit("cookie does not start with 'rpow_session='. paste the full "
                 "header value, not just the JWT-looking part.")

    # Confirm auth before warming up the GPU.
    try:
        status, me = http("GET", "/me", args.cookie)
    except ApiError as e:
        sys.exit(f"auth check failed: {e}")
    if status != 200 or not me or "email" not in me:
        sys.exit(f"unexpected /me response: {me}")
    print(f"signed in: {me['email']}  balance={me['balance']}  minted={me['minted']}",
          file=sys.stderr, flush=True)

    print("compiling SPIR-V kernel (first launch only)...", file=sys.stderr, flush=True)
    warm_up(args.threads, args.iters, args.pipeline)
    print("kernel ready.\n", file=sys.stderr, flush=True)

    minted = 0
    failures = 0
    started_at = time.time()

    def stop_summary(*_):
        elapsed = time.time() - started_at
        print(file=sys.stderr)
        print("---- summary ----", file=sys.stderr)
        print(f"minted:    {minted}", file=sys.stderr)
        print(f"failures:  {failures}", file=sys.stderr)
        print(f"elapsed:   {elapsed:.1f}s", file=sys.stderr)
        if elapsed > 0:
            print(f"avg rate:  {minted/elapsed:.2f} tokens/sec  "
                  f"(~{minted/elapsed*3600:.0f}/hour)", file=sys.stderr)
        sys.exit(0)

    signal.signal(signal.SIGINT,  stop_summary)
    signal.signal(signal.SIGTERM, stop_summary)

    while True:
        if args.rounds and minted >= args.rounds:
            break

        if args.pipeline > 1:
            want = args.pipeline
            if args.rounds:
                want = min(want, args.rounds - minted)

            try:
                with ThreadPoolExecutor(max_workers=want) as pool:
                    futures = [pool.submit(fetch_challenge, args.cookie)
                               for _ in range(want)]
                    challenges = [future.result() for future in futures]
            except ApiError as e:
                failures += 1
                print(f"[!] /challenge failed: {e}", file=sys.stderr, flush=True)
                time.sleep(1.0)
                continue

            t0 = time.time()
            try:
                solutions = solve_batch(
                    challenges, args.threads, args.iters, args.attempt_cap,
                )
            except RuntimeError as e:
                failures += 1
                print(f"[!] batch solve failed: {e}", file=sys.stderr, flush=True)
                continue
            solve_ms = (time.time() - t0) * 1000.0

            with ThreadPoolExecutor(max_workers=want) as pool:
                future_map = {
                    pool.submit(mint_solution, args.cookie, ch, nonce):
                    (ch, nonce, attempts)
                    for ch, (nonce, attempts) in zip(challenges, solutions)
                }
                for future in as_completed(future_map):
                    ch, _nonce, attempts = future_map[future]
                    try:
                        m = future.result()
                    except ApiError as e:
                        failures += 1
                        print(
                            f"[!] /mint failed (challenge {ch['challenge_id']}): {e}",
                            file=sys.stderr, flush=True,
                        )
                        continue

                    minted += 1
                    token_id = (m or {}).get("token", {}).get("id", "?")
                    if not args.quiet:
                        print(
                            f"minted #{minted:<5d}  bits={ch['difficulty_bits']}  "
                            f"batch_solve={solve_ms:>5.0f}ms  "
                            f"attempts={attempts:>10,}  token={token_id}",
                            flush=True,
                        )
            continue

        try:
            ch = fetch_challenge(args.cookie)
        except ApiError as e:
            failures += 1
            print(f"[!] /challenge failed: {e}", file=sys.stderr, flush=True)
            time.sleep(1.0)
            continue

        cid    = ch["challenge_id"]
        prefix = ch["nonce_prefix"]
        bits   = ch["difficulty_bits"]

        t0 = time.time()
        try:
            nonce, attempts = solve(
                prefix, bits, args.threads, args.iters, args.attempt_cap,
            )
        except RuntimeError as e:
            failures += 1
            print(f"[!] solve failed for challenge {cid}: {e}",
                  file=sys.stderr, flush=True)
            continue
        solve_ms = (time.time() - t0) * 1000.0

        try:
            m = mint_solution(args.cookie, ch, nonce)
        except ApiError as e:
            failures += 1
            print(f"[!] /mint failed (challenge {cid}): {e}",
                  file=sys.stderr, flush=True)
            continue

        minted += 1
        token_id = (m or {}).get("token", {}).get("id", "?")
        if not args.quiet:
            print(
                f"minted #{minted:<5d}  bits={bits}  solve={solve_ms:>5.0f}ms  "
                f"attempts={attempts:>10,}  token={token_id}",
                flush=True,
            )

    stop_summary()


if __name__ == "__main__":
    main()
