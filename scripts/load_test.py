import argparse
import concurrent.futures as cf
import json
import statistics
import time
from typing import Callable, Dict, List, Tuple

import requests


def login(base_url: str, email: str, password: str, timeout: float) -> str:
    url = f"{base_url}/api/auth/login"
    data = {
        "username": email,
        "password": password,
    }
    resp = requests.post(url, data=data, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Login failed ({resp.status_code}): {resp.text[:200]}")
    payload = resp.json()
    token = payload.get("access_token")
    if not token:
        raise RuntimeError("Login succeeded but no access_token was returned.")
    return token


def build_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def health_request(base_url: str, headers: Dict[str, str], timeout: float) -> requests.Response:
    return requests.get(f"{base_url}/api/health", headers=headers, timeout=timeout)


def sentence_request(base_url: str, headers: Dict[str, str], timeout: float) -> requests.Response:
    body = {"gloss": ["hello", "how", "are", "you"]}
    return requests.post(
        f"{base_url}/api/translate/sentence",
        headers=headers,
        data=json.dumps(body),
        timeout=timeout,
    )


def signs_request(base_url: str, headers: Dict[str, str], timeout: float) -> requests.Response:
    body = {"words": ["hello", "thank", "you"]}
    return requests.post(
        f"{base_url}/api/signs/batch",
        headers=headers,
        data=json.dumps(body),
        timeout=timeout,
    )


def translate_request(base_url: str, headers: Dict[str, str], timeout: float) -> requests.Response:
    # 30-frame synthetic payload, shape [N_frames, 543, 3].
    frame = [[0.0, 0.0, 0.0] for _ in range(543)]
    body = {"frames": [frame for _ in range(30)]}
    return requests.post(
        f"{base_url}/api/translate",
        headers=headers,
        data=json.dumps(body),
        timeout=timeout,
    )


def run_one(
    request_fn: Callable[[str, Dict[str, str], float], requests.Response],
    base_url: str,
    headers: Dict[str, str],
    timeout: float,
) -> Tuple[bool, float, int, str]:
    start = time.perf_counter()
    try:
        resp = request_fn(base_url, headers, timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        ok = 200 <= resp.status_code < 300
        err = "" if ok else resp.text[:160]
        return ok, elapsed_ms, resp.status_code, err
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return False, elapsed_ms, 0, str(exc)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    k = (len(vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return vals[f]
    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return d0 + d1


def benchmark(
    request_fn: Callable[[str, Dict[str, str], float], requests.Response],
    base_url: str,
    headers: Dict[str, str],
    concurrency: int,
    total_requests: int,
    timeout: float,
) -> None:
    latencies: List[float] = []
    codes: Dict[int, int] = {}
    errors: List[str] = []

    wall_start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_one, request_fn, base_url, headers, timeout)
            for _ in range(total_requests)
        ]
        for fut in cf.as_completed(futures):
            ok, ms, code, err = fut.result()
            latencies.append(ms)
            codes[code] = codes.get(code, 0) + 1
            if not ok and err:
                errors.append(err)

    wall_seconds = max(time.perf_counter() - wall_start, 0.001)

    success_count = sum(v for k, v in codes.items() if 200 <= k < 300)
    fail_count = total_requests - success_count
    rps = total_requests / wall_seconds

    print("\n=== Load Test Summary ===")
    print(f"Total requests:   {total_requests}")
    print(f"Concurrency:      {concurrency}")
    print(f"Duration:         {wall_seconds:.2f}s")
    print(f"Throughput:       {rps:.2f} req/s")
    print(f"Success:          {success_count}")
    print(f"Failures:         {fail_count}")
    print(f"Latency p50:      {percentile(latencies, 0.50):.1f} ms")
    print(f"Latency p90:      {percentile(latencies, 0.90):.1f} ms")
    print(f"Latency p95:      {percentile(latencies, 0.95):.1f} ms")
    print(f"Latency p99:      {percentile(latencies, 0.99):.1f} ms")
    print(f"Latency avg:      {statistics.mean(latencies):.1f} ms")
    print(f"Latency max:      {max(latencies):.1f} ms")

    print("\nStatus code breakdown:")
    for code in sorted(codes):
        label = "EXC" if code == 0 else str(code)
        print(f"  {label}: {codes[code]}")

    if errors:
        print("\nSample errors:")
        for line in errors[:5]:
            print(f"  - {line}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple authenticated load test for ASL Connect API.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the running app")
    parser.add_argument("--email", required=True, help="User email for auth")
    parser.add_argument("--password", required=True, help="User password for auth")
    parser.add_argument(
        "--profile",
        choices=["health", "sentence", "signs", "translate"],
        default="health",
        help="Endpoint profile to test",
    )
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--requests", type=int, default=200, help="Total request count")
    parser.add_argument("--timeout", type=float, default=15.0, help="Per-request timeout in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    request_map: Dict[str, Callable[[str, Dict[str, str], float], requests.Response]] = {
        "health": health_request,
        "sentence": sentence_request,
        "signs": signs_request,
        "translate": translate_request,
    }

    print("Logging in...")
    token = login(args.base_url, args.email, args.password, args.timeout)
    headers = build_headers(token)

    print(
        f"Running profile='{args.profile}' with concurrency={args.concurrency}, "
        f"requests={args.requests}"
    )

    benchmark(
        request_fn=request_map[args.profile],
        base_url=args.base_url,
        headers=headers,
        concurrency=args.concurrency,
        total_requests=args.requests,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
