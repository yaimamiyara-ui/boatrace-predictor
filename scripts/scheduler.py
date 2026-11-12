"""時計役（Claude実装）。GitHubの定時起動（schedule）がこのリポジトリで動かないため、その代わりに
1本のジョブが時刻を見て各ワークフローを呼び出す。約5時間40分たったら次の時計役を呼んで終了する。

呼び出す時刻（日本時間）:
  05:17 backfill-auto（過去データの取りこぼし確認）
  07:21 daily-model settle（前日分の答え合わせ）
  07:41 live-record schedule（締切時刻）
  07:51 daily-model predict（当日の予想）
  08:03〜23:53 の10分おき live-record prerace（締切前オッズ・直前情報・推奨）
  23:30 daily-fetch（前日分の公式データ）
  00:21 live-record final（前日の確定オッズ）
"""
import os
import time
from datetime import datetime, timedelta, timezone

import requests

JST = timezone(timedelta(hours=9))
REPO = os.environ["GITHUB_REPOSITORY"]
TOKEN = os.environ["GITHUB_TOKEN"]
RUN_SECONDS = int(os.environ.get("SCHEDULER_RUN_SECONDS", str(5 * 3600 + 40 * 60)))
CATCHUP_MIN = 5  # 起動直後は、直前5分以内に過ぎた予定も実行する（交代のすき間を埋める）


def fixed_slots():
    """(時, 分, ワークフロー, inputs)"""
    s = [
        (5, 17, "backfill-auto.yml", {}),
        (7, 21, "daily-model.yml", {"mode": "settle", "date": ""}),
        (7, 41, "live-record.yml", {"mode": "schedule", "date": ""}),
        (7, 51, "daily-model.yml", {"mode": "predict", "date": ""}),
        (23, 30, "daily-fetch.yml", {"target_date": ""}),
        (0, 21, "live-record.yml", {"mode": "final", "date": "YESTERDAY"}),
    ]
    for h in range(8, 24):
        for m in (3, 13, 23, 33, 43, 53):
            s.append((h, m, "live-record.yml", {"mode": "prerace", "date": ""}))
    return s


def dispatch(workflow: str, inputs: dict) -> None:
    url = f"https://api.github.com/repos/{REPO}/actions/workflows/{workflow}/dispatches"
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"},
        json={"ref": "main", "inputs": inputs},
        timeout=30,
    )
    print(f"[SCHED] {datetime.now(JST):%H:%M} dispatch {workflow} {inputs} -> {r.status_code}", flush=True)


def resolve(inputs: dict, now: datetime) -> dict:
    out = dict(inputs)
    if out.get("date") == "YESTERDAY":
        out["date"] = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    return out


def main() -> None:
    started = time.monotonic()
    start_now = datetime.now(JST)
    done = set()  # (日付, 時, 分, ワークフロー, mode)
    slots = fixed_slots()
    print(f"[SCHED] start {start_now:%Y-%m-%d %H:%M} JST", flush=True)

    while time.monotonic() - started < RUN_SECONDS:
        now = datetime.now(JST)
        for h, m, wf, inp in slots:
            at = now.replace(hour=h, minute=m, second=0, microsecond=0)
            # 予定時刻を過ぎてから5分以内で、まだ呼んでいないものを呼ぶ
            # （起動直後は交代のすき間に過ぎた予定も拾う。重複しても各処理は同じ結果になるよう作ってある）
            if at > now or (now - at) > timedelta(minutes=CATCHUP_MIN):
                continue
            key = (at.date(), h, m, wf, inp.get("mode"))
            if key in done:
                continue
            done.add(key)
            dispatch(wf, resolve(inp, now))
        time.sleep(20)

    # 次の時計役へ交代
    dispatch("scheduler.yml", {})
    print("[SCHED] handed over", flush=True)


if __name__ == "__main__":
    main()
