"""時計役（Claude実装）。1本のジョブが時刻を見て各ワークフローを呼び出す。約5時間40分たったら次の時計役を呼んで終了する。

呼び出す処理（日本時間）。どれもGoogleドライブだけを読み、公式サイトにはアクセスしない:
  07:45ほか  daily-model settle（前日分の答え合わせ）
  07:51ほか  daily-model predict（当日の予想）
  07:36〜    live-record schedule（締切時刻と監視リスト）
  08:43〜23:58 の5分おき live-record prerace（自宅PCが取得済みのレースの記録と推奨）
  23:45, 00:21 live-record final（確定オッズ）
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
    """(時, 分, ワークフロー, inputs)。どの処理もGoogleドライブだけを読み、公式サイトにはアクセスしない。
    自宅PC（pc/fetch_day.ps1）が7:25から生ファイルを置くので、届くのが遅れても拾えるよう何度か呼ぶ（どれも済んでいれば何もしない）"""
    s = []
    for h, m in ((7, 45), (9, 45), (12, 45)):
        s.append((h, m, "daily-model.yml", {"mode": "settle", "date": ""}))
    for h, m in ((7, 51), (8, 21), (9, 1)):
        s.append((h, m, "daily-model.yml", {"mode": "predict", "date": ""}))
    for h, m in ((7, 36), (7, 46), (7, 56), (8, 6), (8, 16), (8, 36), (9, 6)):
        s.append((h, m, "live-record.yml", {"mode": "schedule", "date": ""}))
    for h in range(8, 24):
        for m in range(3, 60, 5):
            if (h, m) >= (8, 43):
                s.append((h, m, "live-record.yml", {"mode": "prerace", "date": ""}))
    s.append((23, 45, "live-record.yml", {"mode": "final", "date": ""}))
    s.append((0, 21, "live-record.yml", {"mode": "final", "date": "YESTERDAY"}))
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
