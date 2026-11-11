from typing import Optional
from pydantic import BaseModel, Field


class ProgramBoatRecord(BaseModel):
    """番組表 (Bファイル): 1行 = 1艇"""
    race_id: str = Field(..., description="YYYYMMDD_JJ_RR (例: 20240501_01_01)")
    date: str = Field(..., description="YYYY-MM-DD")
    venue_code: str = Field(..., description="場コード 01-24")
    race_no: int = Field(..., ge=1, le=12)
    race_title: str = Field(default="")
    distance: int = Field(default=1800)
    boat_no: int = Field(..., ge=1, le=6)
    racer_id: int = Field(..., description="4桁登録番号")
    age: int = Field(...)
    branch: str = Field(default="")
    weight: float = Field(...)
    grade: str = Field(..., description="A1, A2, B1, B2")
    nat_win_rate: float = Field(...)
    nat_2ren_rate: float = Field(...)
    loc_win_rate: float = Field(...)
    loc_2ren_rate: float = Field(...)
    motor_no: int = Field(...)
    motor_2ren_rate: float = Field(...)
    boat_num: int = Field(..., description="ボート番号")
    boat_2ren_rate: float = Field(...)


class ResultBoatRecord(BaseModel):
    """競走成績 (Kファイル): 1行 = 1艇"""
    race_id: str = Field(..., description="YYYYMMDD_JJ_RR")
    date: str = Field(..., description="YYYY-MM-DD")
    venue_code: str = Field(..., description="場コード 01-24")
    race_no: int = Field(..., ge=1, le=12)
    boat_no: int = Field(..., ge=1, le=6)
    racer_id: int = Field(...)
    motor_no: int = Field(...)
    boat_num: int = Field(...)
    exhibit_time: Optional[float] = Field(default=None, description="展示タイム")
    course: Optional[int] = Field(default=None, description="進入コース 1-6")
    st_time: Optional[float] = Field(default=None, description="ST (Fはマイナス値)")
    st_status: str = Field(default="NORMAL", description="NORMAL, F, L, K(欠場)")
    finish_order: Optional[int] = Field(default=None, description="着順 1-6 (失格・欠場等はNone)")
    finish_status: str = Field(default="WIN", description="1-6, F, L, K, S0, S1, S2")
    race_time_sec: Optional[float] = Field(default=None, description="レースタイム(秒)")
    winning_trick: str = Field(default="", description="決まり手(1着のみ記録)")
    weather: str = Field(default="")
    wind_dir: str = Field(default="")
    wind_speed: Optional[int] = Field(default=None)
    wave_height: Optional[int] = Field(default=None)


class PayoutRaceRecord(BaseModel):
    """払戻・レース確定情報 (Kファイル): 1行 = 1レース"""
    race_id: str = Field(..., description="YYYYMMDD_JJ_RR")
    date: str = Field(..., description="YYYY-MM-DD")
    venue_code: str = Field(..., description="場コード 01-24")
    race_no: int = Field(..., ge=1, le=12)
    winning_trick: str = Field(default="")
    refund_boats: str = Field(default="", description="返還艇番カンマ区切り (例: '1,4')")
    trifecta_comb: str = Field(default="", description="3連単組番 (例: '1-2-3'、同着時はカンマ区切り)")
    trifecta_payout: int = Field(default=0, description="3連単払戻金(100円あたり、同着時は先頭)")
    trifecta_popularity: Optional[int] = Field(default=None)
    exacta_comb: str = Field(default="", description="2連単組番 (例: '1-2')")
    exacta_payout: int = Field(default=0, description="2連単払戻金")
    exacta_popularity: Optional[int] = Field(default=None)
    is_aborted: bool = Field(default=False, description="不成立・中止フラグ")
