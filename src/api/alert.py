"""Alert delivery — threshold check + LINE/SMS notification stub."""
from __future__ import annotations

import logging
import os

import httpx

from src.api.schemas import PredictResponse

logger = logging.getLogger(__name__)

LINE_API_URL = "https://api.line.me/v2/bot/message/push"


def should_alert(response: PredictResponse, horizon_hours: int = 24) -> bool:
    """Return True if the 24h forecast triggers the rain alert."""
    for h in response.forecast:
        if h.horizon_hours == horizon_hours and h.alert:
            return True
    return False


def send_line_alert(farm_id: str, message: str) -> bool:
    """Send LINE push message to a farm user. Returns True on success."""
    token = os.getenv("LINE_CHANNEL_TOKEN")
    if not token:
        logger.warning("LINE_CHANNEL_TOKEN not set — alert not sent")
        return False
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"to": farm_id, "messages": [{"type": "text", "text": message}]}
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(LINE_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
        return True
    except Exception as exc:
        logger.error(f"LINE alert failed for {farm_id}: {exc}")
        return False


def dispatch_rain_alert(farm_id: str, probability: float, horizon_hours: int) -> None:
    """Build and dispatch alert message for a farm."""
    msg = (
        f"🌧️ CropOS แจ้งเตือน: คาดการณ์ฝนตก {int(probability * 100)}% "
        f"ใน {horizon_hours} ชั่วโมง\n"
        f"ควรเลื่อนการใส่ปุ๋ยออกไปก่อน"
    )
    send_line_alert(farm_id, msg)
