import os
import random
import sqlite3
import subprocess
import string
import time
from collections import deque
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template, request, send_from_directory, stream_with_context, url_for
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from util import empty_or_not, get_parking_spots_bboxes


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
MASK_PATH = ROOT / "mask_1920_1080.png"
MASK_CROP_PATH = ROOT / "mask_crop.png"
BOOKINGS_DB_PATH = ROOT / "bookings.db"
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB

STRIPE_MIN_AMOUNT_PAISE = 5000


def get_stripe_secret_key():
    # Read at request time so key updates do not require code changes.
    return (os.getenv("STRIPE_SECRET_KEY") or "").strip()


def get_stripe_webhook_secret():
    return (os.getenv("STRIPE_WEBHOOK_SECRET") or "").strip()


def get_app_base_url():
    return (os.getenv("APP_BASE_URL") or "").strip().rstrip("/")

LIVE_STATE_LOCK = Lock()
LIVE_STATE = {
    "running": False,
    "source": "",
    "available": 0,
    "total": 0,
    "frame": 0,
}

ACTIVE_SESSION_LOCK = Lock()
ACTIVE_SESSION_RESULT = None

LIVE_SOURCES = {
    "full": ROOT / "data" / "parking_1920_1080_loop.mp4",
    "crop": ROOT / "data" / "parking_crop_loop.mp4",
}

DEFAULT_PARKING_LOTS = [
    {
        "name": "Delhi Central Hub",
        "address": "New Delhi, NCR",
        "mask_name": MASK_PATH.name,
        "latitude": 28.6139,
        "longitude": 77.2090,
        "capacity": 396,
        "color": "#f06a42",
    },
    {
        "name": "Mumbai Waterfront",
        "address": "Mumbai, Maharashtra",
        "mask_name": MASK_PATH.name,
        "latitude": 19.0760,
        "longitude": 72.8777,
        "capacity": 224,
        "color": "#14746f",
    },
    {
        "name": "Bengaluru Tech Park",
        "address": "Bengaluru, Karnataka",
        "mask_name": MASK_CROP_PATH.name,
        "latitude": 12.9716,
        "longitude": 77.5946,
        "capacity": 14,
        "color": "#5b8def",
    },
    {
        "name": "Kolkata River Side",
        "address": "Kolkata, West Bengal",
        "mask_name": MASK_PATH.name,
        "latitude": 22.5726,
        "longitude": 88.3639,
        "capacity": 168,
        "color": "#8f63ff",
    },
    {
        "name": "Hyderabad Orbit",
        "address": "Hyderabad, Telangana",
        "mask_name": MASK_PATH.name,
        "latitude": 17.3850,
        "longitude": 78.4867,
        "capacity": 180,
        "color": "#e35d8f",
    },
    {
        "name": "Jaipur Pink Zone",
        "address": "Jaipur, Rajasthan",
        "mask_name": MASK_PATH.name,
        "latitude": 26.9124,
        "longitude": 75.7873,
        "capacity": 152,
        "color": "#f2a541",
    },
    {
        "name": "Chennai Marina Point",
        "address": "Chennai, Tamil Nadu",
        "mask_name": MASK_CROP_PATH.name,
        "latitude": 13.0827,
        "longitude": 80.2707,
        "capacity": 14,
        "color": "#0fb9b1",
    },
    {
        "name": "Ahmedabad Commerce Park",
        "address": "Ahmedabad, Gujarat",
        "mask_name": MASK_PATH.name,
        "latitude": 23.0225,
        "longitude": 72.5714,
        "capacity": 142,
        "color": "#ff6f61",
    },
    {
        "name": "Kochi Lagoon Lot",
        "address": "Kochi, Kerala",
        "mask_name": MASK_CROP_PATH.name,
        "latitude": 9.9312,
        "longitude": 76.2673,
        "capacity": 14,
        "color": "#00b894",
    },
    {
        "name": "Guwahati Gateway",
        "address": "Guwahati, Assam",
        "mask_name": MASK_PATH.name,
        "latitude": 26.1445,
        "longitude": 91.7362,
        "capacity": 96,
        "color": "#fd9644",
    },
]

LEGACY_MAP_HIDDEN_NAMES = {"North Gate Lot", "Riverside Crop Lot", "Demo West Deck"}


def _india_map_position(latitude, longitude):
    lat_min, lat_max = 6.0, 38.0
    lon_min, lon_max = 68.0, 98.0
    x_percent = ((longitude - lon_min) / (lon_max - lon_min)) * 100.0
    y_percent = ((lat_max - latitude) / (lat_max - lat_min)) * 100.0
    return round(max(0.0, min(100.0, x_percent)), 4), round(max(0.0, min(100.0, y_percent)), 4)


def get_db_connection():
    conn = sqlite3.connect(str(BOOKINGS_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_bookings_db():
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mask_name TEXT NOT NULL,
                spot_index INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                UNIQUE(mask_name, spot_index)
            )
            """
        )
        booking_columns = {row[1] for row in conn.execute("PRAGMA table_info(bookings)").fetchall()}
        if "expires_at" not in booking_columns:
            conn.execute("ALTER TABLE bookings ADD COLUMN expires_at INTEGER")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS parking_lots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                address TEXT NOT NULL,
                mask_name TEXT,
                latitude REAL,
                longitude REAL,
                x_percent REAL,
                y_percent REAL,
                capacity INTEGER NOT NULL,
                color TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(parking_lots)").fetchall()}
        if "latitude" not in columns:
            conn.execute("ALTER TABLE parking_lots ADD COLUMN latitude REAL")
        if "longitude" not in columns:
            conn.execute("ALTER TABLE parking_lots ADD COLUMN longitude REAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS payments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mask_name TEXT NOT NULL,
                spot_index INTEGER NOT NULL,
                vehicle_number TEXT NOT NULL,
                duration_hours INTEGER NOT NULL,
                amount_paise INTEGER NOT NULL,
                currency TEXT NOT NULL,
                provider TEXT NOT NULL,
                status TEXT NOT NULL,
                transaction_ref TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def seed_parking_lots():
    conn = get_db_connection()
    try:
        for lot in DEFAULT_PARKING_LOTS:
            existing = conn.execute(
                "SELECT id FROM parking_lots WHERE name = ?",
                (lot["name"],),
            ).fetchone()
            if existing:
                x_percent, y_percent = _india_map_position(lot["latitude"], lot["longitude"])
                conn.execute(
                    """
                    UPDATE parking_lots
                    SET address = ?, mask_name = ?, latitude = ?, longitude = ?, x_percent = ?, y_percent = ?, capacity = ?, color = ?
                    WHERE name = ?
                    """,
                    (
                        lot["address"],
                        lot["mask_name"],
                        float(lot["latitude"]),
                        float(lot["longitude"]),
                        x_percent,
                        y_percent,
                        int(lot["capacity"]),
                        lot["color"],
                        lot["name"],
                    ),
                )
            else:
                x_percent, y_percent = _india_map_position(lot["latitude"], lot["longitude"])
                conn.execute(
                    """
                    INSERT INTO parking_lots(name, address, mask_name, latitude, longitude, x_percent, y_percent, capacity, color, created_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lot["name"],
                        lot["address"],
                        lot["mask_name"],
                        float(lot["latitude"]),
                        float(lot["longitude"]),
                        x_percent,
                        y_percent,
                        int(lot["capacity"]),
                        lot["color"],
                        int(time.time()),
                    ),
                )
        conn.commit()
    finally:
        conn.close()


def get_booked_indices(mask_name):
    cleanup_expired_bookings()
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT spot_index
            FROM bookings
            WHERE mask_name = ?
              AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY spot_index
            """,
            (mask_name, int(time.time())),
        ).fetchall()
    finally:
        conn.close()
    return [int(row["spot_index"]) for row in rows]


def get_active_bookings(mask_name):
    cleanup_expired_bookings()
    now = int(time.time())
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT spot_index, expires_at
            FROM bookings
            WHERE mask_name = ?
              AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY spot_index
            """,
            (mask_name, now),
        ).fetchall()
    finally:
        conn.close()

    bookings = []
    for row in rows:
        expires_at = row["expires_at"]
        expires_value = int(expires_at) if expires_at is not None else None
        remaining_seconds = max(0, expires_value - now) if expires_value is not None else None
        bookings.append(
            {
                "spot_index": int(row["spot_index"]),
                "expires_at": expires_value,
                "remaining_seconds": remaining_seconds,
            }
        )
    return bookings


def create_booking(mask_name, spot_index, duration_hours=None):
    now = int(time.time())
    expires_at = None
    if duration_hours is not None:
        try:
            duration_hours = int(duration_hours)
        except (TypeError, ValueError):
            duration_hours = None
        if duration_hours and duration_hours > 0:
            expires_at = now + (duration_hours * 3600)

    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO bookings(mask_name, spot_index, created_at, expires_at) VALUES(?, ?, ?, ?)",
            (mask_name, int(spot_index), now, expires_at),
        )
        conn.commit()
    finally:
        conn.close()


def cleanup_expired_bookings():
    now = int(time.time())
    conn = get_db_connection()
    try:
        conn.execute(
            "DELETE FROM bookings WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )
        conn.commit()
    finally:
        conn.close()


def get_booking_by_mask_and_spot(mask_name, spot_index):
    cleanup_expired_bookings()
    conn = get_db_connection()
    try:
        row = conn.execute(
            """
            SELECT id, mask_name, spot_index, created_at, expires_at
            FROM bookings
            WHERE mask_name = ? AND spot_index = ?
            LIMIT 1
            """,
            (mask_name, int(spot_index)),
        ).fetchone()
    finally:
        conn.close()
    return row


def clear_bookings_for_mask(mask_name):
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM bookings WHERE mask_name = ?", (mask_name,))
        conn.commit()
    finally:
        conn.close()


def create_payment(
    mask_name,
    spot_index,
    vehicle_number,
    duration_hours,
    amount_paise,
    currency,
    provider,
    status,
    transaction_ref,
):
    now = int(time.time())
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO payments(
                mask_name, spot_index, vehicle_number, duration_hours, amount_paise, currency,
                provider, status, transaction_ref, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mask_name,
                int(spot_index),
                vehicle_number,
                int(duration_hours),
                int(amount_paise),
                currency,
                provider,
                status,
                transaction_ref,
                now,
                now,
            ),
        )
        conn.commit()
        payment_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    finally:
        conn.close()
    return int(payment_id)


def update_payment_status(payment_id, status):
    conn = get_db_connection()
    try:
        conn.execute(
            "UPDATE payments SET status = ?, updated_at = ? WHERE id = ?",
            (status, int(time.time()), int(payment_id)),
        )
        conn.commit()
    finally:
        conn.close()


def get_payment_by_transaction_ref(provider, transaction_ref):
    conn = get_db_connection()
    try:
        row = conn.execute(
            """
            SELECT id, mask_name, spot_index, vehicle_number, duration_hours, amount_paise, currency,
                   provider, status, transaction_ref, created_at, updated_at
            FROM payments
            WHERE provider = ? AND transaction_ref = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (provider, transaction_ref),
        ).fetchone()
    finally:
        conn.close()
    return row


def is_spot_booked(mask_name, spot_index):
    cleanup_expired_bookings()
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT id FROM bookings WHERE mask_name = ? AND spot_index = ? LIMIT 1",
            (mask_name, int(spot_index)),
        ).fetchone()
    finally:
        conn.close()
    return row is not None


def _generate_transaction_ref(prefix="PP"):
    suffix = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    return f"{prefix}{int(time.time())}{suffix}"


def _calculate_booking_amount_paise(duration_hours):
    hourly_rate_paise = 4000  # Rs.40 per hour
    return int(duration_hours) * hourly_rate_paise


def _create_stripe_checkout_session(mask_name, spot_index, vehicle_number, duration_hours, amount_paise):
    stripe_secret_key = get_stripe_secret_key()
    if not stripe_secret_key:
        raise RuntimeError("Stripe is not configured. Set STRIPE_SECRET_KEY in your environment.")

    try:
        import stripe
    except ImportError as exc:
        raise RuntimeError("Stripe SDK is not installed. Run: pip install stripe") from exc

    stripe.api_key = stripe_secret_key

    app_base_url = get_app_base_url()
    if app_base_url:
        success_url = f"{app_base_url}{url_for('stripe_payment_status_page')}?session_id={{CHECKOUT_SESSION_ID}}"
        cancel_url = f"{app_base_url}{url_for('stripe_checkout_cancel')}"
    else:
        success_url = url_for("stripe_payment_status_page", _external=True) + "?session_id={CHECKOUT_SESSION_ID}"
        cancel_url = url_for("stripe_checkout_cancel", _external=True)

    checkout_session = stripe.checkout.Session.create(
        mode="payment",
        success_url=success_url,
        cancel_url=cancel_url,
        billing_address_collection="auto",
        phone_number_collection={"enabled": True},
        customer_creation="always",
        line_items=[
            {
                "quantity": 1,
                "price_data": {
                    "currency": "inr",
                    "unit_amount": int(amount_paise),
                    "product_data": {
                        "name": f"Parking Slot #{int(spot_index) + 1}",
                        "description": f"{int(duration_hours)} hour booking",
                    },
                },
            }
        ],
        metadata={
            "mask_name": mask_name,
            "spot_index": str(int(spot_index)),
            "vehicle_number": vehicle_number,
            "duration_hours": str(int(duration_hours)),
        },
    )

    return checkout_session


def remove_booking(mask_name, spot_index):
    conn = get_db_connection()
    try:
        conn.execute(
            "DELETE FROM bookings WHERE mask_name = ? AND spot_index = ?",
            (mask_name, int(spot_index)),
        )
        conn.commit()
    finally:
        conn.close()


def _fulfill_stripe_checkout_session(session_id, payment_status=None):
    payment_row = get_payment_by_transaction_ref("stripe", session_id)
    if payment_row is None:
        return False, "Payment record not found for this Stripe session.", 404

    status = str(payment_status or "").strip().lower()
    if status and status != "paid":
        if payment_row["status"] != "succeeded":
            update_payment_status(int(payment_row["id"]), "failed")
        return False, "Stripe payment is not completed.", 400

    if payment_row["status"] == "succeeded":
        return True, "Payment already confirmed.", 200

    try:
        create_booking(
            payment_row["mask_name"],
            int(payment_row["spot_index"]),
            duration_hours=int(payment_row["duration_hours"]),
        )
    except sqlite3.IntegrityError:
        update_payment_status(int(payment_row["id"]), "failed")
        return False, "Payment succeeded, but this slot is no longer available.", 409

    update_payment_status(int(payment_row["id"]), "succeeded")
    return True, "Payment successful and booking confirmed.", 200


def _sync_and_get_stripe_payment_status(session_id):
    cleanup_expired_bookings()
    payment_row = get_payment_by_transaction_ref("stripe", session_id)
    if payment_row is None:
        return None

    provider_status = str(payment_row["status"])
    checkout_status = "unknown"
    stripe_payment_status = "unknown"

    # For async methods, checkout may complete before funds are confirmed.
    if provider_status not in {"succeeded", "failed"}:
        stripe_secret_key = get_stripe_secret_key()
        if stripe_secret_key:
            try:
                import stripe

                stripe.api_key = stripe_secret_key
                checkout_session = stripe.checkout.Session.retrieve(session_id)
                checkout_status = str(checkout_session.get("status") or "unknown")
                stripe_payment_status = str(checkout_session.get("payment_status") or "unknown")

                if stripe_payment_status == "paid":
                    _fulfill_stripe_checkout_session(session_id, payment_status="paid")
                elif checkout_status == "expired":
                    latest_row = get_payment_by_transaction_ref("stripe", session_id)
                    if latest_row and latest_row["status"] != "succeeded":
                        update_payment_status(int(latest_row["id"]), "failed")
            except Exception:
                # Keep local status as source of truth if Stripe lookup fails.
                pass

    refreshed = get_payment_by_transaction_ref("stripe", session_id)
    if refreshed is None:
        return None

    booking_row = get_booking_by_mask_and_spot(refreshed["mask_name"], int(refreshed["spot_index"]))
    now = int(time.time())
    booking_remaining_seconds = None
    booking_expires_at = None
    booking_active = False
    if booking_row is not None:
        expires_raw = booking_row["expires_at"]
        if expires_raw is not None:
            booking_expires_at = int(expires_raw)
            booking_remaining_seconds = max(0, booking_expires_at - now)
        booking_active = True

    final_status = str(refreshed["status"])
    if final_status == "succeeded":
        display_status = "succeeded"
        message = "Payment confirmed. Your parking slot is booked."
        is_final = True
        is_success = True
    elif final_status == "failed":
        display_status = "failed"
        message = "Payment failed or expired. No booking was made."
        is_final = True
        is_success = False
    else:
        display_status = "processing"
        message = "Payment is processing. This page will update automatically."
        is_final = False
        is_success = False

    return {
        "session_id": session_id,
        "status": display_status,
        "is_final": is_final,
        "is_success": is_success,
        "message": message,
        "booking": {
            "active": booking_active,
            "expires_at": booking_expires_at,
            "remaining_seconds": booking_remaining_seconds,
        },
        "payment": {
            "provider": str(refreshed["provider"]),
            "currency": str(refreshed["currency"]),
            "amount_paise": int(refreshed["amount_paise"]),
            "amount_display": f"Rs.{int(refreshed['amount_paise']) / 100:.2f}",
            "vehicle_number": str(refreshed["vehicle_number"]),
            "duration_hours": int(refreshed["duration_hours"]),
            "mask_name": str(refreshed["mask_name"]),
            "spot_index": int(refreshed["spot_index"]),
            "spot_label": f"Slot #{int(refreshed['spot_index']) + 1}",
            "transaction_ref": str(refreshed["transaction_ref"]),
        },
        "stripe": {
            "checkout_status": checkout_status,
            "payment_status": stripe_payment_status,
        },
    }


def get_booking_counts_by_mask():
    cleanup_expired_bookings()
    now = int(time.time())
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT mask_name, COUNT(*) AS booked_count
            FROM bookings
            WHERE expires_at IS NULL OR expires_at > ?
            GROUP BY mask_name
            """,
            (now,)
        ).fetchall()
    finally:
        conn.close()

    counts = {}
    for row in rows:
        counts[row["mask_name"]] = int(row["booked_count"])
    return counts


def get_parking_lots():
    booking_counts = get_booking_counts_by_mask()
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, name, address, mask_name, latitude, longitude, x_percent, y_percent, capacity, color, created_at
            FROM parking_lots
            ORDER BY id ASC
            """
        ).fetchall()
    finally:
        conn.close()

    lots = []
    for row in rows:
        mask_name = row["mask_name"]
        booked_count = int(booking_counts.get(mask_name, 0)) if mask_name else 0
        capacity = int(row["capacity"])
        lots.append(
            {
                "id": int(row["id"]),
                "name": row["name"],
                "address": row["address"],
                "mask_name": mask_name,
                "latitude": float(row["latitude"]) if row["latitude"] is not None else None,
                "longitude": float(row["longitude"]) if row["longitude"] is not None else None,
                "x_percent": float(row["x_percent"]) if row["x_percent"] is not None else None,
                "y_percent": float(row["y_percent"]) if row["y_percent"] is not None else None,
                "capacity": capacity,
                "booked_count": booked_count,
                "available_count": max(capacity - booked_count, 0),
                "color": row["color"],
                "created_at": int(row["created_at"]),
                "visible_on_map": row["name"] not in LEGACY_MAP_HIDDEN_NAMES,
            }
        )
    return lots


def set_active_session_result(result):
    global ACTIVE_SESSION_RESULT
    with ACTIVE_SESSION_LOCK:
        ACTIVE_SESSION_RESULT = dict(result)


def get_active_session_result():
    with ACTIVE_SESSION_LOCK:
        result = dict(ACTIVE_SESSION_RESULT) if ACTIVE_SESSION_RESULT else None
    if not result:
        return None

    video_url = str(result.get("video_url") or "")
    filename = video_url.rsplit("/", 1)[-1] if video_url else ""
    if not filename:
        return None
    if not (OUTPUT_DIR / filename).exists():
        return None
    return result


def clear_active_session(release_bookings=False):
    global ACTIVE_SESSION_RESULT
    with ACTIVE_SESSION_LOCK:
        current = dict(ACTIVE_SESSION_RESULT) if ACTIVE_SESSION_RESULT else None
        ACTIVE_SESSION_RESULT = None

    if release_bookings and current and current.get("mask_name"):
        clear_bookings_for_mask(str(current["mask_name"]))


def create_parking_lot(payload):
    name = (payload.get("name") or "").strip()
    address = (payload.get("address") or "").strip()
    mask_name = (payload.get("mask_name") or "").strip() or None
    color = (payload.get("color") or "#f06a42").strip() or "#f06a42"

    try:
        latitude = float(payload.get("latitude"))
        longitude = float(payload.get("longitude"))
        capacity = int(payload.get("capacity"))
    except (TypeError, ValueError):
        raise ValueError("latitude, longitude and capacity must be valid numbers")

    if not name:
        raise ValueError("name is required")
    if not address:
        raise ValueError("address is required")
    if capacity <= 0:
        raise ValueError("capacity must be greater than zero")
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("latitude must be between -90 and 90, longitude between -180 and 180")

    x_percent, y_percent = _india_map_position(latitude, longitude)

    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO parking_lots(name, address, mask_name, latitude, longitude, x_percent, y_percent, capacity, color, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (name, address, mask_name, latitude, longitude, x_percent, y_percent, capacity, color, int(time.time())),
        )
        conn.commit()
        row_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    finally:
        conn.close()

    return int(row_id)


init_bookings_db()
seed_parking_lots()


def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))


def allowed_file(filename):
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def _read_video_size(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Unable to read video dimensions: {video_path}")
    return width, height


def _read_mask_size(mask_path):
    mask = cv2.imread(str(mask_path), 0)
    if mask is None:
        return None
    h, w = mask.shape[:2]
    return w, h


def select_mask_for_video(video_path, uploaded_name=""):
    candidates = []
    if MASK_PATH.exists():
        candidates.append(MASK_PATH)
    if MASK_CROP_PATH.exists():
        candidates.append(MASK_CROP_PATH)

    if not candidates:
        raise FileNotFoundError("No mask files found.")

    lower_name = (uploaded_name or str(video_path)).lower()
    if "crop" in lower_name and MASK_CROP_PATH.exists():
        return MASK_CROP_PATH

    vid_w, vid_h = _read_video_size(video_path)
    vid_ratio = vid_w / vid_h

    best_mask = None
    best_delta = float("inf")
    for mask_path in candidates:
        mask_size = _read_mask_size(mask_path)
        if not mask_size:
            continue
        mask_w, mask_h = mask_size
        mask_ratio = mask_w / mask_h
        delta = abs(vid_ratio - mask_ratio) / mask_ratio
        if delta < best_delta:
            best_delta = delta
            best_mask = mask_path

    if best_mask is None:
        raise FileNotFoundError("Could not read any valid mask file.")

    return best_mask


def process_video(input_video_path, output_video_path, mask_path, step=5):
    mask = cv2.imread(str(mask_path), 0)
    if mask is None:
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {input_video_path}")

    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    spots = get_parking_spots_bboxes(connected_components)
    mask_height, mask_width = mask.shape[:2]

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25

    writer = None
    actual_output_path = None

    spots_status = [False for _ in spots]
    spot_history = [deque(maxlen=5) for _ in spots]
    frame_nmr = 0
    last_available = 0
    scaled_spots = spots

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if writer is None:
                frame_height, frame_width = frame.shape[:2]
                frame_ratio = frame_width / frame_height
                mask_ratio = mask_width / mask_height
                ratio_delta = abs(frame_ratio - mask_ratio) / mask_ratio
                if ratio_delta > 0.03:
                    raise RuntimeError(
                        "Input video aspect ratio does not match the trained mask/camera setup. "
                        "Use a video from the same camera view as mask_1920_1080.png for accurate detection."
                    )
                scale_x = frame_width / mask_width
                scale_y = frame_height / mask_height
                scaled_spots = [
                    [
                        int(round(x1 * scale_x)),
                        int(round(y1 * scale_y)),
                        max(1, int(round(w * scale_x))),
                        max(1, int(round(h * scale_y))),
                    ]
                    for x1, y1, w, h in spots
                ]
                # Use MJPEG codec which is widely compatible; save as AVI
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                actual_output_path = str(output_video_path).rsplit('.', 1)[0] + '.avi'
                writer = cv2.VideoWriter(
                    actual_output_path,
                    fourcc,
                    fps,
                    (int(frame_width), int(frame_height)),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Unable to create output video: {actual_output_path}")

            if frame_nmr % step == 0:
                for spot_index, (x1, y1, w, h) in enumerate(scaled_spots):
                    frame_h, frame_w = frame.shape[:2]
                    margin_x = max(1, int(0.08 * w))
                    margin_y = max(1, int(0.08 * h))
                    x1_clamped = max(0, x1 + margin_x)
                    y1_clamped = max(0, y1 + margin_y)
                    x2 = min(frame_w, x1 + w - margin_x)
                    y2 = min(frame_h, y1 + h - margin_y)

                    if x1_clamped >= x2 or y1_clamped >= y2:
                        continue

                    spot_crop = frame[y1_clamped:y2, x1_clamped:x2, :]
                    if spot_crop.size == 0:
                        continue
                    current_prediction = bool(empty_or_not(spot_crop))
                    spot_history[spot_index].append(current_prediction)
                    empty_votes = sum(spot_history[spot_index])
                    spots_status[spot_index] = empty_votes >= ((len(spot_history[spot_index]) + 1) // 2)

            for spot_index, (x1, y1, w, h) in enumerate(scaled_spots):
                spot_status = spots_status[spot_index]
                color = (0, 255, 0) if spot_status else (0, 0, 255)
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

            last_available = sum(bool(status) for status in spots_status)
            total_spots = len(spots_status)

            cv2.rectangle(frame, (70, 20), (600, 90), (18, 16, 22), -1)
            cv2.putText(
                frame,
                f"Available spots: {last_available} / {total_spots}",
                (90, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (245, 246, 250),
                2,
            )

            writer.write(frame)
            frame_nmr += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if frame_nmr == 0:
        raise RuntimeError("No frames were processed from the input video.")

    # Convert AVI to MP4 using FFmpeg for browser compatibility
    if actual_output_path and Path(actual_output_path).exists():
        avi_path = actual_output_path
        mp4_path = str(output_video_path)
        try:
            subprocess.run(
                ["ffmpeg", "-i", avi_path, "-c:v", "libx264", "-preset", "fast", "-y", mp4_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=300,
            )
            if Path(avi_path).exists():
                os.remove(avi_path)
        except Exception:
            # If FFmpeg fails, try to use the AVI file anyway
            if Path(avi_path).exists():
                try:
                    os.rename(avi_path, mp4_path)
                except Exception:
                    pass

    return {
        "available": last_available,
        "total": len(spots_status),
        "processed_frames": frame_nmr,
        "spot_statuses": [bool(status) for status in spots_status],
        "mask_name": Path(mask_path).name,
    }


def _run_detection_for_frame(frame, scaled_spots, spots_status, spot_history, frame_nmr, step=5):
    if frame_nmr % step == 0:
        for spot_index, (x1, y1, w, h) in enumerate(scaled_spots):
            frame_h, frame_w = frame.shape[:2]
            margin_x = max(1, int(0.08 * w))
            margin_y = max(1, int(0.08 * h))
            x1_clamped = max(0, x1 + margin_x)
            y1_clamped = max(0, y1 + margin_y)
            x2 = min(frame_w, x1 + w - margin_x)
            y2 = min(frame_h, y1 + h - margin_y)

            if x1_clamped >= x2 or y1_clamped >= y2:
                continue

            spot_crop = frame[y1_clamped:y2, x1_clamped:x2, :]
            if spot_crop.size == 0:
                continue

            current_prediction = bool(empty_or_not(spot_crop))
            spot_history[spot_index].append(current_prediction)
            empty_votes = sum(spot_history[spot_index])
            spots_status[spot_index] = empty_votes >= ((len(spot_history[spot_index]) + 1) // 2)

    for spot_index, (x1, y1, w, h) in enumerate(scaled_spots):
        spot_status = spots_status[spot_index]
        color = (0, 255, 0) if spot_status else (0, 0, 255)
        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    available = sum(bool(status) for status in spots_status)
    total = len(spots_status)
    return frame, available, total


def generate_live_feed(source_name):
    video_path = LIVE_SOURCES[source_name]
    mask_path = select_mask_for_video(video_path, video_path.name)
    mask = cv2.imread(str(mask_path), 0)
    if mask is None:
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    spots = get_parking_spots_bboxes(connected_components)
    mask_height, mask_width = mask.shape[:2]

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise RuntimeError(f"Unable to read first frame from: {video_path}")

    frame_height, frame_width = first_frame.shape[:2]
    scale_x = frame_width / mask_width
    scale_y = frame_height / mask_height
    scaled_spots = [
        [
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
            max(1, int(round(w * scale_x))),
            max(1, int(round(h * scale_y))),
        ]
        for x1, y1, w, h in spots
    ]

    spots_status = [False for _ in spots]
    spot_history = [deque(maxlen=5) for _ in spots]
    frame_nmr = 0

    with LIVE_STATE_LOCK:
        LIVE_STATE.update({
            "running": True,
            "source": source_name,
            "available": 0,
            "total": len(spots_status),
            "frame": 0,
        })

    try:
        frame = first_frame
        while True:
            if frame is None:
                break

            annotated, available, total = _run_detection_for_frame(
                frame,
                scaled_spots,
                spots_status,
                spot_history,
                frame_nmr,
                step=5,
            )

            cv2.rectangle(annotated, (70, 20), (650, 90), (18, 16, 22), -1)
            cv2.putText(
                annotated,
                f"LIVE Available spots: {available} / {total}",
                (90, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (245, 246, 250),
                2,
            )

            with LIVE_STATE_LOCK:
                LIVE_STATE.update(
                    {
                        "running": True,
                        "source": source_name,
                        "available": int(available),
                        "total": int(total),
                        "frame": int(frame_nmr),
                    }
                )

            ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )

            frame_nmr += 1
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
    finally:
        cap.release()
        with LIVE_STATE_LOCK:
            LIVE_STATE["running"] = False


@app.route("/")
def index():
    cleanup_expired_bookings()
    return render_template(
        "index.html",
        result=get_active_session_result(),
        error=None,
        parking_lots=get_parking_lots(),
        fresh_result=False,
    )


@app.route("/process", methods=["POST"])
def process_upload():
    file = request.files.get("video")
    if not file or file.filename == "":
        return render_template(
            "index.html",
            result=None,
            error="Please choose a video file to process.",
            parking_lots=get_parking_lots(),
            fresh_result=False,
        )

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            result=None,
            error="Unsupported format. Use mp4, mov, avi, or mkv.",
        )

    safe_name = secure_filename(file.filename)
    timestamp = int(time.time())
    input_name = f"{timestamp}_{safe_name}"
    output_name = f"processed_{timestamp}.mp4"

    input_path = UPLOAD_DIR / input_name
    output_path = OUTPUT_DIR / output_name

    file.save(input_path)

    try:
        selected_mask = select_mask_for_video(input_path, safe_name)
        stats = process_video(input_path, output_path, selected_mask)
    except Exception as exc:
        return render_template(
            "index.html",
            result=None,
            error=str(exc),
            parking_lots=get_parking_lots(),
            fresh_result=False,
        )

    result = {
        "video_url": url_for("serve_output", filename=output_name),
        "download_url": url_for("serve_output", filename=output_name),
        "available": stats["available"],
        "total": stats["total"],
        "processed_frames": stats["processed_frames"],
        "file_name": output_name,
        "spot_statuses": stats["spot_statuses"],
        "mask_name": stats["mask_name"],
    }

    set_active_session_result(result)

    return render_template(
        "index.html",
        result=result,
        error=None,
        parking_lots=get_parking_lots(),
        fresh_result=True,
    )


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@app.route("/api/bookings", methods=["GET"])
def list_bookings():
    cleanup_expired_bookings()
    mask_name = (request.args.get("mask_name") or "").strip()
    if not mask_name:
        return jsonify({"error": "mask_name is required"}), 400
    bookings = get_active_bookings(mask_name)
    return jsonify(
        {
            "mask_name": mask_name,
            "booked_indices": [entry["spot_index"] for entry in bookings],
            "bookings": bookings,
        }
    )


@app.route("/api/bookings/book", methods=["POST"])
def book_spot():
    cleanup_expired_bookings()
    data = request.get_json(silent=True) or {}
    mask_name = (data.get("mask_name") or "").strip()
    spot_index = data.get("spot_index")
    spot_empty = data.get("spot_empty")

    if not mask_name or not isinstance(spot_index, int):
        return jsonify({"error": "mask_name and integer spot_index are required"}), 400

    if spot_empty is not True:
        return jsonify({"error": "Only currently empty spots can be booked"}), 400

    try:
        create_booking(mask_name, spot_index)
    except sqlite3.IntegrityError:
        return jsonify({"error": "Spot already booked"}), 409

    return jsonify({
        "ok": True,
        "mask_name": mask_name,
        "spot_index": spot_index,
        "booked_indices": get_booked_indices(mask_name),
    })


@app.route("/api/bookings/book", methods=["DELETE"])
def unbook_spot():
    cleanup_expired_bookings()
    data = request.get_json(silent=True) or {}
    mask_name = (data.get("mask_name") or "").strip()
    spot_index = data.get("spot_index")

    if not mask_name or not isinstance(spot_index, int):
        return jsonify({"error": "mask_name and integer spot_index are required"}), 400

    remove_booking(mask_name, spot_index)
    return jsonify({
        "ok": True,
        "mask_name": mask_name,
        "spot_index": spot_index,
        "booked_indices": get_booked_indices(mask_name),
    })


@app.route("/api/bookings/pay", methods=["POST"])
def pay_and_book_spot():
    cleanup_expired_bookings()
    data = request.get_json(silent=True) or {}
    mask_name = (data.get("mask_name") or "").strip()
    spot_index = data.get("spot_index")
    spot_empty = data.get("spot_empty")
    vehicle_number = (data.get("vehicle_number") or "").strip().upper()
    provider = (data.get("provider") or "mock").strip().lower()

    try:
        duration_hours = int(data.get("duration_hours"))
    except (TypeError, ValueError):
        return jsonify({"error": "duration_hours must be an integer"}), 400

    if not mask_name or not isinstance(spot_index, int):
        return jsonify({"error": "mask_name and integer spot_index are required"}), 400

    if spot_empty is not True:
        return jsonify({"error": "Only currently empty spots can be booked"}), 400

    if duration_hours < 1 or duration_hours > 24:
        return jsonify({"error": "duration_hours must be between 1 and 24"}), 400

    if not vehicle_number:
        return jsonify({"error": "vehicle_number is required"}), 400

    if is_spot_booked(mask_name, spot_index):
        return jsonify({"error": "This slot has already been booked. Please select another slot."}), 409

    amount_paise = _calculate_booking_amount_paise(duration_hours)
    transaction_ref = _generate_transaction_ref()

    if provider == "stripe":
        if amount_paise < STRIPE_MIN_AMOUNT_PAISE:
            min_hours = (STRIPE_MIN_AMOUNT_PAISE + 3999) // 4000
            return jsonify(
                {
                    "error": (
                        "Stripe minimum charge is Rs.50. "
                        f"Choose at least {min_hours} hours or use Demo Gateway."
                    )
                }
            ), 400

        try:
            checkout_session = _create_stripe_checkout_session(
                mask_name=mask_name,
                spot_index=spot_index,
                vehicle_number=vehicle_number,
                duration_hours=duration_hours,
                amount_paise=amount_paise,
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        payment_id = create_payment(
            mask_name=mask_name,
            spot_index=spot_index,
            vehicle_number=vehicle_number,
            duration_hours=duration_hours,
            amount_paise=amount_paise,
            currency="INR",
            provider=provider,
            status="checkout_created",
            transaction_ref=checkout_session.id,
        )

        return jsonify(
            {
                "ok": True,
                "requires_redirect": True,
                "checkout_url": checkout_session.url,
                "payment": {
                    "payment_id": payment_id,
                    "status": "checkout_created",
                    "provider": provider,
                    "transaction_ref": checkout_session.id,
                    "currency": "INR",
                    "amount_paise": amount_paise,
                    "amount_display": f"Rs.{amount_paise / 100:.2f}",
                },
            }
        )

    if provider != "mock":
        return jsonify({"error": "Unsupported provider. Use 'mock' or 'stripe'."}), 400

    payment_id = create_payment(
        mask_name=mask_name,
        spot_index=spot_index,
        vehicle_number=vehicle_number,
        duration_hours=duration_hours,
        amount_paise=amount_paise,
        currency="INR",
        provider=provider,
        status="initiated",
        transaction_ref=transaction_ref,
    )

    try:
        create_booking(mask_name, spot_index, duration_hours=duration_hours)
    except sqlite3.IntegrityError:
        update_payment_status(payment_id, "failed")
        return jsonify({"error": "Spot already booked"}), 409

    update_payment_status(payment_id, "succeeded")

    return jsonify(
        {
            "ok": True,
            "mask_name": mask_name,
            "spot_index": spot_index,
            "booked_indices": get_booked_indices(mask_name),
            "payment": {
                "payment_id": payment_id,
                "status": "succeeded",
                "provider": provider,
                "transaction_ref": transaction_ref,
                "currency": "INR",
                "amount_paise": amount_paise,
                "amount_display": f"Rs.{amount_paise / 100:.2f}",
            },
        }
    )


@app.route("/payments/stripe/success", methods=["GET"])
def stripe_checkout_success():
    session_id = (request.args.get("session_id") or "").strip()
    if not session_id:
        return "Missing session_id.", 400
    return redirect(url_for("stripe_payment_status_page", session_id=session_id), code=302)


@app.route("/payments/stripe/status", methods=["GET"])
def stripe_payment_status_page():
    session_id = (request.args.get("session_id") or "").strip()
    if not session_id:
        return "Missing session_id.", 400
    return render_template("stripe_payment_status.html", session_id=session_id)


@app.route("/api/payments/stripe/status", methods=["GET"])
def stripe_payment_status_api():
    session_id = (request.args.get("session_id") or "").strip()
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    payload = _sync_and_get_stripe_payment_status(session_id)
    if payload is None:
        return jsonify({"error": "Payment record not found for this Stripe session."}), 404

    return jsonify(payload)


@app.route("/api/session/end", methods=["POST"])
def end_active_session():
    cleanup_expired_bookings()
    clear_active_session(release_bookings=True)
    return jsonify({"ok": True, "message": "Session ended. You can upload a new stream now."})


@app.route("/payments/stripe/cancel", methods=["GET"])
def stripe_checkout_cancel():
    return (
        "<h2>Stripe payment cancelled</h2>"
        "<p>No slot was booked.</p>"
        "<a href='/'>Back to dashboard</a>",
        200,
    )


@app.route("/payments/stripe/webhook", methods=["POST"])
def stripe_webhook():
    stripe_secret_key = get_stripe_secret_key()
    webhook_secret = get_stripe_webhook_secret()
    if not stripe_secret_key or not webhook_secret:
        return jsonify({"error": "Stripe webhook not configured."}), 400

    try:
        import stripe
    except ImportError:
        return jsonify({"error": "Stripe SDK is not installed."}), 500

    stripe.api_key = stripe_secret_key
    payload = request.get_data()
    sig_header = request.headers.get("Stripe-Signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except ValueError:
        return jsonify({"error": "Invalid webhook payload."}), 400
    except stripe.error.SignatureVerificationError:
        return jsonify({"error": "Invalid Stripe webhook signature."}), 400

    event_type = event.get("type", "")
    data_object = (event.get("data") or {}).get("object") or {}

    if event_type in {"checkout.session.completed", "checkout.session.async_payment_succeeded"}:
        session_id = str(data_object.get("id") or "").strip()
        payment_status = str(data_object.get("payment_status") or "").strip()
        _fulfill_stripe_checkout_session(session_id, payment_status=payment_status)

    if event_type in {"checkout.session.expired", "checkout.session.async_payment_failed"}:
        session_id = str(data_object.get("id") or "").strip()
        payment_row = get_payment_by_transaction_ref("stripe", session_id)
        if payment_row and payment_row["status"] != "succeeded":
            update_payment_status(int(payment_row["id"]), "failed")

    return jsonify({"received": True})


@app.route("/api/live_status", methods=["GET"])
def live_status():
    with LIVE_STATE_LOCK:
        payload = dict(LIVE_STATE)
    return jsonify(payload)


@app.route("/api/parking_lots", methods=["GET"])
def api_parking_lots():
    return jsonify({"parking_lots": get_parking_lots()})


@app.route("/api/parking_lots", methods=["POST"])
def api_create_parking_lot():
    payload = request.get_json(silent=True) or {}
    try:
        lot_id = create_parking_lot(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"ok": True, "lot_id": lot_id, "parking_lots": get_parking_lots()})


@app.route("/live_feed", methods=["GET"])
def live_feed():
    source_name = (request.args.get("source") or "full").strip().lower()
    if source_name not in LIVE_SOURCES:
        return jsonify({"error": f"Unknown source '{source_name}'. Use one of: {', '.join(LIVE_SOURCES.keys())}"}), 400
    if not LIVE_SOURCES[source_name].exists():
        return jsonify({"error": f"Source video not found: {LIVE_SOURCES[source_name]}"}), 404

    return Response(
        stream_with_context(generate_live_feed(source_name)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(debug=True)
