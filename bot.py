import os
import re
import logging
from datetime import datetime, time
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # Python <3.9
from typing import Optional

# Load .env if present (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from telegram import Update, BotCommand, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    CallbackQueryHandler,
    filters,
)

from storage import (
    init_db,
    get_or_create_user,
    set_tariff,
    get_tariff,
    month_key_for,
    record_reading,
    get_last_reading_before_month,
    get_reading_for_month,
    get_history,
    list_users,
    get_most_recent_reading,
    delete_reading_by_id,
)

# Optional OCR deps
try:
    import cv2  # type: ignore
    import pytesseract  # type: ignore
    import numpy as np  # type: ignore
    _OCR_AVAILABLE = True
except Exception:
    _OCR_AVAILABLE = False


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


BOT_TOKEN = os.getenv("BOT_TOKEN", "")
DB_PATH = os.getenv("EBOT_DB_PATH", os.path.join("data", "ebot.sqlite3"))
TZ_NAME = os.getenv("EBOT_TZ", "UTC")
REMINDER_START_DAY = int(os.getenv("EBOT_REMINDER_START_DAY", "20"))
REMINDER_END_DAY = int(os.getenv("EBOT_REMINDER_END_DAY", "24"))
REMINDER_HOUR = int(os.getenv("EBOT_REMINDER_HOUR", "9"))  # local hour


def _now_tz() -> datetime:
    return datetime.now(ZoneInfo(TZ_NAME))


def _parse_float(text: str) -> Optional[float]:
    try:
        # accept comma or dot
        text = text.strip().replace(",", ".")
        return float(text)
    except Exception:
        # try to extract first number
        m = re.search(r"[-+]?\d+[\.,]?\d*", text)
        if m:
            try:
                return float(m.group(0).replace(",", "."))
            except Exception:
                return None
        return None


def _parse_date(text: str) -> Optional[datetime]:
    s = text.strip()
    tz = ZoneInfo(TZ_NAME)
    formats = [
        "%Y-%m",
        "%Y-%m-%d",
        "%d.%m.%Y",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y.%m.%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            # If only year-month, default to day 1
            if fmt == "%Y-%m":
                dt = dt.replace(day=1)
            return dt.replace(tzinfo=tz, hour=0, minute=0, second=0, microsecond=0)
        except Exception:
            continue
    return None


def _ocr_number_from_image(img_bgr) -> Optional[float]:
    if not _OCR_AVAILABLE:
        return None
    try:
        img = img_bgr
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Scale up small images to help OCR
        h, w = gray.shape[:2]
        scale = 1.0
        target_max = 1200
        if max(h, w) < target_max:
            scale = min(3.0, target_max / float(max(h, w)))
            if scale > 1.2:
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Denoise & enhance contrast
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray_blur)

        # Try to find the main digit band (display area)
        def find_display_roi(src_gray: np.ndarray) -> np.ndarray:
            g = src_gray
            # Edge map to find rectangular bands
            edges = cv2.Canny(g, 50, 150)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            H, W = g.shape[:2]
            best = None
            best_score = 0.0
            for cnt in contours:
                x, y, w2, h2 = cv2.boundingRect(cnt)
                area = w2 * h2
                if area < 0.01 * W * H:
                    continue
                ar = w2 / float(h2 + 1e-6)
                if ar < 2.2 or ar > 12.0:
                    continue
                # Prefer central and larger areas
                cx = x + w2 / 2.0
                cy = y + h2 / 2.0
                center_score = 1.0 - (abs(cx - W / 2.0) / (W / 2.0)) * 0.3 - (abs(cy - H / 2.0) / (H / 2.0)) * 0.3
                score = (area / (W * H)) * 0.6 + center_score * 0.4
                if score > best_score:
                    best_score = score
                    best = (x, y, w2, h2)
            if best is None:
                return src_gray
            x, y, w2, h2 = best
            pad = int(0.06 * max(w2, h2))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(W, x + w2 + pad)
            y1 = min(H, y + h2 + pad)
            return src_gray[y0:y1, x0:x1]

        roi = find_display_roi(gray_eq)

        # Seven-segment specific recognition (helps distinguish 2 vs 7)
        def sevenseg_read(roi_gray: np.ndarray) -> Optional[str]:
            # Binarize and make segments white
            _, th = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Decide inversion based on mean
            if np.mean(th) > 127:
                th = cv2.bitwise_not(th)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            # Find digit contours
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            H, W = th.shape[:2]
            cand = []
            for c in contours:
                x, y, w2, h2 = cv2.boundingRect(c)
                area = w2 * h2
                if h2 < 0.35 * H or w2 < 0.05 * W:
                    continue
                if area < 0.005 * W * H:
                    continue
                cand.append((x, y, w2, h2, area))
            if len(cand) < 3:
                return None
            # Take 5 largest by area, then left->right
            cand = sorted(cand, key=lambda t: t[4], reverse=True)[:5]
            cand = sorted(cand, key=lambda t: t[0])

            # Segment sampling positions as fractions of bbox
            seg_map = {
                # a,    b,    c,    d,    e,    f,    g
                (1, 1, 1, 1, 1, 1, 0): '0',
                (0, 1, 1, 0, 0, 0, 0): '1',
                (1, 1, 0, 1, 1, 0, 1): '2',
                (1, 1, 1, 1, 0, 0, 1): '3',
                (0, 1, 1, 0, 0, 1, 1): '4',
                (1, 0, 1, 1, 0, 1, 1): '5',
                (1, 0, 1, 1, 1, 1, 1): '6',
                (1, 1, 1, 0, 0, 0, 0): '7',
                (1, 1, 1, 1, 1, 1, 1): '8',
                (1, 1, 1, 1, 0, 1, 1): '9',
            }

            def classify_digit(dimg: np.ndarray) -> Optional[str]:
                hh, ww = dimg.shape[:2]
                # pad to avoid border cut-offs
                d = cv2.copyMakeBorder(dimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
                hh, ww = d.shape[:2]
                # Define segment ROIs
                def roi_rect(xr0, yr0, xr1, yr1):
                    x0 = int(xr0 * ww)
                    x1 = int(xr1 * ww)
                    y0 = int(yr0 * hh)
                    y1 = int(yr1 * hh)
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(ww, x1), min(hh, y1)
                    if x1 <= x0 or y1 <= y0:
                        return d[0:1, 0:1]
                    return d[y0:y1, x0:x1]

                a = roi_rect(0.20, 0.08, 0.80, 0.22)
                b = roi_rect(0.72, 0.15, 0.90, 0.52)
                c = roi_rect(0.72, 0.50, 0.90, 0.88)
                dseg = roi_rect(0.20, 0.78, 0.80, 0.92)
                e = roi_rect(0.10, 0.50, 0.28, 0.88)
                f = roi_rect(0.10, 0.15, 0.28, 0.52)
                g = roi_rect(0.22, 0.45, 0.78, 0.58)

                def on(m):
                    # since digits are white on black, mean close to 255 indicates ON
                    return (np.mean(m) / 255.0) > 0.45

                key = (int(on(a)), int(on(b)), int(on(c)), int(on(dseg)), int(on(e)), int(on(f)), int(on(g)))
                return seg_map.get(key)

            out = []
            for (x, y, w2, h2, _) in cand:
                dimg = th[y:y+h2, x:x+w2]
                ch = classify_digit(dimg)
                if ch is None:
                    return None
                out.append(ch)
            s = ''.join(out)
            return s if len(s) >= 5 else None

        s7 = sevenseg_read(roi)
        if s7 and len(s7) >= 5:
            try:
                return float(s7[:5])
            except Exception:
                pass

        # Build multiple candidates (thresholded/raw). We'll first try to get
        # the 5 largest digit boxes using Tesseract's box output.
        _, th_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, th_inv = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        th_bin_c = cv2.morphologyEx(th_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
        th_inv_c = cv2.morphologyEx(th_inv, cv2.MORPH_CLOSE, kernel, iterations=1)

        def top5_from_boxes(image_single_channel) -> Optional[str]:
            try:
                conf = "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789 load_system_dawg=0 load_freq_dawg=0"
                boxes_txt = pytesseract.image_to_boxes(image_single_channel, config=conf)
                if not boxes_txt:
                    return None
                digits = []
                for line in boxes_txt.splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    ch = parts[0]
                    if ch < '0' or ch > '9':
                        continue
                    try:
                        x1, y1, x2, y2 = map(int, parts[1:5])
                        area = max(0, x2 - x1) * max(0, y2 - y1)
                        digits.append((ch, x1, y1, x2, y2, area))
                    except Exception:
                        continue
                if len(digits) < 5:
                    return None
                # Pick 5 with largest area
                digits_sorted_area = sorted(digits, key=lambda t: t[5], reverse=True)[:5]
                # Sort left-to-right
                digits_ltr = sorted(digits_sorted_area, key=lambda t: t[1])
                s = ''.join(d[0] for d in digits_ltr)
                if len(s) == 5:
                    return s
                # If more than 5 due to duplicates, truncate to 5
                return s[:5]
            except Exception:
                return None

        for cand in (th_bin_c, gray_eq, th_inv_c):
            s5 = top5_from_boxes(cand)
            if s5 and len(s5) >= 5:
                try:
                    return float(s5)
                except Exception:
                    pass

        # Fallback: classic OCR and take first 5 digits from the longest run
        def best_digits_from_text(text: str) -> Optional[str]:
            import re as _re
            runs = _re.findall(r"\d+", text)
            if not runs:
                digits = _re.sub(r"\D", "", text)
                return digits if len(digits) >= 4 else None
            longest = max(runs, key=len)
            return longest[:5] if len(longest) >= 5 else longest

        ocr_confs = [
            "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789 load_system_dawg=0 load_freq_dawg=0",
            "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789 load_system_dawg=0 load_freq_dawg=0",
            "--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789 load_system_dawg=0 load_freq_dawg=0",
        ]
        for conf in ocr_confs:
            try:
                text = pytesseract.image_to_string(roi, config=conf)
                digits = best_digits_from_text(text)
                if digits and len(digits) >= 5:
                    return float(digits[:5])
            except Exception:
                continue
        return None
    except Exception:
        return None


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    chat_id = update.effective_chat.id
    get_or_create_user(DB_PATH, user.id, chat_id)
    msg = (
        "Welcome to ebot!\n\n"
        "I’ll remind you before the 25th each month to enter your electricity meter reading.\n\n"
        "Commands:\n"
        "- /tariff — show current tariff\n"
        "- /set_tariff <price> — set tariff per kWh\n"
        "- /enter <reading> [YYYY-MM or YYYY-MM-DD] — record a reading (optionally for a past date)\n"
        "- /remove_last — remove your last saved reading\n"
        "- /history — show recent months\n\n"
        "You can also send a number as a message, or upload a photo. I will try to read the number automatically; otherwise please send it as a message."
    )
    await update.message.reply_text(msg)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


async def cmd_tariff(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    t = get_tariff(DB_PATH, user.id)
    await update.message.reply_text(f"Your tariff: {t:.4f} per kWh")


async def cmd_set_tariff(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not context.args:
        await update.message.reply_text("Usage: /set_tariff <price>")
        return
    value = _parse_float(" ".join(context.args))
    if value is None or value < 0:
        await update.message.reply_text("Please provide a valid non-negative number.")
        return
    set_tariff(DB_PATH, user.id, float(value))
    await update.message.reply_text(f"Tariff updated: {float(value):.4f} per kWh")


async def _save_reading_and_reply(
    update: Update,
    reading_value: Optional[float],
    photo_file_id: Optional[str],
    target_dt: Optional[datetime] = None,
) -> None:
    user = update.effective_user
    now = _now_tz()
    dt = target_dt or now
    mk = month_key_for(dt)
    tariff = get_tariff(DB_PATH, user.id)
    record_reading(DB_PATH, user.id, mk, reading_value, photo_file_id, tariff if reading_value is not None else None)
    current = get_reading_for_month(DB_PATH, user.id, mk)
    prev = get_last_reading_before_month(DB_PATH, user.id, mk)

    if reading_value is None:
        await update.effective_message.reply_text("Photo saved. Please send the reading value as a number.")
        return

    if prev and prev.get("reading_value") is not None:
        delta = float(reading_value) - float(prev["reading_value"])  # type: ignore
        if delta < 0:
            msg = (
                f"Saved {reading_value} for {mk}. Previous was {prev['reading_value']} — delta negative, please double-check."
            )
            await update.effective_message.reply_text(msg)
            return
        cost = delta * float(tariff)
        msg = (
            f"Saved {reading_value} for {mk}.\n"
            f"Used: {delta:.3f} kWh × {tariff:.4f} = {cost:.2f}"
        )
        await update.effective_message.reply_text(msg)
    else:
        await update.effective_message.reply_text(
            f"Saved {reading_value} for {mk}. No previous reading to compute usage."
        )


async def cmd_enter(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /enter <reading> [YYYY-MM or YYYY-MM-DD]")
        return
    reading_text = context.args[0]
    date_text = context.args[1] if len(context.args) >= 2 else None
    value = _parse_float(reading_text)
    if value is None:
        await update.message.reply_text("Please provide a valid number for the reading.")
        return
    target_dt: Optional[datetime] = None
    if date_text:
        parsed = _parse_date(date_text)
        if not parsed:
            await update.message.reply_text("Invalid date. Use YYYY-MM or YYYY-MM-DD.")
            return
        # Optional: prevent future dates
        if parsed > _now_tz():
            await update.message.reply_text("Date cannot be in the future.")
            return
        target_dt = parsed
    await _save_reading_and_reply(update, value, None, target_dt)


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.photo:
        return
    file_id = msg.photo[-1].file_id
    # Do not use numbers from captions; only OCR or manual entry
    reading_value: Optional[float] = None
    if reading_value is None and _OCR_AVAILABLE:
        try:
            # Download photo to temp path
            file = await context.bot.get_file(file_id)
            import tempfile, os as _os
            _os.makedirs("data/tmp", exist_ok=True)
            with tempfile.NamedTemporaryFile(dir="data/tmp", delete=False, suffix=".jpg") as tmp:
                tmp_path = tmp.name
            await file.download_to_drive(custom_path=tmp_path)
            # Read and OCR
            img = cv2.imread(tmp_path)
            if img is not None:
                detected = _ocr_number_from_image(img)
                if detected is not None:
                    # Ask for confirmation before storing
                    context.user_data["ocr_pending"] = {
                        "value": float(detected),
                        "file_id": file_id,
                    }
                    kb = InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton("Save", callback_data="ocr_yes"),
                                InlineKeyboardButton("Cancel", callback_data="ocr_no"),
                            ]
                        ]
                    )
                    await msg.reply_text(
                        f"Detected reading: {int(detected) if float(detected).is_integer() else detected}. Save?",
                        reply_markup=kb,
                    )
                    return
        except Exception as e:
            logger.warning("OCR failed: %s", e)
        finally:
            try:
                if 'tmp_path' in locals():
                    os.remove(tmp_path)
            except Exception:
                pass
    # No OCR value detected or OCR not available: save photo only and ask for number
    await _save_reading_and_reply(update, reading_value, file_id)


async def cb_ocr_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    pending = context.user_data.get("ocr_pending")
    if not pending:
        await query.edit_message_text("No pending reading to save. Please resend the photo or enter the number.")
        return
    if data == "ocr_yes":
        value = float(pending.get("value"))
        file_id = pending.get("file_id")
        # Clear pending before save to avoid duplicates on errors
        context.user_data.pop("ocr_pending", None)
        try:
            # Save value + photo
            await _save_reading_and_reply(update, value, file_id)
            await query.edit_message_text(f"Saved reading: {int(value) if float(value).is_integer() else value}")
        except Exception as e:
            logger.warning("Failed to save confirmed OCR reading: %s", e)
            await query.edit_message_text("Failed to save reading. Please try again or enter it manually.")
    else:
        # Cancel: save photo only and prompt for number
        file_id = pending.get("file_id")
        context.user_data.pop("ocr_pending", None)
        try:
            await _save_reading_and_reply(update, None, file_id)
        except Exception:
            pass
        await query.edit_message_text("Canceled. Photo saved. Please send the reading as a number.")


async def on_number_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.text:
        return
    value = _parse_float(msg.text)
    if value is None:
        return
    await _save_reading_and_reply(update, value, None)


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    # optional limit arg, default 12
    limit = 12
    if context.args:
        try:
            limit = max(1, min(36, int(context.args[0])))
        except Exception:
            pass
    items = get_history(DB_PATH, user.id, limit=limit)
    if not items:
        await update.message.reply_text("No readings yet. Send a number or use /enter to add one.")
        return
    # Present oldest first for readability
    lines = []
    for item in reversed(items):
        mk: str = item.get("month_key")  # type: ignore[assignment]
        ym = mk[:7] if mk else "?"
        rv = item.get("reading_value")
        if rv is None:
            lines.append(f"{ym}: [no numeric reading]")
            continue
        prev = get_last_reading_before_month(DB_PATH, user.id, mk)
        if prev and prev.get("reading_value") is not None:
            delta = float(rv) - float(prev["reading_value"])  # type: ignore[index]
            if delta < 0:
                lines.append(f"{ym}: {rv:.3f} (delta negative vs {prev['reading_value']})")
            else:
                tariff_applied = item.get("tariff_applied")
                if tariff_applied is None:
                    tariff_applied = get_tariff(DB_PATH, user.id)
                cost = delta * float(tariff_applied)
                lines.append(f"{ym}: {rv:.3f} | used {delta:.3f} kWh | cost {cost:.2f}")
        else:
            lines.append(f"{ym}: {rv:.3f} (no previous reading)")
    msg = "Recent readings:\n" + "\n".join(lines)
    await update.message.reply_text(msg)


async def cmd_remove_last(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    last = get_most_recent_reading(DB_PATH, user.id)
    if not last:
        await update.message.reply_text("No readings found to remove.")
        return
    removed = delete_reading_by_id(DB_PATH, last["id"])  # type: ignore[index]
    if removed:
        rv = last.get("reading_value")
        mk = last.get("month_key")
        if rv is not None:
            await update.message.reply_text(f"Removed last reading {rv} for {mk}.")
        else:
            await update.message.reply_text(f"Removed last entry for {mk} (no numeric reading saved).")
    else:
        await update.message.reply_text("Failed to remove the last reading. Please try again.")


async def job_daily_reminder(context: ContextTypes.DEFAULT_TYPE) -> None:
    now = _now_tz()
    day = now.day
    if day < REMINDER_START_DAY or day > REMINDER_END_DAY:
        return
    mk = month_key_for(now)
    for user in list_users(DB_PATH):
        # if user has no reading this month, remind
        existing = get_reading_for_month(DB_PATH, user["user_id"], mk)
        if not existing or existing.get("reading_value") is None:
            try:
                await context.bot.send_message(
                    chat_id=user["chat_id"],
                    text=(
                        "Reminder: please enter your electricity meter reading for this month "
                        "(you can send a number or use /enter)."
                    ),
                )
            except Exception as e:
                logger.warning("Failed to send reminder to %s: %s", user["user_id"], e)


async def _post_init(app: Application) -> None:
    # Register the command menu with Telegram clients
    commands = [
        BotCommand("start", "Register and show help"),
        BotCommand("tariff", "Show current tariff"),
        BotCommand("set_tariff", "Set tariff per kWh"),
        BotCommand("enter", "Save this month's reading"),
        BotCommand("history", "Show recent readings"),
        BotCommand("remove_last", "Remove your last saved reading"),
    ]
    try:
        await app.bot.set_my_commands(commands)
    except Exception as e:
        logger.warning("Failed to set bot commands: %s", e)


def main() -> None:
    if not BOT_TOKEN or "YOUR_TELEGRAM_BOT_TOKEN" in BOT_TOKEN or BOT_TOKEN.startswith("123456:"):
        raise SystemExit(
            "Invalid or placeholder BOT_TOKEN. Set a real token from @BotFather (e.g., export BOT_TOKEN=123:ABC...)."
        )
    init_db(DB_PATH)

    app = Application.builder().token(BOT_TOKEN).post_init(_post_init).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("tariff", cmd_tariff))
    app.add_handler(CommandHandler("set_tariff", cmd_set_tariff))
    app.add_handler(CommandHandler("enter", cmd_enter))
    app.add_handler(CommandHandler("remove_last", cmd_remove_last))
    app.add_handler(CommandHandler("history", cmd_history))

    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(CallbackQueryHandler(cb_ocr_confirm, pattern=r"^ocr_(yes|no)$"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_number_text))

    # Schedule a daily reminder job at REMINDER_HOUR local time (if JobQueue is available)
    local_tz = ZoneInfo(TZ_NAME)
    if getattr(app, "job_queue", None):
        app.job_queue.run_daily(job_daily_reminder, time=time(hour=REMINDER_HOUR, tzinfo=local_tz))
    else:
        logger.warning(
            "JobQueue is not available. Install 'python-telegram-bot[job-queue]' to enable reminders."
        )

    logger.info("Starting bot...")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
