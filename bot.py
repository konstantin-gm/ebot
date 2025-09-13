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

from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
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
        "You can also send a number as a message, or upload a photo (optionally with the reading in the caption)."
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
        await update.effective_message.reply_text("Photo saved. Please send the reading value as a number or with a caption.")
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
    # If caption contains a number, use it
    reading_value: Optional[float] = None
    if msg.caption:
        reading_value = _parse_float(msg.caption)
    await _save_reading_and_reply(update, reading_value, file_id)


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
