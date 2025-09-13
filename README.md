# ebot

Telegram Python bot for electricity cost accounting.

Features:
- Reminds every month before the 25th to enter electricity meter readings.
- Accepts manual readings via text or `/enter <value>`.
- Accepts a photo upload (optionally with the reading in the caption); stores photo for reference.
- Tries to extract the reading from uploaded photos using OpenCV + Tesseract OCR (optional; see below).
- Stores all readings, tracks per-user tariff, and calculates the current month cost from the previous reading.
- Tariff can be changed by the user at any time.

## Quick start

1) Install dependencies:

```
pip install -r requirements.txt
```

2) Set environment variables (create a `.env` from `.env.example` or export directly):

```
export BOT_TOKEN=123456:YOUR_TELEGRAM_BOT_TOKEN
# Optional:
# export EBOT_DB_PATH=data/ebot.sqlite3
# export EBOT_TZ=Europe/Berlin  # default UTC
# export EBOT_REMINDER_START_DAY=20
# export EBOT_REMINDER_END_DAY=24
# export EBOT_REMINDER_HOUR=9
```

3) Run the bot:

```
python bot.py
```

## Commands

- `/start`: Register and show help.
- `/help`: Show help.
- `/tariff`: Show current tariff per kWh.
- `/set_tariff <price>`: Set tariff (e.g., `/set_tariff 0.14`).
- `/enter <reading>`: Save meter reading for the current month.
- Send a number as a plain message to save it as the current month reading.
- Send a photo (optionally with a numeric caption) to attach it to the current month; if caption contains a number, the reading is saved too.

Notes:
- The bot computes usage as the difference between this month’s reading and the last previous reading available. The cost uses your current tariff at the time you save the reading.
- If the delta is negative, the bot warns you to double‑check the input.

## Reminders

- The bot schedules a daily job at a configurable hour (default 09:00 in `EBOT_TZ`).
- On days `EBOT_REMINDER_START_DAY` to `EBOT_REMINDER_END_DAY` (inclusive), it reminds users who have not yet submitted a reading for the month.

## Storage

- Data is stored in a local SQLite database (default `data/ebot.sqlite3`). Tables are created automatically on first run.

## Development

- Main entry: `bot.py`
- Storage module: `storage.py`

## OCR (optional)

- The bot can OCR numbers from photos.
- Requires: `opencv-python`, `pytesseract`, and the Tesseract binary installed on your system.
- Install packages: `pip install -r requirements.txt` (already includes OCR libs).
- Install Tesseract binary:
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - macOS (Homebrew): `brew install tesseract`
  - Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
- If Tesseract isn’t installed, the bot still works; it just asks you to input the number manually.
