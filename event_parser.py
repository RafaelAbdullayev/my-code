"""Utility functions for extracting structured events and tasks from transcripts.

The previous implementation delegated all analytics to the LLM which resulted in
unstable answers when Telegram buttons were pressed.  This module contains a
rule-based fallback parser that runs locally and guarantees deterministic output
for the four key buttons required by the product specification.

The design goals of the parser are:
* Be resilient to partially formatted speech-to-text transcripts.
* Extract dates, times, participants and short summaries from every sentence.
* Support relative temporal expressions such as "сегодня", "завтра",
  "на прошлой неделе" using the user's timezone.
* Provide predictable markdown/HTML friendly formatting helpers so that the bot
  always returns the same field set (title, date, time, participants, summary).

The module exposes a single public entry point – :func:`analyze_transcript` –
which returns an :class:`AnalysisResult` object.  Formatting helpers then use
this object to prepare responses for the buttons.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple
import html
import logging
import re
import textwrap

from datetime import datetime, timedelta

import pytz
from dateparser.search import search_dates

logger = logging.getLogger(__name__)


# --- Data structures -----------------------------------------------------------------

@dataclass
class Entry:
    """Represents a task or meeting extracted from the transcript."""

    kind: str  # either "event" or "task"
    title: str
    summary: str
    participants: List[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None  # timezone aware datetime if available
    time_hint: Optional[str] = None  # textual time if exact datetime is unknown
    order_index: int = 0  # preserves the order of appearance in the transcript

    def pretty_date(self, tz: pytz.BaseTzInfo) -> str:
        """Return a formatted date string or a fallback dash."""
        if self.timestamp:
            localized = self.timestamp.astimezone(tz)
            return localized.strftime("%d.%m.%Y")
        return "—"

    def pretty_time(self, tz: pytz.BaseTzInfo) -> str:
        """Return a formatted time string, using textual hints if necessary."""
        if self.timestamp:
            localized = self.timestamp.astimezone(tz)
            return localized.strftime("%H:%M")
        if self.time_hint:
            return self.time_hint
        return "—"

    def pretty_participants(self) -> str:
        return ", ".join(self.participants) if self.participants else "—"


@dataclass
class AnalysisResult:
    """Container with all extracted items and contextual information."""

    tz: pytz.BaseTzInfo
    base_time: datetime
    entries: List[Entry]

    @property
    def tasks(self) -> List[Entry]:
        return [item for item in self.entries if item.kind == "task"]

    @property
    def events(self) -> List[Entry]:
        return [item for item in self.entries if item.kind == "event"]


# --- Extraction helpers ---------------------------------------------------------------

EVENT_KEYWORDS = {
    "встреча": "Встреча",
    "встречался": "Встреча",
    "созвон": "Созвон",
    "звонок": "Звонок",
    "совещание": "Совещание",
    "митинг": "Митинг",
    "переговор": "Переговоры",
    "обсуждени": "Обсуждение",
    "презентаци": "Презентация",
    "бриф": "Бриф",
    "демо": "Демо",
    "интервью": "Интервью",
    "конферен": "Конференция",
}

TASK_KEYWORDS = {
    "нужно": "Задача",
    "надо": "Задача",
    "поруч": "Задача",
    "задач": "Задача",
    "сделать": "Задача",
    "подготов": "Задача",
    "состав": "Задача",
    "планир": "План",
    "оформ": "Задача",
    "отправ": "Задача",
    "обнов": "Задача",
}

DATE_ONLY_MAX_TOKENS = 6  # heuristics: short sentences describing only a date

NAME_PATTERN = re.compile(
    r"(?:с|со|для|вместе\s+с|с\s+командой|с\s+командою|участвуют\s+|собираюсь\s+с)\s+"
    r"([А-ЯЁA-Z][а-яёa-zA-Z]+(?:\s+[А-ЯЁA-Z][а-яёa-zA-Z]+)?)",
    flags=re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r"[\w.\-+]+@[\w.\-]+\.[A-Za-z]{2,}")
TIME_PATTERN = re.compile(r"(\d{1,2})(?:[:.](\d{2}))?\s*(?:час(?:а|ов)?|ч)?")
EXPLICIT_DATE_PATTERN = re.compile(
    r"(\d{1,2}[./]\d{1,2}([./]\d{2,4})?)|"
    r"(\d{1,2}\s+(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр))|"
    r"(сегодня|завтра|послезавтра|вчера|позавчера|понедельник|вторник|сред[ау]|четверг|пятниц[аеу]|"
    r"суббот[ау]|воскресень[ея])",
    flags=re.IGNORECASE,
)


def _detect_kind(sentence: str) -> Tuple[bool, bool, Optional[str], Optional[str]]:
    """Return flags indicating whether the sentence looks like an event or a task."""
    lower = sentence.lower()
    event_label = None
    task_label = None

    for keyword, label in EVENT_KEYWORDS.items():
        if keyword in lower:
            event_label = label
            break

    for keyword, label in TASK_KEYWORDS.items():
        if keyword in lower:
            task_label = label
            break

    is_event = event_label is not None
    is_task = task_label is not None

    # If none matched but "встреч" substring appears (e.g. "встречи"), treat as event
    if not is_event and "встреч" in lower:
        event_label = "Встреча"
        is_event = True

    return is_event, is_task, event_label, task_label


def _extract_participants(sentence: str) -> List[str]:
    """Extract potential participants (names, emails) from the sentence."""
    participants = set()

    for match in NAME_PATTERN.findall(sentence):
        cleaned = match.strip().strip(",. ")
        if 2 <= len(cleaned) <= 60:
            participants.add(cleaned)

    for email in EMAIL_PATTERN.findall(sentence):
        participants.add(email)

    return sorted(participants)


def _looks_like_date_only(sentence: str, matched_dates: Sequence[Tuple[str, datetime]]) -> bool:
    """Heuristic: true when sentence only defines a date context without actions."""
    tokens = sentence.split()
    if len(tokens) > DATE_ONLY_MAX_TOKENS:
        return False

    if not matched_dates:
        return False

    if any(keyword in sentence.lower() for keyword in ("встр", "звон", "надо", "нужно", "задач")):
        return False

    # If the entire sentence is almost equal to the matched fragment – treat as date context
    sentence_clean = sentence.strip().strip(".,")
    return any(match_text.strip().strip(".,").lower() == sentence_clean.lower() for match_text, _ in matched_dates)


def _ensure_timezone(dt: datetime, tz: pytz.BaseTzInfo) -> datetime:
    """Make the datetime timezone-aware using the provided timezone."""
    if dt.tzinfo is None:
        return tz.localize(dt)
    return dt.astimezone(tz)


def _contains_explicit_date(fragment: str) -> bool:
    return bool(EXPLICIT_DATE_PATTERN.search(fragment))


def _extract_time_hint(fragment: str) -> Optional[str]:
    match = TIME_PATTERN.search(fragment)
    if match:
        hour = int(match.group(1))
        minute = match.group(2)
        if minute is None:
            return f"{hour:02d}:00"
        return f"{hour:02d}:{int(minute):02d}"
    explicit_clock = re.search(r"\b\d{1,2}:\d{2}\b", fragment)
    if explicit_clock:
        time_text = explicit_clock.group(0)
        hour, minute = time_text.split(":")
        return f"{int(hour):02d}:{int(minute):02d}"
    return None


def _make_title(sentence: str, event_label: Optional[str], task_label: Optional[str], kind: str) -> str:
    """Generate a short, human readable title for an entry."""
    if kind == "event" and event_label:
        # Try to capture "встреча с X" to include participant
        match = re.search(r"встреч[аеи]\s+(?:с|со)\s+([А-ЯЁA-Z][^.,;]+)", sentence, flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            return f"{event_label} с {name}".strip()
        return event_label

    if kind == "task" and task_label:
        # Try to reuse the verb phrase after the trigger word
        match = re.search(r"(?:нужно|надо|планирую|следует|поручено|задача)\s+([^.,;]+)", sentence, flags=re.IGNORECASE)
        if match:
            phrase = match.group(1).strip()
            phrase = phrase[0].upper() + phrase[1:] if phrase else phrase
            if len(phrase) > 80:
                phrase = textwrap.shorten(phrase, width=80, placeholder="…")
            return phrase
        return task_label

    # Fallback: use the first 80 characters of the sentence
    return textwrap.shorten(sentence.strip(), width=80, placeholder="…") or "Событие"


def analyze_transcript(transcript: str, tz: pytz.BaseTzInfo) -> AnalysisResult:
    """Parse transcript into structured entries.

    Args:
        transcript: Recognised text from the audio note.
        tz: User timezone (for example pytz.timezone("Europe/Moscow")).

    Returns:
        AnalysisResult containing all extracted entries.
    """
    base_time = datetime.now(tz)
    entries: List[Entry] = []

    if not transcript or not transcript.strip():
        return AnalysisResult(tz=tz, base_time=base_time, entries=entries)

    sentences = re.split(r"(?<=[.!?])\s+", transcript)
    context_date: Optional[datetime] = None

    for index, sentence in enumerate(sentences):
        original_sentence = sentence.strip()
        if not original_sentence:
            continue

        # Use dateparser to find date/time mentions
        matched_dates = search_dates(
            original_sentence,
            languages=["ru"],
            settings={
                "RELATIVE_BASE": context_date or base_time,
                "PREFER_DATES_FROM": "future",
                "DATE_ORDER": "DMY",
                "RETURN_AS_TIMEZONE_AWARE": False,
            },
        ) or []

        if _looks_like_date_only(original_sentence, matched_dates):
            # This sentence merely updates the context date for following sentences
            _, dt = matched_dates[-1]
            context_date = _ensure_timezone(dt, tz).replace(hour=0, minute=0, second=0, microsecond=0)
            logger.debug("Context date updated to %s from sentence '%s'", context_date, original_sentence)
            continue

        is_event, is_task, event_label, task_label = _detect_kind(original_sentence)
        if not is_event and not is_task:
            # No actionable content, but we still update the context if the sentence contains a date
            if matched_dates:
                _, dt = matched_dates[-1]
                context_date = _ensure_timezone(dt, tz).replace(hour=0, minute=0, second=0, microsecond=0)
            continue

        resolved_datetime: Optional[datetime] = None
        time_hint: Optional[str] = None

        if matched_dates:
            # Take the last occurrence – it is usually the most specific in Russian
            matched_text, parsed_dt = matched_dates[-1]
            parsed_dt = _ensure_timezone(parsed_dt, tz)
            time_hint = _extract_time_hint(matched_text)

            if context_date and not _contains_explicit_date(matched_text):
                parsed_dt = context_date.replace(
                    hour=parsed_dt.hour,
                    minute=parsed_dt.minute,
                    second=0,
                    microsecond=0,
                )
            resolved_datetime = parsed_dt
            context_date = parsed_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif context_date:
            # Sentence lacks explicit date but we have context, reuse it
            resolved_datetime = context_date

        participants = _extract_participants(original_sentence)
        kind = "event" if is_event else "task"
        title = _make_title(original_sentence, event_label, task_label, kind)

        entry = Entry(
            kind=kind,
            title=title,
            summary=original_sentence,
            participants=participants,
            timestamp=resolved_datetime,
            time_hint=time_hint,
            order_index=len(entries) if resolved_datetime is None else index,
        )
        entries.append(entry)

    return AnalysisResult(tz=tz, base_time=base_time, entries=entries)


# --- Filtering and formatting ---------------------------------------------------------

def _sorted_entries(entries: Iterable[Entry], base_time: datetime) -> List[Entry]:
    def sort_key(item: Entry) -> Tuple[int, datetime, int]:
        # Entries with known timestamp go first ordered by datetime; the rest keep the transcript order.
        has_time = 0 if item.timestamp else 1
        timestamp = item.timestamp or base_time + timedelta(days=365)
        order = item.order_index
        return has_time, timestamp, order

    return sorted(entries, key=sort_key)


def _format_entries(title: str, entries: Sequence[Entry], tz: pytz.BaseTzInfo, base_time: datetime) -> str:
    if not entries:
        return ""

    lines: List[str] = [f"<b>{html.escape(title)}</b>"]

    for idx, entry in enumerate(_sorted_entries(entries, base_time), start=1):
        lines.append(f"{idx}. <b>{html.escape(entry.title)}</b>")
        lines.append(f"&nbsp;&nbsp;• Тип: {'Встреча' if entry.kind == 'event' else 'Задача'}")
        lines.append(f"&nbsp;&nbsp;• Дата: {entry.pretty_date(tz)}")
        lines.append(f"&nbsp;&nbsp;• Время: {entry.pretty_time(tz)}")
        lines.append(f"&nbsp;&nbsp;• Участники: {html.escape(entry.pretty_participants())}")
        summary = textwrap.shorten(entry.summary, width=180, placeholder="…") if entry.summary else "—"
        lines.append(f"&nbsp;&nbsp;• Кратко: {html.escape(summary)}")
        lines.append("")

    return "\n".join(lines).strip()


def format_plan_response(result: AnalysisResult) -> str:
    """Format combined list of tasks and meetings for the planning button."""
    combined = result.entries
    message = _format_entries("События и задачи, требующие планирования", combined, result.tz, result.base_time)
    if not message:
        return (
            "📭 В записи не нашёл задач или встреч для планирования.\n"
            "Попробуйте сформулировать запрос с конкретными глаголами (например, "
            "«нужно созвониться завтра в 10 с Анной»)."
        )
    return message


def format_today_response(result: AnalysisResult) -> str:
    tz = result.tz
    today_start = result.base_time.replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow_start = today_start + timedelta(days=1)

    todays_items = [
        entry
        for entry in result.events
        if entry.timestamp and today_start <= entry.timestamp < tomorrow_start
    ]
    message = _format_entries("Встречи за сегодня", todays_items, tz, result.base_time)
    if not message:
        return (
            "📭 Сегодня не нашёл встреч в тексте.\n"
            "Если они были, добавьте явное указание даты или слова «сегодня»."
        )
    return message


def format_tomorrow_response(result: AnalysisResult) -> str:
    tz = result.tz
    today_start = result.base_time.replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow_start = today_start + timedelta(days=1)
    day_after_tomorrow = tomorrow_start + timedelta(days=1)

    tomorrow_items = [
        entry
        for entry in result.events
        if entry.timestamp and tomorrow_start <= entry.timestamp < day_after_tomorrow
    ]
    message = _format_entries("План встреч на завтра", tomorrow_items, tz, result.base_time)
    if not message:
        return (
            "📭 На завтра встреч не найдено.\n"
            "Добавьте в запись формулировки вроде «завтра в 15:00 созвон с командой»."
        )
    return message


def format_last_week_response(result: AnalysisResult) -> str:
    tz = result.tz
    base = result.base_time
    days_since_monday = base.weekday()
    start_of_current_week = base - timedelta(days=days_since_monday)
    end_of_last_week = start_of_current_week - timedelta(seconds=1)
    start_of_last_week = start_of_current_week - timedelta(days=7)

    start = start_of_last_week.replace(hour=0, minute=0, second=0, microsecond=0)
    end = end_of_last_week.replace(hour=23, minute=59, second=59, microsecond=999999)

    last_week_items = [
        entry
        for entry in result.entries
        if entry.timestamp and start <= entry.timestamp <= end
    ]
    message = _format_entries("Встречи и задачи прошлой недели", last_week_items, tz, result.base_time)
    if not message:
        return (
            "📭 За прошлую неделю событий не нашёл.\n"
            "Если встречи были, укажите конкретные даты в формате «в понедельник 3 июня …»."
        )
    return message


__all__ = [
    "Entry",
    "AnalysisResult",
    "analyze_transcript",
    "format_plan_response",
    "format_today_response",
    "format_tomorrow_response",
    "format_last_week_response",
]
