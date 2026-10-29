"""Single-process SQLite store; no raw text in control or job records."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA secure_delete=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS controls (
            user_id TEXT PRIMARY KEY, no_save INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT NOT NULL, expires REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS preferences (
            user_id TEXT PRIMARY KEY, content TEXT NOT NULL, version INTEGER NOT NULL,
            receipt TEXT NOT NULL, confirmed_at REAL NOT NULL, expires REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs (
            user_id TEXT NOT NULL, id TEXT NOT NULL, status TEXT NOT NULL,
            expires REAL NOT NULL, PRIMARY KEY(user_id, id));
        CREATE TABLE IF NOT EXISTS usage (
            day TEXT PRIMARY KEY, calls INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS search_usage (
            day TEXT PRIMARY KEY, calls INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS notes (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL,
            content TEXT NOT NULL, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS calendar_events (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL,
            starts REAL NOT NULL, description TEXT NOT NULL DEFAULT '');
        CREATE TABLE IF NOT EXISTS reminders (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, content TEXT NOT NULL,
            due REAL NOT NULL, kind TEXT NOT NULL DEFAULT 'reminder',
            status TEXT NOT NULL DEFAULT 'scheduled', event_id TEXT);
        CREATE TABLE IF NOT EXISTS notifications (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, content TEXT NOT NULL,
            created REAL NOT NULL, source TEXT NOT NULL, seen INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS followups (
            user_id TEXT PRIMARY KEY, enabled INTEGER NOT NULL DEFAULT 0,
            hour INTEGER NOT NULL DEFAULT 20, minute INTEGER NOT NULL DEFAULT 0,
            last_date TEXT NOT NULL DEFAULT '', last_activity REAL NOT NULL DEFAULT 0);
        CREATE INDEX IF NOT EXISTS messages_user ON messages(user_id, id);
        CREATE INDEX IF NOT EXISTS reminders_due ON reminders(status, due);
        CREATE INDEX IF NOT EXISTS notifications_user ON notifications(user_id, seen, created);
        """)
        # Prototype tasks are never replayed after a process restart.
        self.db.execute("UPDATE jobs SET status='interrupted' WHERE status='generating'")
        self.db.commit()

    def prune(self, now):
        with self.db:
            for table in ("messages", "preferences", "jobs"):
                self.db.execute(f"DELETE FROM {table} WHERE expires<=?", (now,))
            self.db.execute("DELETE FROM usage WHERE day < ?", (self.day(now - 7 * 86400),))
            self.db.execute("DELETE FROM search_usage WHERE day < ?", (self.day(now - 7 * 86400),))
            self.db.execute("DELETE FROM notifications WHERE seen=1 AND created<?", (now - 7 * 86400,))
            self.db.execute("DELETE FROM reminders WHERE status<>'scheduled' AND due<?", (now - 7 * 86400,))

    def no_save(self, user):
        row = self.db.execute("SELECT no_save FROM controls WHERE user_id=?", (user,)).fetchone()
        return bool(row[0]) if row else False

    def set_mode(self, user, no_save):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO controls VALUES (?, ?)", (user, no_save))

    def append_pair(self, user, text, reply, expires):
        with self.db:
            self.db.executemany("INSERT INTO messages(user_id, role, content, expires) VALUES (?,?,?,?)",
                                [(user, "user", text, expires), (user, "assistant", reply, expires)])

    def history(self, user, now, limit):
        rows = self.db.execute("SELECT role, content FROM messages WHERE user_id=? AND expires>? ORDER BY id DESC LIMIT ?",
                               (user, now, limit)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def preferences(self, user, now):
        row = self.db.execute("SELECT * FROM preferences WHERE user_id=? AND expires>?", (user, now)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["content"] = json.loads(result["content"])
        result.pop("user_id")
        return result

    def save_preferences(self, user, content, now, expires):
        old = self.preferences(user, now)
        version = old["version"] + 1 if old else 1
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO preferences VALUES (?,?,?,?,?,?)",
                            (user, json.dumps(content, ensure_ascii=False), version, str(uuid4()), now, expires))

    def forget_preferences(self, user):
        with self.db:
            self.db.execute("DELETE FROM preferences WHERE user_id=?", (user,))

    def delete_all(self, user):
        with self.db:
            for table in ("messages", "preferences", "jobs", "notes", "calendar_events", "reminders", "notifications", "followups"):
                self.db.execute(f"DELETE FROM {table} WHERE user_id=?", (user,))

    def add_note(self, user, title, content, now):
        note_id = uuid4().hex
        with self.db:
            self.db.execute("INSERT INTO notes VALUES (?,?,?,?,?)", (note_id, user, title, content, now))
        return note_id

    def notes(self, user, query=""):
        # instr matches literal text, not SQL wildcard patterns.
        rows = self.db.execute("SELECT id,title,content,created FROM notes WHERE user_id=? AND "
                               "(?='' OR instr(title,?)>0 OR instr(content,?)>0) "
                               "ORDER BY created DESC LIMIT 20", (user, query, query, query)).fetchall()
        return [dict(row) for row in rows]

    def delete_note(self, user, note_id):
        with self.db:
            return self.db.execute("DELETE FROM notes WHERE user_id=? AND id=?", (user, note_id)).rowcount > 0

    def note(self, user, note_id):
        row = self.db.execute("SELECT id,title,content,created FROM notes WHERE user_id=? AND id=?", (user, note_id)).fetchone()
        return dict(row) if row else None

    def add_event(self, user, title, starts, description, remind_before):
        event_id = uuid4().hex
        with self.db:
            self.db.execute("INSERT INTO calendar_events VALUES (?,?,?,?,?)", (event_id, user, title, starts, description))
            if remind_before is not None:
                self.db.execute("INSERT INTO reminders VALUES (?,?,?,?,?,?,?)",
                                (uuid4().hex, user, title, starts - remind_before * 60,
                                 "event", "scheduled", event_id))
        return event_id

    def events(self, user, start, end):
        rows = self.db.execute("SELECT id,title,starts,description FROM calendar_events "
                               "WHERE user_id=? AND starts>=? AND starts<? ORDER BY starts LIMIT 50",
                               (user, start, end)).fetchall()
        return [dict(row) for row in rows]

    def delete_event(self, user, event_id):
        with self.db:
            found = self.db.execute("DELETE FROM calendar_events WHERE user_id=? AND id=?", (user, event_id)).rowcount
            if found:
                self.db.execute("DELETE FROM notifications WHERE user_id=? AND seen=0 AND id IN "
                                "(SELECT id FROM reminders WHERE user_id=? AND event_id=?)", (user, user, event_id))
                self.db.execute("UPDATE reminders SET status='cancelled' WHERE user_id=? AND event_id=? AND status='scheduled'",
                                (user, event_id))
        return bool(found)

    def add_reminder(self, user, content, due, kind="reminder"):
        reminder_id = uuid4().hex
        with self.db:
            self.db.execute("INSERT INTO reminders VALUES (?,?,?,?,?,?,NULL)",
                            (reminder_id, user, content, due, kind, "scheduled"))
        return reminder_id

    def reminders(self, user):
        rows = self.db.execute("SELECT id,content,due,kind,event_id FROM reminders "
                               "WHERE user_id=? AND status='scheduled' ORDER BY due LIMIT 50", (user,)).fetchall()
        return [dict(row) for row in rows]

    def cancel_reminder(self, user, reminder_id):
        with self.db:
            return self.db.execute("UPDATE reminders SET status='cancelled' "
                                   "WHERE user_id=? AND id=? AND status='scheduled'", (user, reminder_id)).rowcount > 0

    def queue_due_reminders(self, now, allowed_users):
        # Create the inbox entry and update its job in one transaction. The
        # original reminder ID is also the notification ID, preventing duplicates.
        with self.db:
            rows = self.db.execute("SELECT * FROM reminders WHERE status='scheduled' AND due<=? ORDER BY due", (now,)).fetchall()
            for row in rows:
                if row['user_id'] not in allowed_users:
                    continue
                label = {'reminder': '提醒', 'event': '日程提醒', 'followup': '跟进'}.get(row['kind'], '提醒')
                self.db.execute("INSERT OR IGNORE INTO notifications VALUES (?,?,?,?,?,0)",
                                (row['id'], row['user_id'], f"{label}：{row['content']}", now, row['kind']))
                self.db.execute("UPDATE reminders SET status='delivered' WHERE id=? AND status='scheduled'", (row['id'],))

    def notifications(self, user):
        rows = self.db.execute("SELECT id,content,created,source FROM notifications "
                               "WHERE user_id=? AND seen=0 ORDER BY created,id LIMIT 100", (user,)).fetchall()
        return [dict(row) for row in rows]

    def acknowledge_notifications(self, user, ids):
        with self.db:
            self.db.executemany("UPDATE notifications SET seen=1 WHERE user_id=? AND id=?", [(user, value) for value in ids])

    def followup(self, user):
        row = self.db.execute("SELECT enabled,hour,minute,last_date,last_activity FROM followups WHERE user_id=?", (user,)).fetchone()
        return dict(row) if row else {'enabled': 0, 'hour': 20, 'minute': 0, 'last_date': '', 'last_activity': 0}

    def set_followup(self, user, enabled, hour, minute):
        with self.db:
            self.db.execute("INSERT INTO followups(user_id,enabled,hour,minute) VALUES (?,?,?,?) "
                            "ON CONFLICT(user_id) DO UPDATE SET enabled=excluded.enabled,hour=excluded.hour,minute=excluded.minute",
                            (user, enabled, hour, minute))
            if not enabled:
                self.db.execute("DELETE FROM notifications WHERE user_id=? AND source='daily_followup' AND seen=0", (user,))
                self.db.execute("DELETE FROM notifications WHERE user_id=? AND source='followup' AND seen=0", (user,))
                self.db.execute("UPDATE reminders SET status='cancelled' WHERE user_id=? AND kind='followup' AND status='scheduled'", (user,))

    def record_activity(self, user, now):
        with self.db:
            self.db.execute("INSERT INTO followups(user_id,last_activity) VALUES (?,?) "
                            "ON CONFLICT(user_id) DO UPDATE SET last_activity=excluded.last_activity", (user, now))

    def queue_daily_followup(self, user, local_date, now):
        with self.db:
            updated = self.db.execute("UPDATE followups SET last_date=? WHERE user_id=? AND enabled=1 AND last_date<>?",
                                      (local_date, user, local_date)).rowcount
            if updated:
                self.db.execute("INSERT INTO notifications VALUES (?,?,?,?,?,0)",
                                (uuid4().hex, user, "今天过得怎么样？想聊时再来就好。", now, 'daily_followup'))

    def job(self, user, request_id):
        row = self.db.execute("SELECT status FROM jobs WHERE user_id=? AND id=?", (user, request_id)).fetchone()
        return row[0] if row else None

    def start_job(self, user, request_id, now):
        with self.db:
            self.db.execute("INSERT INTO jobs VALUES (?,?,?,?)", (user, request_id, "generating", now + 86400))

    def finish_job(self, user, request_id, status):
        with self.db:
            self.db.execute("UPDATE jobs SET status=? WHERE user_id=? AND id=?", (status, user, request_id))

    @staticmethod
    def day(now):
        return datetime.fromtimestamp(now, timezone.utc).date().isoformat()

    def reserve_call(self, now, limit):
        day = self.day(now)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO usage VALUES (?,0)", (day,))
            cursor = self.db.execute("UPDATE usage SET calls=calls+1 WHERE day=? AND calls<?", (day, limit))
            return cursor.rowcount == 1

    def reserve_search(self, now, limit):
        day = self.day(now)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO search_usage VALUES (?,0)", (day,))
            cursor = self.db.execute("UPDATE search_usage SET calls=calls+1 WHERE day=? AND calls<?", (day, limit))
            return cursor.rowcount == 1

    def close(self):
        self.db.close()
