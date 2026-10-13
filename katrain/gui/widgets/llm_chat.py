"""Chat panel that explains KataGo's move choices."""

import threading
import time

from kivy.app import App
from kivy.clock import Clock
from kivy.properties import BooleanProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout

from katrain.core.lang import i18n
from katrain.core.llm import (
    analysis_ready,
    build_user_prompt,
    describe_position,
    encode_thread,
    followup_messages,
    grade_word,
    prepare_messages,
    replace_commentary,
    request_chat,
    settings_ready,
)
from katrain.gui.speech import (
    SpeechPlayer,
    completed_sentences,
    doubao_ready,
    highlight_span,
    should_follow_scroll,
    speech_backend,
    split_sentences,
)
from katrain.gui.theme import Theme, to_hexcol

ANALYSIS_WAIT = 8.0

USER_COLOR = to_hexcol(Theme.SCORE_COLOR)
STATUS_COLOR = to_hexcol(Theme.POINTLOSS_COLOR)
ERROR_COLOR = to_hexcol(Theme.ERROR_BORDER_COLOR)
SPEECH_COLOR = to_hexcol([0.96, 0.78, 0.42, 1])


def escape_markup(text):
    return (text or "").replace("&", "&amp;").replace("[", "\\[").replace("]", "\\]")


class LLMChatPanel(BoxLayout):
    """Commentary for the move currently on screen.

    Each game node keeps its own commentary and follow-up questions. Switching
    moves shows that move's text again, and saving the game writes it into the SGF.
    """

    chat_input = ObjectProperty(None)
    log_label = ObjectProperty(None)
    scroll = ObjectProperty(None)
    active = BooleanProperty(False)
    speech_state = StringProperty("idle")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lines = []
        self.queue = []
        self.busy = False
        self.view_node = None
        self._partial = None
        self._status = None
        self._error = None
        self._jump_bottom = True
        self._spoken_index = 0
        self._spoken_token = None
        self._speech_hold = False
        self._speaking_span = None
        self._speaker = SpeechPlayer(on_state=self._speech_state, on_chunk=self._speech_chunk)
        Clock.schedule_interval(self._pump, 0.4)
        Clock.schedule_once(self._render_current, 0)

    def on_active(self, *_args):
        if not self.active and self.chat_input is not None:
            self.chat_input.focus = False

    def reset(self):
        """Drop the on-screen view when a new game replaces the node tree."""
        self.queue = []
        self.busy = False
        self.view_node = None
        self._partial = None
        self._status = None
        self._error = None
        self._jump_bottom = True
        self._spoken_token = None
        self._speech_hold = False
        self._speaker.stop()
        self._render_current()

    def show_node(self, node):
        if node is self.view_node:
            return
        self.view_node = node
        self._jump_bottom = True
        self._spoken_token = None
        self._speech_hold = False
        self._speaker.stop()
        self._render_current()

    def clear_current(self):
        node = self._current_node()
        if node is not None:
            node._llm_token = None
            self._store_thread(node, [])
        if self._partial and self._partial[0] is node:
            self._partial = None
        if self._status and self._status[0] is node:
            self._status = None
        if self._error and self._error[0] is node:
            self._error = None
        self.queue = [job for job in self.queue if job["node"] is not node]
        self._spoken_token = None
        self._speaker.stop()
        self._render_current()

    def explain_current(self, mode="played"):
        katrain = self._katrain()
        if katrain is None or not getattr(katrain, "game", None):
            return
        self._focus_tab()
        self.queue.append(
            {
                "kind": "played" if mode != "ask" else "ask",
                "node": katrain.game.current_node,
                "text": "",
                "since": time.time(),
            }
        )

    def send_question(self):
        if self.chat_input is None:
            return
        text = self.chat_input.text.strip()
        if not text:
            return
        self.chat_input.text = ""
        katrain = self._katrain()
        node = katrain.game.current_node if katrain and katrain.game else None
        self._focus_tab()
        self.queue.append({"kind": "ask", "node": node, "text": text, "since": time.time()})

    def _pump(self, _dt):
        if self.busy or not self.queue:
            return
        job = self.queue[0]
        cfg = self._cfg()
        if not settings_ready(cfg):
            self.queue.pop(0)
            self._focus_tab()
            self._error = (job["node"], i18n._("llm:not-configured"))
            self._render_current()
            return
        if job["kind"] == "played":
            full = analysis_ready(job["node"], "played")
            partial = analysis_ready(job["node"], "played", partial_ok=True)
            if not full and time.time() - job["since"] < ANALYSIS_WAIT:
                if self._status != (job["node"], "wait"):
                    self._focus_tab()
                    self._status = (job["node"], "wait")
                    self._render_current()
                return
            if not partial:
                self.queue.pop(0)
                self._status = None
                self._focus_tab()
                self._error = (job["node"], i18n._("llm:no-analysis"))
                self._render_current()
                return
        self.queue.pop(0)
        self._start_job(job, cfg)

    def _start_job(self, job, cfg):
        self._status = None
        self._error = None
        language = i18n.lang or "en"
        node = job["node"]
        token = object()
        if node is not None:
            node._llm_token = token
        if job["kind"] == "ask":
            history = list(getattr(node, "llm_thread", None) or [])
            snapshot = describe_position(node, "played")
            messages = followup_messages(history, job["text"], language, snapshot)
            header = None
            kind = "ask"
        else:
            snapshot = describe_position(node, "played")
            if snapshot is None:
                self._error = (node, i18n._("llm:no-analysis"))
                self._render_current()
                return
            header = self._header(snapshot)
            messages = prepare_messages([{"role": "user", "content": build_user_prompt(snapshot, language)}], language)
            kind = "played"
        self._partial = (node, kind, token, header, job["text"], "")
        self._status = (node, "think")
        self.busy = True
        self._render_current()

        def work():
            parts = []
            pending = {"text": "", "scheduled": False}

            def on_delta(piece):
                parts.append(piece)
                pending["text"] = "".join(parts)
                if pending["scheduled"]:
                    return
                pending["scheduled"] = True

                def flush(_dt):
                    pending["scheduled"] = False
                    self._show_partial(node, token, pending["text"])

                Clock.schedule_once(flush, 0.08)

            try:
                text = request_chat(cfg["api_base"], cfg["api_key"], cfg["model"], messages, on_delta=on_delta)
                error = None
            except Exception as exc:
                text = None
                error = str(exc) if str(exc) else exc.__class__.__name__
            Clock.schedule_once(lambda _dt: self._finish(node, token, kind, header, job["text"], text, error), 0)

        threading.Thread(target=work, daemon=True).start()

    def _show_partial(self, node, token, text):
        if getattr(node, "_llm_token", None) is not token or self._partial is None or self._partial[2] is not token:
            return
        node_ref, kind, token_ref, header, question, _old = self._partial
        self._partial = (node_ref, kind, token_ref, header, question, text)
        self._status = None
        if kind != "ask" or question:
            self._feed_speech(node, token, text, final=False)
        if self.view_node is node or self._current_node() is node:
            self.view_node = node
            self._render_current()

    def _finish(self, node, token, kind, header, question, text, error):
        if self._partial and self._partial[2] is token:
            self._partial = None
        self.busy = False
        if self._status and self._status[0] is node:
            self._status = None
        if node is None or getattr(node, "_llm_token", None) is not token:
            self._render_current()
            return
        if error:
            self._error = (node, error)
            self._render_current()
            return
        thread = list(getattr(node, "llm_thread", None) or [])
        if kind == "played":
            thread = replace_commentary(thread, header, text)
        else:
            thread = [
                *thread,
                {"role": "user", "content": question, "kind": "ask"},
                {"role": "assistant", "content": text, "kind": "ask"},
            ]
        self._store_thread(node, thread)
        self._error = None
        if kind == "played" or kind == "ask":
            self._feed_speech(node, token, text, final=True)
        self._render_current()

    def _header(self, snapshot):
        color = i18n._(f"short color {snapshot.get('played_player') or 'B'}")
        number = snapshot.get("move_number")
        header = i18n._("llm:move-header").format(
            number=number if number is not None else "?",
            color=color,
            coord=snapshot.get("played") or "?",
        )
        word = grade_word(snapshot.get("grade"), i18n.lang or "en")
        if word:
            return f"{header}  {word}"
        return header

    def _display_messages(self, node):
        thread = list(getattr(node, "llm_thread", None) or [])
        partial = self._partial if self._partial and self._partial[0] is node else None
        if partial is None:
            return thread
        _node, kind, _token, header, question, text = partial
        if kind == "played":
            kept = [item for item in thread if item.get("kind") != "commentary"]
            shown = [{"role": "user", "content": header, "kind": "commentary"}]
            if text:
                shown.append({"role": "assistant", "content": text, "kind": "commentary"})
            return shown + kept
        shown = list(thread)
        shown.append({"role": "user", "content": question, "kind": "ask"})
        if text:
            shown.append({"role": "assistant", "content": text, "kind": "ask"})
        return shown

    def _render_current(self, *_args):
        node = self.view_node if self.view_node is not None else self._current_node()
        messages = self._display_messages(node) if node is not None else []
        lines = []
        for item in messages:
            kind = "user" if item.get("role") == "user" else "assistant"
            lines.append((kind, item.get("content") or ""))
        if self._status and self._status[0] is node:
            key = "llm:waiting-analysis" if self._status[1] == "wait" else "llm:thinking"
            lines.append(("status", i18n._(key)))
        if self._error and self._error[0] is node:
            lines.append(("error", self._error[1]))
        if not lines:
            lines = [("intro", i18n._("llm:intro"))]
        self.lines = lines
        self._render()

    def _render(self):
        if self.log_label is None:
            return
        blocks = []
        for kind, text in self.lines:
            body = escape_markup(text)
            if kind == "assistant":
                body = self._highlight(body)
            if kind == "user":
                blocks.append(f"[b][color={USER_COLOR}]{escape_markup(i18n._('llm:you'))}[/color][/b]\n{body}")
            elif kind == "assistant":
                blocks.append(f"[b]{escape_markup(i18n._('llm:assistant'))}[/b]\n{body}")
            elif kind == "error":
                blocks.append(f"[color={ERROR_COLOR}]{body}[/color]")
            else:
                blocks.append(f"[color={STATUS_COLOR}]{body}[/color]")
        follow = False
        if self.scroll is not None:
            inner = getattr(self.scroll, "label", None)
            fits = inner is None or inner.height <= self.scroll.height + 2
            follow = should_follow_scroll(
                self.scroll.scroll_y,
                fits,
                getattr(self.scroll, "_touch", None) is not None,
                self._jump_bottom,
            )
        self._jump_bottom = False
        self.log_label.text = "\n\n".join(blocks)
        if follow:
            Clock.schedule_once(self._snap_bottom, 0)

    def _snap_bottom(self, _dt):
        scroll = self.scroll
        if scroll is None or getattr(scroll, "_touch", None) is not None:
            return
        scroll.scroll_y = 0

    def toggle_speech(self):
        if self.speech_state == "speaking":
            self._speaker.pause()
            return
        if self.speech_state == "paused":
            self._speaker.resume()
            return
        if not self._can_speak():
            self._error = (self.view_node if self.view_node is not None else self._current_node(), i18n._("llm:no-speech"))
            self._render_current()
            return
        text = self._assistant_text()
        if not text:
            return
        self._speech_hold = True
        self._apply_speaker_settings()
        for sentence in split_sentences(text):
            self._speaker.enqueue(sentence)

    def _assistant_text(self):
        parts = [text for kind, text in self.lines if kind == "assistant" and text]
        return "\n".join(parts).strip()

    def _feed_speech(self, node, token, text, final):
        if not self._cfg().get("speak") or self._speech_hold and token is self._spoken_token:
            return
        if token is not self._spoken_token:
            self._spoken_token = token
            self._spoken_index = 0
            self._speech_hold = False
            self._speaker.stop()
        if not text:
            return
        if final:
            pending = text[self._spoken_index :]
            self._spoken_index = len(text)
        else:
            pending, self._spoken_index = completed_sentences(text, self._spoken_index)
        if not pending.strip():
            return
        self._apply_speaker_settings()
        for sentence in split_sentences(pending):
            self._speaker.enqueue(sentence)

    def _highlight(self, escaped):
        span = self._speaking_span
        if not span:
            return escaped
        found = highlight_span(escaped, escape_markup(span))
        if found is None:
            return escaped
        before, mid, after = found
        return f"{before}[color={SPEECH_COLOR}]{mid}[/color]{after}"

    def _speech_chunk(self, text):
        Clock.schedule_once(lambda _dt, spoken=text: self._apply_speech_chunk(spoken), 0)

    def _apply_speech_chunk(self, text):
        self._speaking_span = text or None
        self._render_current()

    def _speech_state(self, speaking):
        Clock.schedule_once(lambda _dt: self._apply_speech_state(speaking), 0)

    def _apply_speech_state(self, state):
        self.speech_state = state
        if state == "idle":
            self._speaking_span = None
            self._render_current()

    def _apply_speaker_settings(self):
        cfg = self._cfg()
        self._speaker.lang = i18n.lang or "en"
        self._speaker.api_key = cfg.get("tts_api_key") or ""
        self._speaker.speaker = cfg.get("tts_speaker") or ""

    def _can_speak(self):
        self._apply_speaker_settings()
        return doubao_ready(self._speaker.api_key, self._speaker.speaker) or speech_backend() is not None

    def _store_thread(self, node, thread):
        node.llm_thread = thread
        if thread:
            node.set_property("KTLM", encode_thread(thread))
        else:
            node.clear_property("KTLM")

    def _has_commentary(self, node):
        return any(item.get("kind") == "commentary" for item in (getattr(node, "llm_thread", None) or []))

    def _focus_tab(self):
        katrain = self._katrain()
        panel = getattr(getattr(katrain, "controls", None), "notes_panel", None)
        if panel is None:
            return
        if panel.state != "open":
            panel.state = "open"
        panel.set_option_state({"info": False, "info-details": False, "notes": False, "llm": True})

    def _current_node(self):
        katrain = self._katrain()
        game = getattr(katrain, "game", None) if katrain else None
        return getattr(game, "current_node", None) if game else None

    def _katrain(self):
        app = App.get_running_app()
        return getattr(app, "gui", None) if app else None

    def _cfg(self):
        katrain = self._katrain()
        cfg = katrain.config("llm") if katrain else None
        cfg = cfg or {}
        return {
            "enabled": bool(cfg.get("enabled")),
            "speak": bool(cfg.get("speak", False)),
            "tts_api_key": (cfg.get("tts_api_key") or "").strip(),
            "tts_speaker": (cfg.get("tts_speaker") or "").strip(),
            "api_base": (cfg.get("api_base") or "").strip(),
            "api_key": (cfg.get("api_key") or "").strip(),
            "model": (cfg.get("model") or "").strip(),
        }


__all__ = ["LLMChatPanel"]
