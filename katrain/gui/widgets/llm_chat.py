"""Chat panel that explains KataGo's move choices."""

import ctypes
import glob
import os
import sys
import threading
import time

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.properties import BooleanProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput

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
    doubao_ready,
    paragraph_parts,
    should_follow_scroll,
    speech_backend,
)
from katrain.gui.theme import Theme, to_hexcol

ANALYSIS_WAIT = 8.0

USER_COLOR = to_hexcol(Theme.SCORE_COLOR)
STATUS_COLOR = to_hexcol(Theme.POINTLOSS_COLOR)
ERROR_COLOR = to_hexcol(Theme.ERROR_BORDER_COLOR)
SPEECH_COLOR = to_hexcol([0.96, 0.78, 0.42, 1])
ICON_COLOR = to_hexcol([0.62, 0.66, 0.70, 1])
SPEAKER_ICON = "[font=speaker.ttf][size=18]\ue000[/size][/font]"


def escape_markup(text):
    return (text or "").replace("&", "&amp;").replace("[", "\\[").replace("]", "\\]")


def ime_anchor(cursor_x, line_top, line_height, window_height, candidate_room=220):
    """Caret rectangle for the input method, origin at the top-left of the window.

    Kivy measures the line from the bottom of the window. The candidate list
    opens underneath this rectangle. The question box sits on the bottom edge,
    so the list is shifted up while the caret is too low to leave room for it.
    """
    height = max(int(line_height), 1)
    top = int(line_top)
    y = max(0, int(window_height) - top)
    if y + height + candidate_room > int(window_height):
        y = max(0, y - candidate_room)
    return int(cursor_x), y, height, height


class _SDLRect(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("w", ctypes.c_int),
        ("h", ctypes.c_int),
    ]


def _sdl_library():
    """The SDL2 that this process already uses to draw the window."""
    cached = getattr(_sdl_library, "lib", None)
    if cached is not None:
        return cached or None
    paths = []
    if sys.platform == "win32":
        kernel = ctypes.windll.kernel32
        kernel.GetModuleHandleW.restype = ctypes.c_void_p
        handle = kernel.GetModuleHandleW("SDL2.dll")
        if handle:
            library = ctypes.WinDLL("SDL2.dll", handle=handle)
            _sdl_library.lib = library
            return library
        try:
            import kivy_deps.sdl2 as sdl2_dep

            paths.extend(glob.glob(os.path.join(os.path.dirname(sdl2_dep.__file__), "**", "SDL2.dll"), recursive=True))
        except Exception:
            pass
        paths.append("SDL2.dll")
    else:
        import kivy

        bundled = os.path.abspath(os.path.join(os.path.dirname(kivy.__file__), os.pardir, "Kivy.libs"))
        paths.extend(sorted(glob.glob(os.path.join(bundled, "libSDL2-2*.so*"))))
        paths.append("libSDL2-2.0.so.0")
    for path in paths:
        try:
            library = ctypes.CDLL(path)
        except OSError:
            continue
        if hasattr(library, "SDL_SetTextInputRect"):
            _sdl_library.lib = library
            return library
    _sdl_library.lib = False
    return None


def _enable_candidate_list(library):
    """Let the input method draw the candidate list beside the caret.

    Sending the composition into the text box makes fcitx show only the
    pinyin. The list of characters is the input method's own window.
    """
    if not hasattr(library, "SDL_SetHint"):
        return
    library.SDL_SetHint.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.SDL_SetHint.restype = ctypes.c_int
    library.SDL_SetHint(b"SDL_IME_SHOW_UI", b"1")
    library.SDL_SetHint(b"SDL_IME_INTERNAL_EDITING", b"1")


def place_ime_cursor(cursor_x, line_top, line_height, window_height):
    library = _sdl_library()
    if library is None:
        return
    _enable_candidate_list(library)
    x, y, width, height = ime_anchor(cursor_x, line_top, line_height, window_height)
    rect = _SDLRect(x, y, width, height)
    library.SDL_SetTextInputRect(ctypes.byref(rect))


class ChatInput(TextInput):
    """One-line question box that can show the chosen Chinese characters.

    The candidate list belongs to the input method. This box only receives a
    character after it is chosen, so it needs a font that contains Chinese
    and a caret position the input method can place its list against.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("font_name", i18n.font_name)
        super().__init__(**kwargs)
        self.bind(focus=self._place_ime, cursor_pos=self._place_ime, pos=self._place_ime, size=self._place_ime)

    def _place_ime(self, *_args):
        if not self.focus:
            return
        place_ime_cursor(self.cursor_pos[0], self.cursor_pos[1], max(self.line_height, self.font_size), Window.height)


def assistant_markup(text, complete, speakable, speaking_index, speaking_text):
    """One speaker after each finished paragraph. The open paragraph has none."""
    finished, tail = paragraph_parts(text, complete)
    blocks = []
    for paragraph in finished:
        index = len(speakable)
        speakable.append(paragraph)
        body = escape_markup(paragraph)
        active = index == speaking_index and paragraph == speaking_text
        if active:
            body = f"[color={SPEECH_COLOR}]{body}[/color]"
        color = SPEECH_COLOR if active else ICON_COLOR
        blocks.append(f"{body}\n[ref={index}][color={color}]{SPEAKER_ICON}[/color][/ref]")
    if tail:
        blocks.append(escape_markup(tail))
    return "\n\n".join(blocks)


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
        self._speakable = []
        self._speaking_index = None
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
        self._speaker.stop()
        self._render_current()

    def show_node(self, node):
        if node is self.view_node:
            return
        self.view_node = node
        self._jump_bottom = True
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
        partial = self._partial if self._partial and self._partial[0] is node and self._partial[5] else None
        lines = []
        matches = []
        for item in messages:
            kind = "user" if item.get("role") == "user" else "assistant"
            text = item.get("content") or ""
            lines.append((kind, text))
            if partial and kind == "assistant" and text == partial[5]:
                matches.append(len(lines) - 1)
        if not partial or not matches:
            streaming_at = None
        elif partial[1] == "ask":
            streaming_at = matches[-1]
        else:
            streaming_at = matches[0]
        if self._status and self._status[0] is node:
            key = "llm:waiting-analysis" if self._status[1] == "wait" else "llm:thinking"
            lines.append(("status", i18n._(key)))
        if self._error and self._error[0] is node:
            lines.append(("error", self._error[1]))
        if not lines:
            lines = [("intro", i18n._("llm:intro"))]
        self.lines = lines
        self._streaming_at = streaming_at
        self._render()

    def _render(self):
        if self.log_label is None:
            return
        speakable = []
        blocks = []
        streaming_at = getattr(self, "_streaming_at", None)
        for index, (kind, text) in enumerate(self.lines):
            if kind == "assistant":
                body = assistant_markup(
                    text,
                    index != streaming_at,
                    speakable,
                    self._speaking_index,
                    self._speaking_span,
                )
            else:
                body = escape_markup(text)
            if kind == "user":
                blocks.append(f"[b][color={USER_COLOR}]{escape_markup(i18n._('llm:you'))}[/color][/b]\n{body}")
            elif kind == "assistant":
                blocks.append(f"[b]{escape_markup(i18n._('llm:assistant'))}[/b]\n{body}")
            elif kind == "error":
                blocks.append(f"[color={ERROR_COLOR}]{body}[/color]")
            else:
                blocks.append(f"[color={STATUS_COLOR}]{body}[/color]")
        self._speakable = speakable
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

    def read_paragraph(self, ref):
        """Read the paragraph under the clicked speaker, and nothing after it."""
        try:
            index = int(ref)
            text = self._speakable[index]
        except (TypeError, ValueError, IndexError):
            return
        if self.speech_state == "speaking" and index == self._speaking_index and text == self._speaking_span:
            self._speaker.stop()
            return
        if not self._can_speak():
            self._error = (self.view_node if self.view_node is not None else self._current_node(), i18n._("llm:no-speech"))
            self._render_current()
            return
        self._speaker.stop(announce=False)
        self._speaking_index = index
        self._speaking_span = text
        self._apply_speaker_settings()
        self._speaker.enqueue(text)
        self._render_current()

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
            self._speaking_index = None
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
            "tts_api_key": (cfg.get("tts_api_key") or "").strip(),
            "tts_speaker": (cfg.get("tts_speaker") or "").strip(),
            "api_base": (cfg.get("api_base") or "").strip(),
            "api_key": (cfg.get("api_key") or "").strip(),
            "model": (cfg.get("model") or "").strip(),
        }


__all__ = ["LLMChatPanel"]
