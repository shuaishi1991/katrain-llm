"""Read commentary aloud with the operating system's speech tools."""

import ctypes
import os
import shutil
import signal
import subprocess
import sys
import threading

_SENTENCE_END = set("。！？!?；;\n")

_CULTURE = {
    "cn": "zh-CN",
    "tw": "zh-TW",
    "jp": "ja-JP",
    "ko": "ko-KR",
    "en": "en-US",
    "de": "de-DE",
    "fr": "fr-FR",
    "ru": "ru-RU",
    "ua": "uk-UA",
    "tr": "tr-TR",
}

_ESPEAK_VOICE = {
    "cn": "cmn",
    "tw": "cmn",
    "jp": "ja",
    "ko": "ko",
    "en": "en",
    "de": "de",
    "fr": "fr",
    "ru": "ru",
    "tr": "tr",
}

_SPD_LANG = {"cn": "zh", "tw": "zh", "jp": "ja", "ko": "ko", "en": "en", "de": "de", "fr": "fr", "ru": "ru", "tr": "tr"}

_WINDOWS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = New-Object System.Text.UTF8Encoding $false
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $culture = [System.Globalization.CultureInfo]::new($env:KATRAIN_SPEECH_CULTURE)
    $synth.SelectVoiceByHints(
        [System.Speech.Synthesis.VoiceGender]::NotSet,
        [System.Speech.Synthesis.VoiceAge]::NotSet,
        0,
        $culture
    )
} catch {}
$synth.Speak([Console]::In.ReadToEnd())
"""


def should_follow_scroll(scroll_y, content_fits, dragging, jump):
    """Follow new text only when the reader is already at the bottom."""
    if jump:
        return not dragging
    if dragging:
        return False
    if content_fits:
        return True
    return scroll_y <= 0.04


def completed_sentences(text, start):
    """Return the newly finished sentences and the index just after them."""
    if start < 0:
        start = 0
    if start >= len(text):
        return "", start
    last = None
    for index, char in enumerate(text[start:], start):
        if char in _SENTENCE_END:
            last = index + 1
    if last is None:
        return "", start
    return text[start:last], last


def split_sentences(text):
    """Split speech into the pieces that should be highlighted one at a time."""
    parts = []
    start = 0
    text = text or ""
    while start < len(text):
        chunk, index = completed_sentences(text, start)
        if not chunk or index <= start:
            rest = text[start:].strip()
            if rest:
                parts.append(rest)
            break
        piece = chunk.strip()
        if piece:
            parts.append(piece)
        start = index
    return parts


def highlight_span(text, span):
    """Return the text before, inside, and after the passage being read."""
    if not text or not span:
        return None
    index = text.find(span)
    if index < 0:
        return None
    return text[:index], text[index : index + len(span)], text[index + len(span) :]


def speech_backend():
    if sys.platform == "win32":
        return "powershell" if shutil.which("powershell") else None
    if shutil.which("espeak-ng"):
        return "espeak-ng"
    if shutil.which("espeak"):
        return "espeak"
    if shutil.which("spd-say"):
        return "spd-say"
    return None


def _command(backend, lang):
    if backend == "powershell":
        return ["powershell", "-NoProfile", "-Command", _WINDOWS_SCRIPT]
    if backend == "espeak-ng":
        return ["espeak-ng", "-v", _ESPEAK_VOICE.get(lang, "en"), "--stdin"]
    if backend == "espeak":
        return ["espeak", "--stdin"]
    if backend == "spd-say":
        return ["spd-say", "-l", _SPD_LANG.get(lang, "en"), "-w", "-e"]
    return None


def _cancel_speech_dispatcher():
    if not shutil.which("spd-say"):
        return
    subprocess.run(["spd-say", "-S"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2, check=False)


class SpeechPlayer:
    """Queue sentences and speak them one after another.

    pause() freezes the current sentence. resume() continues it. stop() drops the rest.
    """

    def __init__(self, on_state=None, on_chunk=None):
        self.lang = "en"
        self._on_state = on_state
        self._on_chunk = on_chunk
        self._token = 0
        self._queue = []
        self._lock = threading.Lock()
        self._proc = None
        self._worker = None
        self._paused = False
        self._current_text = None
        self._gate = threading.Event()
        self._gate.set()

    def stop(self):
        with self._lock:
            self._token += 1
            self._paused = False
            self._gate.set()
            self._queue.clear()
            self._current_text = None
            proc = self._proc
            self._proc = None
        self._kill(proc)
        try:
            _cancel_speech_dispatcher()
        except (OSError, subprocess.TimeoutExpired):
            pass
        self._emit("idle")
        self._emit_chunk(None)

    def pause(self):
        with self._lock:
            if self._paused:
                return
            if self._proc is None and not self._queue and self._current_text is None:
                return
            self._paused = True
            self._gate.clear()
            proc = self._proc
            current = self._current_text
            token = self._token
            replay = speech_backend() == "spd-say" and current
            if replay:
                self._queue.insert(0, (token, current))
                self._proc = None
                self._current_text = None
        if replay:
            self._kill(proc)
            try:
                _cancel_speech_dispatcher()
            except (OSError, subprocess.TimeoutExpired):
                pass
        else:
            _freeze(proc, True)
        self._emit("paused")

    def resume(self):
        with self._lock:
            if not self._paused:
                return
            self._paused = False
            proc = self._proc
            self._gate.set()
        _freeze(proc, False)
        self._emit("speaking")

    def enqueue(self, text):
        text = (text or "").strip()
        if not text or speech_backend() is None:
            return False
        with self._lock:
            self._queue.append((self._token, text))
            paused = self._paused
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, daemon=True)
                self._worker.start()
        if not paused:
            self._emit("speaking")
        return True

    def _run(self):
        while True:
            self._gate.wait()
            with self._lock:
                if self._paused:
                    continue
                if not self._queue:
                    idle = self._worker is threading.current_thread()
                    break
                token, text = self._queue.pop(0)
                if token != self._token:
                    continue
                self._current_text = text
            self._speak(token, text)
            with self._lock:
                if self._current_text == text:
                    self._current_text = None
        if idle:
            self._emit("idle")
            self._emit_chunk(None)

    def _speak(self, token, text):
        backend = speech_backend()
        command = _command(backend, self.lang)
        if command is None:
            return
        env = os.environ.copy()
        env["KATRAIN_SPEECH_CULTURE"] = _CULTURE.get(self.lang, "en-US")
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
        except OSError:
            return
        with self._lock:
            if token != self._token or self._paused:
                self._kill(proc)
                if self._paused and token == self._token and speech_backend() != "spd-say":
                    self._queue.insert(0, (token, text))
                return
            self._proc = proc
        self._emit_chunk(text)
        try:
            proc.stdin.write(text.encode("utf-8"))
            proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            self._kill(proc)
        proc.wait()
        with self._lock:
            if self._proc is proc:
                self._proc = None

    def _emit(self, state):
        if self._on_state is not None:
            self._on_state(state)

    def _emit_chunk(self, text):
        if self._on_chunk is not None:
            self._on_chunk(text)

    @staticmethod
    def _kill(proc):
        if proc is None or proc.poll() is not None:
            return
        _freeze(proc, False)
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()


def _freeze(proc, suspend):
    """Pause or continue the speech process itself, so playback stops mid-sentence."""
    if proc is None or proc.poll() is not None:
        return
    if sys.platform == "win32":
        _windows_freeze(proc.pid, suspend)
        return
    try:
        os.kill(proc.pid, signal.SIGSTOP if suspend else signal.SIGCONT)
    except OSError:
        pass


def _windows_freeze(pid, suspend):
    process_suspend_resume = 0x0800
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    handle = kernel32.OpenProcess(process_suspend_resume, False, pid)
    if not handle:
        return
    try:
        (ntdll.NtSuspendProcess if suspend else ntdll.NtResumeProcess)(handle)
    finally:
        kernel32.CloseHandle(handle)
