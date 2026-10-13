"""Read commentary aloud with Doubao speech, or the operating system voice."""

import base64
import json
import time
import ctypes
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request

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
if ($env:KATRAIN_SPEECH_FILE) {
    $spoken = [IO.File]::ReadAllText($env:KATRAIN_SPEECH_FILE, [Text.UTF8Encoding]::new($false))
} else {
    $spoken = [Console]::In.ReadToEnd()
}
if ($spoken) { $synth.Speak($spoken) }
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
    """Split speech into the pieces that should be highlighted one at a time.

    Walk one sentence ending at a time. A finished commentary contains several
    endings, and taking every ending at once would highlight the whole passage.
    """
    parts = []
    start = 0
    text = text or ""
    while start < len(text):
        end = None
        for index in range(start, len(text)):
            if text[index] in _SENTENCE_END:
                end = index + 1
                break
        if end is None:
            rest = text[start:].strip()
            if rest:
                parts.append(rest)
            break
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        start = end
    return parts


def highlight_span(text, span):
    """Return the text before, inside, and after the passage being read."""
    if not text or not span:
        return None
    index = text.find(span)
    if index < 0:
        return None
    return text[:index], text[index : index + len(span)], text[index + len(span) :]


def _open_audio_player(path, on_event):
    """Open an audio file. The player calls on_event('eof', '') when the sound ends."""
    from ffpyplayer.player import MediaPlayer

    return MediaPlayer(path, callback=on_event, ff_opts={"vn": True, "sn": True})


def _audio_duration(player):
    """File length in seconds, once the player has read it."""
    try:
        duration = (player.get_metadata() or {}).get("duration")
    except Exception:
        return None
    if isinstance(duration, (int, float)) and duration > 0:
        return duration
    return None


DOUBAO_TTS_URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
DOUBAO_RESOURCE_ID = "seed-tts-2.0"


class DoubaoSpeechError(Exception):
    """The Doubao speech request did not return audio."""


def doubao_ready(api_key, speaker):
    """Both the key and a voice id are required before calling Doubao."""
    return bool((api_key or "").strip() and (speaker or "").strip())


def doubao_payload(text, speaker):
    """One finished sentence. speech_rate 0 is the service's normal reading pace."""
    return {
        "req_params": {
            "text": text,
            "speaker": speaker,
            "audio_params": {"format": "mp3", "sample_rate": 24000, "speech_rate": 0},
        }
    }


def audio_from_doubao_event(payload):
    """Audio bytes from one streamed event. An event may carry no audio."""
    code = payload.get("code", 0)
    if code not in (0, 20000000):
        raise DoubaoSpeechError(payload.get("message") or f"speech {code}")
    data = payload.get("data")
    if not data:
        return b""
    try:
        return base64.b64decode(data)
    except (ValueError, TypeError) as exc:
        raise DoubaoSpeechError("bad audio") from exc


def audio_from_doubao_body(raw):
    """Join the mp3 pieces from a body of JSON lines."""
    audio = bytearray()
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DoubaoSpeechError("bad response") from exc
        audio.extend(audio_from_doubao_event(payload))
    if not audio:
        raise DoubaoSpeechError("no audio")
    return bytes(audio)


def fetch_doubao_audio(text, api_key, speaker, timeout=60):
    """Request one sentence and return its mp3 bytes."""
    body = json.dumps(doubao_payload(text, speaker), ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        DOUBAO_TTS_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Api-Key": api_key.strip(),
            "X-Api-Resource-Id": DOUBAO_RESOURCE_ID,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise DoubaoSpeechError(f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise DoubaoSpeechError("network") from exc
    return audio_from_doubao_body(raw.decode("utf-8", errors="replace"))


def _system_backend():
    if sys.platform == "win32":
        return "powershell" if shutil.which("powershell") else None
    if shutil.which("espeak-ng"):
        return "espeak-ng"
    if shutil.which("espeak"):
        return "espeak"
    if shutil.which("spd-say"):
        return "spd-say"
    return None


def speech_backend():
    return _system_backend()


def _hidden_window_kwargs():
    """Keep a speech process from flashing a console window on Windows."""
    if sys.platform != "win32":
        return {}
    kwargs = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    if hasattr(subprocess, "STARTUPINFO"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        kwargs["startupinfo"] = startupinfo
    return kwargs


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
        self.api_key = ""
        self.speaker = ""
        self._prepared = {}
        self._preparing = set()
        self._on_state = on_state
        self._on_chunk = on_chunk
        self._token = 0
        self._queue = []
        self._lock = threading.Lock()
        self._proc = None
        self._player = None
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
            player = self._player
            self._player = None
            prepared = list(self._prepared.values())
            self._prepared.clear()
            self._preparing.clear()
        self._close_player(player)
        for path in prepared:
            try:
                os.remove(path)
            except OSError:
                pass
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
            if self._proc is None and self._player is None and not self._queue and self._current_text is None:
                return
            self._paused = True
            self._gate.clear()
            proc = self._proc
            player = self._player
            current = self._current_text
            token = self._token
            replay = speech_backend() == "spd-say" and current
            if replay:
                self._queue.insert(0, (token, current))
                self._proc = None
                self._current_text = None
        if player is not None:
            try:
                player.set_pause(True)
            except Exception:
                pass
        elif replay:
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
            player = self._player
            self._gate.set()
        if player is not None:
            try:
                player.set_pause(False)
            except Exception:
                pass
        else:
            _freeze(proc, False)
        self._emit("speaking")

    def enqueue(self, text):
        text = (text or "").strip()
        if not text or not (doubao_ready(self.api_key, self.speaker) or speech_backend()):
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
                    # Release this thread before unlocking. A sentence that arrives
                    # as the queue runs dry must be able to start another thread.
                    if self._worker is threading.current_thread():
                        self._worker = None
                    break
                token, text = self._queue.pop(0)
                if token != self._token:
                    continue
                self._current_text = text
            self._speak(token, text)
            with self._lock:
                if self._current_text == text:
                    self._current_text = None
        with self._lock:
            still_idle = self._worker is None and not self._queue
        if still_idle:
            self._emit("idle")
            self._emit_chunk(None)

    def _close_player(self, player):
        if player is None:
            return
        try:
            player.close_player()
        except Exception:
            pass

    def _prefetch_next(self, token):
        """Synthesize the next sentence while this one is playing."""
        if not doubao_ready(self.api_key, self.speaker):
            return
        with self._lock:
            nxt = next((item for item in self._queue if item[0] == token), None)
            if nxt is None or nxt in self._prepared or nxt in self._preparing:
                return
            self._preparing.add(nxt)
            api_key = self.api_key
            speaker = self.speaker
        _token, text = nxt

        def work():
            handle, path = tempfile.mkstemp(prefix="katrain-speech-", suffix=".mp3")
            os.close(handle)
            try:
                audio = fetch_doubao_audio(text, api_key, speaker)
                with open(path, "wb") as output:
                    output.write(audio)
            except (DoubaoSpeechError, OSError):
                try:
                    os.remove(path)
                except OSError:
                    pass
                with self._lock:
                    self._preparing.discard(nxt)
                return
            with self._lock:
                self._preparing.discard(nxt)
                if self._token == token:
                    self._prepared[nxt] = path
                else:
                    try:
                        os.remove(path)
                    except OSError:
                        pass

        threading.Thread(target=work, daemon=True).start()

    def _wait_prepared(self, token, text):
        """Use audio that is already being fetched, so the sentence is not billed twice.

        Returns the file path, None when nothing is in flight, or False when this
        reading was stopped.
        """
        key = (token, text)
        deadline = time.time() + 90
        while time.time() < deadline:
            with self._lock:
                if token != self._token:
                    return False
                if key in self._prepared:
                    return self._prepared.pop(key)
                preparing = key in self._preparing
            if not preparing:
                return None
            time.sleep(0.05)
        with self._lock:
            if token != self._token:
                return False
            return self._prepared.pop(key, None)

    def _speak_doubao(self, token, text):
        """Speak one sentence with Doubao. Returns False when synthesis fails."""
        self._prefetch_next(token)
        prepared = self._wait_prepared(token, text)
        if prepared is False:
            return True
        path = prepared
        if path is None:
            handle, path = tempfile.mkstemp(prefix="katrain-speech-", suffix=".mp3")
            os.close(handle)
            try:
                audio = fetch_doubao_audio(text, self.api_key, self.speaker)
                with open(path, "wb") as output:
                    output.write(audio)
            except (DoubaoSpeechError, OSError):
                try:
                    os.remove(path)
                except OSError:
                    pass
                return False
        with self._lock:
            if token != self._token or self._paused:
                try:
                    os.remove(path)
                except OSError:
                    pass
                if self._paused and token == self._token:
                    self._queue.insert(0, (token, text))
                return True
        finished = threading.Event()

        def on_event(selector, _value):
            # The player reports eof when the audio itself has finished.
            # get_frame() only reports that for a file that contains video.
            if selector in ("eof", b"eof"):
                finished.set()

        try:
            player = _open_audio_player(path, on_event)
        except Exception:
            try:
                os.remove(path)
            except OSError:
                pass
            return False
        with self._lock:
            if token != self._token or self._paused:
                self._close_player(player)
                try:
                    os.remove(path)
                except OSError:
                    pass
                if self._paused and token == self._token:
                    self._queue.insert(0, (token, text))
                return True
            self._player = player
        self._emit_chunk(text)
        paused = False
        played = 0.0
        mark = time.monotonic()
        while not finished.wait(0.05):
            now = time.monotonic()
            with self._lock:
                if token != self._token or self._player is not player:
                    break
                want_pause = self._paused
            if want_pause != paused:
                try:
                    player.set_pause(want_pause)
                except Exception:
                    break
                paused = want_pause
            if not paused:
                played += now - mark
                duration = _audio_duration(player)
                # The eof callback is the end of the sentence. This limit only
                # moves on if that callback never arrives.
                if duration is not None and played >= duration + 2:
                    break
                if duration is None and played >= 10:
                    break
            mark = now
        with self._lock:
            owned = self._player is player
            if owned:
                self._player = None
        if owned:
            self._close_player(player)
        try:
            os.remove(path)
        except OSError:
            pass
        return True

    def _speak(self, token, text):
        if doubao_ready(self.api_key, self.speaker) and self._speak_doubao(token, text):
            return
        backend = _system_backend()
        command = _command(backend, self.lang)
        if command is None:
            return
        env = os.environ.copy()
        env["KATRAIN_SPEECH_CULTURE"] = _CULTURE.get(self.lang, "en-US")
        speech_file = None
        # A hidden PowerShell window has no console, so reading stdin there
        # never sees the end of the text and the next sentence never starts.
        if backend == "powershell":
            handle, speech_file = tempfile.mkstemp(prefix="katrain-speech-", suffix=".txt")
            with os.fdopen(handle, "w", encoding="utf-8") as output:
                output.write(text)
            env["KATRAIN_SPEECH_FILE"] = speech_file
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL if backend == "powershell" else subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                **_hidden_window_kwargs(),
            )
        except OSError:
            if speech_file:
                os.remove(speech_file)
            return
        with self._lock:
            if token != self._token or self._paused:
                self._kill(proc)
                if self._paused and token == self._token and speech_backend() != "spd-say":
                    self._queue.insert(0, (token, text))
                if speech_file:
                    try:
                        os.remove(speech_file)
                    except OSError:
                        pass
                return
            self._proc = proc
        self._emit_chunk(text)
        if backend != "powershell":
            try:
                proc.stdin.write(text.encode("utf-8"))
                proc.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                self._kill(proc)
        proc.wait()
        with self._lock:
            if self._proc is proc:
                self._proc = None
        if speech_file:
            try:
                os.remove(speech_file)
            except OSError:
                pass

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
