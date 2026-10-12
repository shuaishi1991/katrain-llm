import json
import urllib.request

from katrain.core.llm import (
    analysis_ready,
    build_user_prompt,
    completion_url,
    decode_thread,
    describe_position,
    encode_thread,
    followup_messages,
    move_grade,
    parse_chat_response,
    prepare_messages,
    replace_commentary,
    request_chat,
    settings_ready,
    stream_piece,
)


class _Move:
    def __init__(self, gtp, player):
        self._gtp = gtp
        self.player = player
        self.is_pass = gtp == "pass"

    def gtp(self):
        return self._gtp


class _Node:
    def __init__(self, move=None, parent=None, candidates=None, score=None, winrate=None, points_lost=None, depth=0):
        self.move = move
        self.parent = parent
        self.player = move.player if move else None
        self.next_player = "W" if self.player == "B" else "B"
        self._candidates = candidates or []
        self.score = score
        self.winrate = winrate
        self._points_lost = points_lost
        self.depth = depth
        self.komi = 7.5
        self.board_size = (19, 19)
        self.ownership = None
        self.analysis_exists = score is not None

    @property
    def candidate_moves(self):
        return list(self._candidates)

    @property
    def points_lost(self):
        return self._points_lost

    @property
    def root(self):
        return self.parent.root if self.parent else self

    def get_property(self, key, default=None):
        return {"RU": "japanese"}.get(key, default)


def _cand(move, visits, winrate, score, loss, pv, prior=0.2):
    return {
        "move": move,
        "visits": visits,
        "winrate": winrate,
        "scoreLead": score,
        "pointsLost": loss,
        "prior": prior,
        "pv": pv,
    }


def _position(played="R14", points_lost=1.7):
    candidates = [
        _cand("Q16", 400, 0.56, 4.2, 0.0, ["Q16", "R17", "Q17"], 0.31),
        _cand("R14", 220, 0.53, 2.5, 1.7, ["R14", "D4", "C6"], 0.18),
        _cand("D16", 140, 0.52, 2.1, 2.1, ["D16", "C14", "E16"], 0.11),
        _cand("C16", 40, 0.51, 1.4, 2.8, ["C16", "D14"], 0.04),
        _cand("K10", 25, 0.50, 1.0, 3.2, ["K10", "K4"], 0.02),
        _cand("A1", 5, 0.40, -2.0, 6.2, ["A1"], 0.01),
    ]
    parent = _Node(candidates=candidates, score=4.5, winrate=0.57, depth=11)
    nxt = [
        _cand("F3", 300, 0.55, 3.1, 0.0, ["F3", "E3", "F4"], 0.22),
        _cand("C3", 180, 0.52, 1.4, 1.6, ["C3", "C4"], 0.12),
        _cand("K4", 90, 0.51, 1.1, 2.0, ["K4", "K3"], 0.08),
    ]
    return _Node(
        move=_Move(played, "B"),
        parent=parent,
        candidates=nxt,
        score=2.5,
        winrate=0.53,
        points_lost=points_lost,
        depth=12,
    )


def test_commentary_judges_the_played_move_and_the_next_recommendation():
    node = _position("R14")
    assert analysis_ready(node, "played")
    snapshot = describe_position(node, "played")
    text = build_user_prompt(snapshot, "cn")

    assert "Move just played: move 12, B R14 (KataGo rank 2)." in text
    assert "Objective grade required for the played move: 俗手." in text
    assert "KataGo's best point before this stone was Q16" in text
    assert "rank 2: R14 [this is the move that was played]" in text
    assert "Points given up by the played move: 1.70" in text
    assert "帮助棋手看懂思路、提高下棋的能力" in text
    assert "不要写出目差、胜率" in text
    assert "rank 1: F3 [this is the next move KataGo recommends]" in text
    assert "第二、第三选点" in text
    assert "A1" not in text


def test_large_loss_is_a_blunder_and_a_unique_best_move_is_brilliant():
    crude = move_grade(1.7, 2, 0.0)
    assert crude["cn"] == "俗手"
    blunder = move_grade(6.2, 6, 0.0)
    assert blunder["en"] == "blunder"
    brilliant = move_grade(0.1, 1, 4.0)
    assert brilliant["cn"] == "妙手"
    proper = move_grade(0.1, 1, 0.4)
    assert proper["cn"] == "本手"

    node = _position("A1", points_lost=6.2)
    text = build_user_prompt(describe_position(node, "played"), "en")
    assert "KataGo rank 6" in text
    assert "Objective grade required for the played move: blunder." in text
    assert "rank 6: A1 [this is the move that was played]" in text
    assert "rank 1: F3 [this is the next move KataGo recommends]" in text


def test_candidate_mode_explains_why_the_next_point_is_recommended():
    node = _position("R14").parent
    assert analysis_ready(node, "candidates")
    snapshot = describe_position(node, "candidates")
    text = build_user_prompt(snapshot, "cn")

    assert "No new stone" in text
    assert "推荐的下一手想达到什么" in text
    assert "不要写出目差、胜率" in text
    assert "rank 1: Q16" in text
    assert "this is the move that was played" not in text


def test_missing_katago_candidates_are_not_explained():
    node = _Node(move=_Move("D4", "B"), parent=_Node(), depth=1)
    assert not analysis_ready(node, "played", partial_ok=True)
    assert describe_position(node, "played") is None


def test_followup_uses_this_moves_commentary_as_history():
    history = [
        {"role": "user", "content": "第12手 黑 R14  俗手", "kind": "commentary"},
        {"role": "assistant", "content": "这一手是在拆边。", "kind": "commentary"},
        {"role": "user", "content": "你好", "kind": "ask"},
        {"role": "assistant", "content": "你好，想问这一手的哪里？", "kind": "ask"},
    ]
    messages = followup_messages(history, "第二选点是什么意思？", "cn")

    assert messages[0]["role"] == "system"
    assert "Answer only what the latest message asks" in messages[0]["content"]
    assert "rank 2" not in messages[0]["content"]
    assert [item["content"] for item in messages[1:]] == [
        "第12手 黑 R14  俗手",
        "这一手是在拆边。",
        "你好",
        "你好，想问这一手的哪里？",
        "第二选点是什么意思？",
    ]
    greeting = followup_messages(history, "你好", "cn")
    assert greeting[-1] == {"role": "user", "content": "你好"}
    assert "KataGo" not in greeting[-1]["content"]


def test_scroll_follows_only_while_the_reader_stays_at_the_bottom():
    from katrain.gui.speech import completed_sentences, should_follow_scroll

    assert should_follow_scroll(0.0, False, False, False)
    assert not should_follow_scroll(0.4, False, False, False)
    assert not should_follow_scroll(0.0, False, True, False)
    assert should_follow_scroll(0.8, False, False, True)
    assert not should_follow_scroll(0.8, False, True, True)
    chunk, index = completed_sentences("这一手是在拆边。接下来", 0)
    assert chunk == "这一手是在拆边。"
    assert index == len("这一手是在拆边。")
    assert completed_sentences("还没说完", 0) == ("", 0)
    from katrain.gui.speech import highlight_span, split_sentences

    assert split_sentences("先说这句。再说那句") == ["先说这句。", "再说那句"]
    assert highlight_span("先说这句。再说那句", "再说那句") == ("先说这句。", "再说那句", "")


def test_new_commentary_replaces_the_old_text_and_keeps_questions():
    thread = [
        {"role": "user", "content": "旧标题", "kind": "commentary"},
        {"role": "assistant", "content": "旧解说", "kind": "commentary"},
        {"role": "user", "content": "你好", "kind": "ask"},
        {"role": "assistant", "content": "你好", "kind": "ask"},
    ]
    updated = replace_commentary(thread, "新标题", "新解说")
    assert [item["content"] for item in updated] == ["新标题", "新解说", "你好", "你好"]
    assert decode_thread(encode_thread(updated)) == updated


def test_completion_url_accepts_base_or_full_endpoint():
    assert completion_url("https://api.openai.com/v1") == "https://api.openai.com/v1/chat/completions"
    assert completion_url("https://api.deepseek.com/") == "https://api.deepseek.com/v1/chat/completions"
    assert (
        completion_url("http://127.0.0.1:8000/v1/chat/completions") == "http://127.0.0.1:8000/v1/chat/completions"
    )
    assert (
        completion_url("https://open.bigmodel.cn/api/paas/v4")
        == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    )
    assert (
        completion_url("https://ark.cn-beijing.volces.com/api/coding/v3")
        == "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions"
    )


def test_request_sends_the_key_as_a_bearer_token_and_not_in_the_body(monkeypatch):
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "Q16 是为了拆边。"}}]}).encode()

    def fake_urlopen(request, timeout=0, context=None):
        captured["url"] = request.full_url
        captured["body"] = request.data.decode()
        captured["auth"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    text = request_chat("https://example.com/v1", "secret-key", "demo-model", [{"role": "user", "content": "hi"}])

    assert text == "Q16 是为了拆边。"
    assert captured["auth"] == "Bearer secret-key"
    assert "secret-key" not in captured["body"]
    body = json.loads(captured["body"])
    assert body["model"] == "demo-model"
    assert body["max_tokens"] == 500
    assert "thinking" not in body
    assert "stream" not in body
    assert captured["url"] == "https://example.com/v1/chat/completions"


def test_ark_request_streams_and_skips_the_thinking_step(monkeypatch):
    captured = {}
    seen = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def readline(self):
            if not self.lines:
                return b""
            return self.lines.pop(0)

        lines = [
            'data: {"choices":[{"delta":{"reasoning_content":"先想很久"}}]}\n'.encode(),
            'data: {"choices":[{"delta":{"content":"这一手"}}]}\n'.encode(),
            'data: {"choices":[{"delta":{"content":"是缓手。"}}]}\n'.encode(),
            b"data: [DONE]\n",
        ]

    def fake_urlopen(request, timeout=0, context=None):
        captured["body"] = json.loads(request.data.decode())
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    text = request_chat(
        "https://ark.cn-beijing.volces.com/api/coding/v3",
        "secret-key",
        "ark-code-latest",
        [{"role": "user", "content": "hi"}],
        on_delta=seen.append,
    )

    assert text == "这一手是缓手。"
    assert seen == ["这一手", "是缓手。"]
    assert captured["body"]["stream"] is True
    assert captured["body"]["thinking"] == {"type": "disabled"}
    assert stream_piece({"choices": [{"delta": {"reasoning_content": "hidden"}}]}) == ""


def test_parse_chat_response_accepts_string_and_parts():
    assert parse_chat_response({"choices": [{"message": {"content": "  ok  "}}]}) == "ok"
    payload = {"choices": [{"message": {"content": [{"text": "目"}, {"text": "的"}]}}]}
    assert parse_chat_response(payload) == "目的"


def test_history_keeps_the_system_prompt_and_the_latest_turns():
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(20)]
    trimmed = prepare_messages(messages, "cn")
    assert trimmed[0]["role"] == "system"
    assert "Simplified Chinese" in trimmed[0]["content"]
    assert trimmed[-1]["content"] == "19"
    assert len(trimmed) == 13


def test_pause_freezes_the_current_sentence_and_resume_continues(monkeypatch):
    from katrain.gui import speech

    class Proc:
        def poll(self):
            return None

        def terminate(self):
            self.ended = True

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self.ended = True

        pid = 1

    frozen = []
    monkeypatch.setattr(speech, "speech_backend", lambda: "espeak-ng")
    monkeypatch.setattr(speech, "_freeze", lambda proc, suspend: frozen.append((proc, suspend)))
    states = []
    player = speech.SpeechPlayer(on_state=states.append)
    player._proc = Proc()
    player._current_text = "这一句。"
    player._queue = [(0, "下一句。")]
    player.pause()
    assert states[-1] == "paused"
    assert player._current_text == "这一句。"
    assert player._queue == [(0, "下一句。")]
    assert frozen == [(player._proc, True)]
    player.resume()
    assert states[-1] == "speaking"
    assert frozen[-1] == (player._proc, False)
    player.stop()
    assert states[-1] == "idle"
    assert player._queue == []
    assert player._proc is None


def test_speech_dispatcher_pause_replays_the_current_sentence(monkeypatch):
    from katrain.gui import speech

    class Proc:
        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            return None

        pid = 1

    monkeypatch.setattr(speech, "speech_backend", lambda: "spd-say")
    monkeypatch.setattr(speech, "_freeze", lambda proc, suspend: None)
    monkeypatch.setattr(speech, "_cancel_speech_dispatcher", lambda: None)
    player = speech.SpeechPlayer()
    player._proc = Proc()
    player._current_text = "这一句。"
    player._queue = [(0, "下一句。")]
    player.pause()
    assert player._queue[0] == (0, "这一句。")
    assert player._proc is None
    assert player._paused
    player.stop()


def test_settings_require_endpoint_key_and_model():
    assert not settings_ready({"enabled": True, "api_base": "https://api.openai.com/v1", "api_key": "", "model": "m"})
    assert settings_ready(
        {"enabled": True, "api_base": "https://api.openai.com/v1", "api_key": "k", "model": "demo"}
    )
