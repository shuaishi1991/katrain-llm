"""OpenAI-compatible commentary grounded in KataGo analysis."""

import json
import ssl
import urllib.error
import urllib.request
from urllib.parse import urlsplit

try:
    import certifi
except ImportError:  # pragma: no cover - certifi is a runtime dependency
    certifi = None

TOP_CANDIDATES = 3
PV_PLIES = 6
REPLY_TOKENS = 500
HISTORY_TURNS = 12
REQUEST_TIMEOUT = 90

LANGUAGE_NAMES = {
    "en": "English",
    "cn": "Simplified Chinese",
    "tw": "Traditional Chinese",
    "jp": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "ua": "Ukrainian",
    "tr": "Turkish",
    "es": "Spanish",
}


class LLMError(Exception):
    pass


def language_name(code):
    return LANGUAGE_NAMES.get(code or "en", "English")


def completion_url(api_base):
    """Turn a user-entered API base into a chat-completions URL.

    A bare host such as https://api.openai.com gets /v1/chat/completions.
    A host that already has a path, including /v1 or /api/paas/v4, only gets
    /chat/completions, so a non-OpenAI prefix is not turned into a 404.
    """
    base = (api_base or "").strip().rstrip("/")
    if not base:
        raise LLMError("API base URL is empty")
    if base.endswith("/chat/completions"):
        return base
    path = urlsplit(base).path.rstrip("/")
    if path in ("", "/"):
        return base + "/v1/chat/completions"
    return base + "/chat/completions"


def settings_ready(cfg):
    return bool(cfg.get("enabled") and cfg.get("api_base") and cfg.get("api_key") and cfg.get("model"))


def analysis_ready(node, mode, partial_ok=False):
    """Whether KataGo has enough data to explain this node.

    Judging the move just played needs the candidates from before that stone.
    A full comment also waits for the evaluation after the stone and for
    KataGo's recommendation for the next move. After the wait, the played
    move can still be judged without that recommendation.
    """
    if node is None:
        return False
    if mode == "candidates":
        return bool(getattr(node, "candidate_moves", None))
    parent = getattr(node, "parent", None)
    move = getattr(node, "move", None)
    if not parent or not move or not getattr(parent, "candidate_moves", None):
        return False
    if partial_ok:
        return True
    return bool(getattr(node, "analysis_exists", False) and getattr(node, "candidate_moves", None))


def describe_position(node, mode):
    """Snapshot the KataGo facts the commentary is allowed to use.

    Returns None when KataGo has not produced the candidate list yet.
    """
    if not analysis_ready(node, mode, partial_ok=True):
        return None
    if mode == "played":
        source = node.parent
        played = node.move.gtp()
        played_player = node.move.player
    else:
        source = node
        played = None
        played_player = getattr(node, "next_player", None)
    candidates = list(source.candidate_moves)
    chosen = _select_candidates(candidates, played)
    if not chosen:
        return None
    root = getattr(node, "root", node)
    rules = "japanese"
    if hasattr(root, "get_property"):
        rules = root.get_property("RU", rules)
    size = getattr(node, "board_size", None) or getattr(root, "board_size", None)
    return {
        "mode": mode,
        "move_number": getattr(node, "depth", None),
        "played": played,
        "played_player": played_player,
        "played_rank": _rank_of(candidates, played) if played else None,
        "to_move": getattr(node, "next_player", None),
        "board_size": size,
        "komi": getattr(node, "komi", None),
        "rules": rules,
        "before": _eval_of(source) if mode == "played" else _eval_of(node),
        "after": _eval_of(node) if mode == "played" else None,
        "points_lost": getattr(node, "points_lost", None) if mode == "played" else None,
        "grade": _grade_for_played(node, candidates, played) if mode == "played" else None,
        "best_other": _best_other(candidates, played) if mode == "played" else None,
        "ownership_sum": _ownership_sum(node if mode == "played" else source),
        "recent": _recent_moves(node if mode == "played" else source),
        "candidates": [_candidate_view(cand, index, candidates) for index, cand in enumerate(chosen, start=1)],
        "next_moves": _next_moves(node) if mode == "played" else [],
    }


def build_user_prompt(snapshot, language_code):
    lines = [
        "Use only the KataGo analysis below. Do not invent moves, variations, or evaluations.",
        "",
        _game_header(snapshot),
        "",
        _field_meanings(),
        "",
        _decision_header(snapshot, language_code),
        "",
        _candidate_section(snapshot),
    ]
    if snapshot["mode"] == "played":
        lines.append("")
        lines.append(_next_section(snapshot))
    if snapshot.get("recent"):
        lines.append("")
        lines.append("Recent moves, for context only, not extra analysis: " + ", ".join(snapshot["recent"]))
    lines.append("")
    lines.append(_task_instructions(snapshot["mode"], language_code))
    return "\n".join(lines)


def followup_system(language_code):
    name = language_name(language_code)
    return (
        "You are a Go teacher continuing a conversation about one move. "
        "The earlier messages are the commentary and questions already written for this move. "
        "Answer only what the latest message asks. "
        "A greeting gets a short greeting and nothing about the board. "
        "Do not start a new commentary unless the user asks for one. "
        "Do not write calculation numbers. "
        f"Reply in {name}."
    )


def followup_messages(history, question, language_code):
    """Continue this move's commentary. The history is that text, not a new analysis."""
    question = (question or "").strip()
    prior = [
        {"role": item["role"], "content": item["content"]}
        for item in history or []
        if item.get("role") in ("user", "assistant") and item.get("content")
    ]
    return [
        {"role": "system", "content": followup_system(language_code)},
        *prior[-HISTORY_TURNS:],
        {"role": "user", "content": question},
    ]


def replace_commentary(thread, header, text):
    """Replace this move's generated commentary and keep the questions asked after it."""
    rest = [item for item in thread or [] if item.get("kind") != "commentary"]
    return [
        {"role": "user", "content": header, "kind": "commentary"},
        {"role": "assistant", "content": text, "kind": "commentary"},
        *rest,
    ]


def encode_thread(thread):
    return json.dumps({"v": 1, "messages": list(thread or [])}, ensure_ascii=False, separators=(",", ":"))


def decode_thread(raw):
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list):
        return []
    thread = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content:
            continue
        kind = item.get("kind") if item.get("kind") in ("commentary", "ask") else "ask"
        thread.append({"role": role, "content": content, "kind": kind})
    return thread


def system_prompt(language_code):
    name = language_name(language_code)
    return (
        "You are a Go teacher inside KaTrain. Your commentary exists to teach a human: "
        "help them understand how to think about the position and become a stronger player. "
        "Use only the KataGo analysis in the conversation to decide what is true. "
        "Teach the idea behind the move just played, then the idea behind KataGo's recommended next point "
        "and the following candidate points: the meaning, purpose, and effect, and what a player should learn from the choice. "
        "Use the objective grade supplied with the analysis, and do not make it kinder or harsher. "
        "Do not write calculation numbers: no win rates, point losses, score leads, visit counts, or priors. "
        "Speak in ordinary Go language, as a teacher to a student. "
        "If the analysis is incomplete, say what is missing instead of guessing. "
        "Keep it short: at most four sentences for each part, and do not narrate every move of a variation. "
        f"Reply in {name}."
    )


def prepare_messages(messages, language_code):
    rest = [msg for msg in messages if msg.get("role") != "system"]
    rest = rest[-HISTORY_TURNS:]
    return [{"role": "system", "content": system_prompt(language_code)}, *rest]


def _disable_thinking(api_base):
    """Ark coding models think for a long time before the first word."""
    host = (urlsplit(api_base or "").hostname or "").lower()
    return host.endswith("volces.com") or host.endswith("volcengine.com")


def request_chat(api_base, api_key, model, messages, timeout=REQUEST_TIMEOUT, on_delta=None):
    extra = {"thinking": {"type": "disabled"}} if _disable_thinking(api_base) else None
    try:
        return _request_chat(api_base, api_key, model, messages, timeout, on_delta, extra)
    except LLMError as exc:
        if extra is None or not str(exc).startswith("HTTP 400"):
            raise
        return _request_chat(api_base, api_key, model, messages, timeout, on_delta, None)


def _request_chat(api_base, api_key, model, messages, timeout, on_delta, extra):
    url = completion_url(api_base)
    payload = {"model": model, "messages": messages, "temperature": 0.4, "max_tokens": REPLY_TOKENS}
    if extra:
        payload.update(extra)
    if on_delta is not None:
        payload["stream"] = True
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if on_delta is not None else "application/json",
        },
        method="POST",
    )
    context = ssl.create_default_context(cafile=certifi.where()) if certifi else None
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            if on_delta is None:
                return _parse_body(response.read().decode("utf-8"))
            return _read_stream(response, on_delta)
    except urllib.error.HTTPError as exc:
        raise _http_error(url, exc) from None
    except urllib.error.URLError as exc:
        raise LLMError(str(exc.reason)) from None


def _http_error(url, exc):
    detail = exc.read().decode("utf-8", "replace")[:300].strip()
    if exc.code == 404:
        lowered = detail.lower()
        if "model" in lowered and any(word in lowered for word in ("not found", "does not exist", "model_not_found")):
            return LLMError(f"HTTP 404：模型不存在。请核对模型名称。{detail}")
        message = (
            f"HTTP 404：这个地址没有解说接口：{url}。"
            "API 地址填到服务商给出的前缀即可，例如 https://api.openai.com/v1；"
            "路径不是 /v1 时，填到该路径末尾，不要自己再加 /chat/completions。"
        )
        if detail:
            message = f"{message} {detail}"
        return LLMError(message)
    extra = f": {detail}" if detail else ""
    return LLMError(f"HTTP {exc.code}{extra}")


def _parse_body(body):
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Could not read the model response: {exc}") from None
    return parse_chat_response(parsed)


def _read_stream(response, on_delta):
    parts = []
    while True:
        raw = response.readline()
        if not raw:
            break
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        if line.startswith("{") and not parts:
            rest = response.read().decode("utf-8", "replace")
            return _parse_body(line + rest)
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        piece = stream_piece(event)
        if not piece:
            continue
        parts.append(piece)
        on_delta(piece)
    text = "".join(parts).strip()
    if not text:
        raise LLMError("The model returned an empty commentary")
    return text


def stream_piece(event):
    try:
        choice = event["choices"][0]
    except (KeyError, IndexError, TypeError):
        return ""
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        delta = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return ""


def parse_chat_response(payload):
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected model response: {payload}") from exc
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
        text = "".join(parts).strip()
    else:
        text = ""
    if not text:
        raise LLMError("The model returned an empty commentary")
    return text


def _grade_for_played(node, candidates, played):
    other = _best_other(candidates, played)
    second_loss = None if other is None else other.get("pointsLost")
    return move_grade(getattr(node, "points_lost", None), _rank_of(candidates, played), second_loss)


def move_grade(points_lost, rank, second_loss):
    """Objective grade for the move just played, fixed by KataGo's point loss."""
    if points_lost is None:
        return {
            "key": "unknown",
            "en": "unrated",
            "cn": "未评",
            "tw": "未評",
            "reason": "KataGo has not measured the point loss, so do not call the move brilliant or a blunder.",
        }
    if points_lost < 0.5 and rank == 1 and second_loss is not None and second_loss >= 3:
        return {
            "key": "brilliant",
            "en": "brilliant",
            "cn": "妙手",
            "tw": "妙手",
            "reason": (
                "It is KataGo's best move, and the next alternative gives up "
                f"{second_loss:.2f} points, so the choice is uniquely good."
            ),
        }
    bands = (
        (0.5, "proper", "proper", "本手", "本手", "Point loss is under 0.5, so the move matches KataGo's evaluation."),
        (1.5, "slow", "slow", "缓手", "緩手", "Point loss is from 0.5 up to 1.5: a small slip, not a blunder."),
        (
            3,
            "crude",
            "crude",
            "俗手",
            "俗手",
            "Point loss is from 1.5 up to 3: clearly worse than KataGo's choice, without deciding the game alone.",
        ),
        (6, "bad", "bad", "恶手", "惡手", "Point loss is from 3 up to 6."),
        (12, "blunder", "blunder", "大恶手", "大惡手", "Point loss is from 6 up to 12."),
    )
    for limit, key, en, cn, tw, reason in bands:
        if points_lost < limit:
            return {"key": key, "en": en, "cn": cn, "tw": tw, "reason": reason}
    return {
        "key": "collapse",
        "en": "collapse",
        "cn": "败着",
        "tw": "敗著",
        "reason": "Point loss is 12 or more.",
    }


def _best_other(candidates, played):
    for cand in candidates:
        if cand.get("move") != played:
            return cand
    return None


def _next_moves(node):
    candidates = list(getattr(node, "candidate_moves", None) or [])
    chosen = candidates[:TOP_CANDIDATES]
    return [_candidate_view(cand, index, candidates) for index, cand in enumerate(chosen, start=1)]


def grade_word(grade, language_code):
    if not grade:
        return "unrated"
    if language_code == "cn":
        return grade["cn"]
    if language_code == "tw":
        return grade["tw"]
    return grade["en"]


def _field_meanings():
    return "\n".join(
        [
            "Field meanings:",
            "- scoreLead is Black's lead. Positive means Black is ahead.",
            "- winrate is Black's win rate.",
            "- pointsLost is how many points the player to move gives up. 0 keeps the evaluation; larger is worse.",
            "- pv is KataGo's principal variation, starting with the candidate.",
            "- rank 1 is KataGo's best point, then the second, the third, and so on.",
            "These numbers are private context. Do not repeat any of them in the commentary.",
        ]
    )


def _candidate_section(snapshot):
    if snapshot["mode"] == "played":
        title = "Choices KataGo considered before this stone was played:"
    else:
        title = "KataGo's candidate points for the next move:"
    lines = [title]
    for cand in snapshot["candidates"]:
        lines.append(_format_candidate(cand, snapshot.get("played")))
    return "\n".join(lines)


def _next_section(snapshot):
    player = snapshot.get("to_move") or "?"
    lines = [f"KataGo's recommended next move, for {player} to play now:"]
    if not snapshot.get("next_moves"):
        lines.append("KataGo has not returned a next-move recommendation yet. Do not guess one.")
        return "\n".join(lines)
    for cand in snapshot["next_moves"]:
        lines.append(_format_candidate(cand, recommended=cand.get("rank") == 1))
    return "\n".join(lines)


def _select_candidates(candidates, played_gtp):
    chosen = list(candidates[:TOP_CANDIDATES])
    if played_gtp and not any(cand.get("move") == played_gtp for cand in chosen):
        match = next((cand for cand in candidates if cand.get("move") == played_gtp), None)
        if match is not None:
            chosen.append(match)
    return chosen


def _rank_of(candidates, gtp):
    for index, cand in enumerate(candidates, start=1):
        if cand.get("move") == gtp:
            return index
    return None


def _candidate_view(cand, list_index, full_list):
    pv = list(cand.get("pv") or [])[:PV_PLIES]
    return {
        "list_index": list_index,
        "rank": _rank_of(full_list, cand.get("move")) or list_index,
        "move": cand.get("move"),
        "visits": cand.get("visits"),
        "winrate": cand.get("winrate"),
        "scoreLead": cand.get("scoreLead"),
        "pointsLost": cand.get("pointsLost"),
        "prior": cand.get("prior"),
        "pv": pv,
    }


def _eval_of(node):
    if not getattr(node, "analysis_exists", False):
        return None
    return {"scoreLead": getattr(node, "score", None), "winrate": getattr(node, "winrate", None)}


def _ownership_sum(node):
    ownership = getattr(node, "ownership", None)
    if not ownership:
        return None
    try:
        return float(sum(ownership))
    except TypeError:
        return None


def _recent_moves(node, limit=8):
    moves = []
    current = node
    while current is not None and getattr(current, "move", None) is not None and len(moves) < limit:
        moves.append(f"{current.move.player} {current.move.gtp()}")
        current = getattr(current, "parent", None)
    moves.reverse()
    return moves


def _game_header(snapshot):
    size = snapshot.get("board_size") or ("?", "?")
    try:
        size_text = f"{size[0]}x{size[1]}"
    except (TypeError, IndexError):
        size_text = str(size)
    komi = snapshot.get("komi")
    komi_text = f"{komi:.1f}" if isinstance(komi, (int, float)) else "n/a"
    return f"Game: {size_text}, rules={snapshot.get('rules')}, komi={komi_text}"


def _decision_header(snapshot, language_code):
    lines = []
    if snapshot["mode"] == "played":
        player = snapshot.get("played_player") or "?"
        coord = snapshot.get("played") or "?"
        number = snapshot.get("move_number")
        number_text = f"move {number}" if number is not None else "the latest move"
        rank = snapshot.get("played_rank")
        rank_text = f"KataGo rank {rank}" if rank else "not in KataGo's listed candidates"
        lines.append(f"Move just played: {number_text}, {player} {coord} ({rank_text}).")
        grade = snapshot.get("grade") or {}
        word = grade_word(grade, language_code)
        lines.append(
            f"Objective grade required for the played move: {word}. {grade.get('reason', '')} "
            "Do not use a kinder or harsher grade."
        )
        other = snapshot.get("best_other")
        if other and snapshot.get("played_rank") != 1:
            lines.append(
                "KataGo's best point before this stone was "
                f"{other.get('move')}, pointsLost {_signed(other.get('pointsLost'))}."
            )
        elif other and snapshot.get("played_rank") == 1:
            lines.append(
                "The next alternative before this stone was "
                f"{other.get('move')}, pointsLost {_signed(other.get('pointsLost'))}."
            )
    else:
        lines.append(f"No new stone. The player to move is {snapshot.get('to_move') or '?'}.")
        lines.append("These are KataGo's candidate points for the next move.")
    if snapshot["mode"] == "played":
        lines.extend(_eval_lines("Before the move", snapshot.get("before")))
        lines.extend(_eval_lines("After the move", snapshot.get("after")))
        lost = snapshot.get("points_lost")
        if lost is not None:
            lines.append(f"Points given up by the played move: {lost:.2f}")
    else:
        lines.extend(_eval_lines("Position", snapshot.get("before")))
    ownership = snapshot.get("ownership_sum")
    if ownership is not None:
        lines.append(f"Sum of KataGo ownership (positive means Black): {ownership:.1f}")
    return "\n".join(lines)


def _eval_lines(label, evaluation):
    if not evaluation:
        return [f"{label}: KataGo evaluation missing."]
    score = evaluation.get("scoreLead")
    winrate = evaluation.get("winrate")
    score_text = "n/a" if score is None else f"{score:+.2f}"
    win_text = "n/a" if winrate is None else f"{winrate:.1%}"
    return [f"{label}: Black score lead {score_text}, Black win rate {win_text}."]


def _format_candidate(cand, played=None, recommended=False):
    marker = ""
    if played and cand.get("move") == played:
        marker = " [this is the move that was played]"
    elif recommended:
        marker = " [this is the next move KataGo recommends]"
    pv = " ".join(cand.get("pv") or []) or "n/a"
    return "\n".join(
        [
            f"rank {cand['rank']}: {cand.get('move')}{marker}",
            f"  visits: {_num(cand.get('visits'), 0)}",
            f"  prior: {_pct(cand.get('prior'))}",
            f"  winrate: {_pct(cand.get('winrate'))}",
            f"  scoreLead: {_signed(cand.get('scoreLead'))}",
            f"  pointsLost: {_signed(cand.get('pointsLost'))}",
            f"  pv: {pv}",
        ]
    )


def _task_instructions(mode, language_code):
    if language_code == "cn":
        if mode == "played":
            return (
                "这段解说是讲给学棋的人听的，目的是帮助棋手看懂思路、提高下棋的能力，不是交一份分析报告。"
                "请只用上面的 KataGo 分析来判断对错，不要写出目差、胜率、损失目数、访问次数这些计算数值。\n"
                "1. 讲解刚刚下出来的这一手的意义、目的和作用：这里为什么要这样想，走到了什么，又让出了什么，好处和坏处是什么，以后遇到类似局面可以学到什么。"
                "先使用上面规定的客观等级，不要把等级说得更好或更差。如果它不是当时的最佳选点，用棋理说明它和最佳选点的思路差在哪里。\n"
                "2. 讲解 KataGo 推荐的下一手，以及后面的第二、第三选点：每个点的意义、目的和作用是什么，下在那里是顺着哪一种思路，棋手应该从中学到什么。\n"
                "不要编造列表之外的变化。如果写明还没有下一手推荐，就不要猜下一手。每部分三到四句话，不要逐手复述整条主变化。"
            )
        return (
            "当前没有新的一手棋要评价。请像老师一样，只用上面的 KataGo 候选点讲解思路，帮助棋手看懂并提高，不要写出目差、胜率、损失目数、访问次数。\n"
            "1. 推荐的下一手想达到什么，这个思路是什么。\n"
            "2. 第二选点的意义和作用，它代表哪一种想法。\n"
            "3. 第三选点的意义和作用，它代表哪一种想法。\n"
            "不要编造列表之外的变化。每一点用两三句话。"
        )
    if language_code == "tw":
        if mode == "played":
            return (
                "這段解說是講給學棋的人聽的，目的是幫助棋手看懂思路、提高下棋的能力，不是交一份分析報告。"
                "請只用上面的 KataGo 分析來判斷對錯，不要寫出目差、勝率、損失目數、訪問次數這些計算數值。\n"
                "1. 講解剛剛下出來的這一手的意義、目的和作用：這裡為什麼要這樣想，走到了什麼，又讓出了什麼，好處和壞處是什麼，以後遇到類似局面可以學到什麼。"
                "先使用上面規定的客觀等級，不要把等級說得更好或更差。如果它不是當時的最佳選點，用棋理說明它和最佳選點的思路差在哪裡。\n"
                "2. 講解 KataGo 推薦的下一手，以及後面的第二、第三選點：每個點的意義、目的和作用是什麼，下在那裡是順著哪一種思路，棋手應該從中學到什麼。\n"
                "不要編造列表之外的變化。如果寫明還沒有下一手推薦，就不要猜下一手。每部分三到四句話，不要逐手複述整條主要變化。"
            )
        return (
            "目前沒有新的一手棋要評價。請像老師一樣，只用上面的 KataGo 候選點講解思路，幫助棋手看懂並提高，不要寫出目差、勝率、損失目數、訪問次數。\n"
            "1. 推薦的下一手想達到什麼，這個思路是什麼。\n"
            "2. 第二選點的意義和作用，它代表哪一種想法。\n"
            "3. 第三選點的意義和作用，它代表哪一種想法。\n"
            "不要編造列表之外的變化。每一點用兩三句話。"
        )
    if mode == "played":
        return (
            "This commentary is a lesson for a human player: help them understand the thinking and become stronger. "
            "It is not an analysis report. Using only the KataGo analysis above, decide what is true. "
            "Do not write calculation numbers such as win rate, point loss, score lead, or visit count.\n"
            "1. Teach the meaning, purpose, and effect of the move just played: why a player would think this way, "
            "what it gains, what it gives up, and what to learn for a similar position. "
            "Use the required grade, and do not make it kinder or harsher. "
            "If it was not the best point, compare the thinking with that point.\n"
            "2. Teach KataGo's recommended next point and the second and third points: "
            "the meaning, purpose, and effect of each, which idea it follows, and what the player should learn.\n"
            "Do not invent variations that are not listed. If no next move is recommended yet, do not guess one. "
            "Use three or four sentences for each part, and do not narrate every move of a variation."
        )
    return (
        "No new stone needs a grade. Teach the human from the KataGo candidates above: "
        "the meaning, purpose, and effect of each point, and the idea a player should learn. "
        "Do not write calculation numbers such as win rate, point loss, score lead, or visit count.\n"
        "1. What the recommended next move is trying to achieve, and the thinking behind it.\n"
        "2. The meaning and effect of the second point, and which idea it represents.\n"
        "3. The meaning and effect of the third point, and which idea it represents.\n"
        "Do not invent variations. Use two or three sentences for each point."
    )


def _num(value, digits):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _signed(value):
    if value is None:
        return "n/a"
    return f"{value:+.2f}"


def _pct(value):
    if value is None:
        return "n/a"
    return f"{value:.1%}"
