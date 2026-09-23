"""TypeSafe makes choices; an optional small OpenAI-compatible model writes field values."""

import json
import math
import os
import time

import httpx

from .questions import NEXT_ACTION, TARGET, TEXT_VALUE

CLIENT = httpx.Client(http2=True, timeout=25)


def system_one_url():
    """Where decisions are sent. Any server implementing /v1/systemone answers this request, so
    pointing it at a local one is a base URL, not a fork."""
    return os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/") + "/v1/systemone"


def describe_target(index, action):
    """One option description, as text. The spec takes a description as a string, and a server that
    holds to it rejects anything else, so the role and current value go in the sentence."""
    parts = [f"Element [{index}], labelled {action['label']!r}"]
    if action.get("role"):
        parts.append(f"a {action['role']}")
    value = action.get("current_value", action.get("value", ""))
    if value:
        parts.append(f"currently holding {value!r}")
    for flag in ("checked", "selected", "expanded"):
        if flag in action:
            parts.append(f"{flag}: {action[flag]}")
    return ", ".join(parts) + "."


def flatten_instructions(goal, rules, operation=None):
    """Instructions as one string, for the same reason."""
    head = [f"Goal: {goal}"]
    if operation:
        head.append(f"Operation under consideration: {operation}")
    return "\n\n".join(head + ([rules] if isinstance(rules, str) else list(rules)))


def post_json(url, key, body):
    for attempt in range(3):
        try:
            response = CLIENT.post(url, json=body, headers={"Authorization": f"Bearer {key}"})
        except httpx.HTTPError:
            raise RuntimeError("Model connection failed; no action executed.") from None
        if response.status_code in {429, 529, 503} and attempt < 2:
            time.sleep(0.5 * 2**attempt)
            continue
        if response.is_error:
            raise RuntimeError(f"Model provider returned HTTP {response.status_code}; no action executed.")
        return response.json()
    raise RuntimeError("Model unavailable")


def validate_choice(answer, ids):
    try:
        probabilities = answer["probabilities"]
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Invalid TypeSafe response; no action executed.")
    return answer


def action_space(actions):
    """One index per observed element; each operation has its own valid target choices."""
    elements, indices, targets, controls = [], {}, {}, {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
    for action in actions:
        kind = action["kind"]
        if kind not in operations:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in ("role", "value", "checked", "selected", "expanded") if k in action}
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = operations[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action["label"], "value": action["value"]})
        group[target] = action
    return elements, targets, controls


def plan_questions(pending, operations, targets, settled):
    """One satisfaction check per outstanding sub-goal.

    These ride in the request that was being sent anyway. A whole-task DONE is one broad judgement
    and answers weakly when the page could arguably be said to satisfy the goal already; a sub-goal
    asks something narrow enough to answer sharply.
    """
    questions = {
        f"plan{offset}_satisfied": {
            "type": "noul",
            "instructions": f"Is this step already satisfied on the page as it stands?\n\nStep: {text}",
            "criteria": {
                "true": "The page already shows this step's outcome.",
                "false": "This step still needs an action, or the page does not show its outcome.",
            },
        }
        for offset, text in enumerate(pending)
    }
    # A full decision per outstanding sub-goal. Asking costs almost nothing next to a round trip,
    # and an answer that is still applicable when its turn comes replaces the call that turn needs.
    for offset, text in enumerate(pending):
        questions[f"plan{offset}_operation"] = {
            "type": "choice", "criteria": operations, "instructions": flatten_instructions(text, NEXT_ACTION),
        }
        for operation, candidates in targets.items():
            if operation in settled:
                continue
            questions[f"plan{offset}_{operation.lower()}_target"] = {
                "type": "choice",
                "criteria": {index: describe_target(index, a) for index, a in candidates.items()},
                "instructions": flatten_instructions(text, [NEXT_ACTION, TARGET], operation),
            }
    return questions


def read_plan_answers(answers, pending, operations, targets, settled):
    """Probability that each outstanding sub-goal is already satisfied, or None if unreadable.
    A malformed reading must not stop the run, so it is dropped rather than raised."""
    read = []
    for offset in range(len(pending)):
        value = answers.get(f"plan{offset}_satisfied", {}).get("noul")
        satisfied = value if isinstance(value, (int, float)) and 0 <= value <= 1 else None
        entry = {"satisfied": satisfied, "operation": None, "label": None, "kind": None, "confidence": None}
        try:
            answer = validate_choice(answers.get(f"plan{offset}_operation", {}), operations)
            entry["operation"], entry["confidence"] = answer["choice"], answer["confidence"]
            group = targets.get(entry["operation"])
            if group is not None:
                if entry["operation"] in settled:
                    action = group[settled[entry["operation"]]]
                else:
                    target = validate_choice(
                        answers.get(f"plan{offset}_{entry['operation'].lower()}_target", {}), group
                    )
                    action = group[target["choice"]]
                # Held by label and kind, never by node id: an id survives a change of meaning.
                entry["label"], entry["kind"] = action["label"], action["kind"]
        except (ValueError, KeyError):
            entry["label"] = None
        read.append(entry)
    return read


def choose(state, goal, history, pending=(), suppress=()):
    elements, targets, controls = action_space(state["actions"])
    # A control that has been chosen repeatedly without moving the page is not going to move it
    # this time either. Withholding it costs nothing and forces the next-best answer.
    if suppress:
        targets = {
            operation: kept
            for operation, candidates in targets.items()
            if (kept := {i: a for i, a in candidates.items() if (a["label"], a["kind"]) not in suppress})
        }
        if not targets:  # never strand the agent: an empty action space can only answer BLOCKED
            _, targets, _ = action_space(state["actions"])
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")
    questions = {
        "operation": {"type": "choice", "criteria": operations, "instructions": flatten_instructions(goal, NEXT_ACTION)}
    }
    # A head offering one candidate is not a choice: it answers 1.00 whatever the element is, which
    # reads as certainty to anything downstream weighing confidence.
    settled = {operation: next(iter(candidates)) for operation, candidates in targets.items() if len(candidates) == 1}
    for operation, candidates in targets.items():
        if operation in settled:
            continue
        questions[operation.lower() + "_target"] = {
            "type": "choice",
            "criteria": {index: describe_target(index, a) for index, a in candidates.items()},
            "instructions": flatten_instructions(goal, [NEXT_ACTION, TARGET], operation),
        }
    questions.update(plan_questions(pending, operations, targets, settled))
    body = {
        "model": os.environ.get("TYPESAFE_MODEL", "jev-latest"),
        "state": {
            "page": {k: state[k] for k in ("url", "title", "text")},
            "elements": elements,
            "recent_actions": [
                {k: h.get(k) for k in ("action", "kind", "text", "page_changed")} for h in history[-10:]
            ],
        },
        "questions": questions,
    }
    started = time.perf_counter()
    result = post_json(system_one_url(), os.environ["TYPESAFE_API_KEY"], body)
    operation_answer = validate_choice(result["answers"].get("operation", {}), operations)
    operation = operation_answer["choice"]
    target = None
    target_answer = None
    probabilities = {}
    if operation in settled:
        target = settled[operation]
        choice = targets[operation][target]["id"]
        probabilities = {choice: operation_answer["probabilities"][operation]}
    elif operation in targets:
        # Unused target heads cannot cause an action. Validate the head selected by the operation.
        target_answer = validate_choice(result["answers"].get(operation.lower() + "_target", {}), targets[operation])
        target = target_answer["choice"]
        choice = targets[operation][target]["id"]
        probabilities = {a["id"]: target_answer["probabilities"][index] for index, a in targets[operation].items()}
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"][operation]
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "plan": read_plan_answers(result["answers"], pending, operations, targets, settled),
        "raw_answers": result["answers"],
        "model": result["model"],
        "usage": result.get("usage", {}),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": body,
    }


def field_context(goal, action, page, history):
    return {
        "goal": goal,
        "field": {k: action.get(k) for k in ("label", "role", "value")},
        "page": {"title": page["title"], "text": page["text"][:6000]},
        "recent_actions": [{k: h.get(k) for k in ("action", "text")} for h in history[-6:]],
    }


def field_value(content):
    """The JSON object a text helper meant to send, out of what it actually sent.

    Asking for a JSON object does not guarantee one arrives alone: a model may fence it, introduce
    it, or follow it with a remark. Recovering the object costs nothing and does not widen what
    counts as a valid answer - the value inside it is still checked as strictly as before.
    """
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def field_text(context):
    key = os.environ.get("TEXT_MODEL_API_KEY")
    if not key:
        raise ValueError("TYPE_TEXT needs TEXT_MODEL_API_KEY; no text is hardcoded or guessed by the executor.")
    base = os.environ.get("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.environ.get("TEXT_MODEL", "deepseek-chat")
    reasoning = {"thinking": {"type": "disabled"}} if "api.deepseek.com/" in base else {"reasoning": {"effort": "low"}}
    if os.environ.get("TEXT_MODEL_REASONING") == "none":
        reasoning = {"reasoning": {"enabled": False}}
    started = time.perf_counter()
    result = post_json(
        base + "/chat/completions",
        key,
        {
            "model": model,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
            **reasoning,
            "messages": [
                {"role": "system", "content": TEXT_VALUE},
                {
                    "role": "user",
                    "content": json.dumps(context),
                },
            ],
        },
    )
    content = (result.get("choices") or [{}])[0].get("message", {}).get("content")
    try:
        output = field_value(content)
        value = output["text"]
        if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        # Carry what came back: without it the next occurrence is as undiagnosable as the last.
        raise ValueError(
            f"Text helper returned no valid field value; nothing typed. Got: {str(content)[:120]!r}"
        ) from None
    return value, {
        "model": model,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": result.get("usage", {}),
    }
