"""The desk: a side question answered by a fork of the main request, and the solver
settled before the main agent's first one.

The side chat's whole economy is the prompt cache: a side question is worth asking only
because it reads the main conversation at the cache-read rate, and it does only while
its request is the main agent's last request byte for byte, with one message appended.
So most of what is pinned here is that shape -- the same system prompt, the same tool
list, the same settings, the same messages -- and the other half of the contract: that
nothing of it comes back into the main thread except the person's words, when they ask.
"""

from __future__ import annotations

import pytest

from conftest import ScriptedReader, install_model, message, text_block, tool_block
from openreynolds import cli, desk, solvers
from openreynolds.config import Config
from openreynolds.desk import NOT_ENABLED, SIDE_NOTE, SideChat
from openreynolds.llm import ProviderError
from openreynolds.loop import Loop, _as_operator_text


def make_loop(ctx, store, view, responses):
    loop = Loop(Config(llm_api_key="k", model="claude-opus-5-5"), ctx, store, view)
    fake = install_model(loop, responses)
    return loop, fake


def side_for(loop, view, reader=None):
    reader = reader if reader is not None else ScriptedReader([])
    return SideChat(loop, view, deliver=lambda words, note: cli._deliver_forward(
        loop, reader, words, note))


def asked_mid_turn(ctx, side, question, cmd="ls"):
    """Ask `question` while the main agent's `bash` call is running -- the moment a
    side question is for."""
    real = ctx.backend.exec
    seen = []

    def exec_and_ask(c, cwd=None, timeout_s=120, *, background=False):
        if c == cmd and not seen:
            seen.append(c)
            side.ask(question, wait=True)
        return real(c, cwd, timeout_s, background=background)

    ctx.backend.exec = exec_and_ask
    return seen


def _texts(content):
    if isinstance(content, str):
        return [content]
    return [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]


def _all_text(messages):
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for b in content:
                if isinstance(b, dict):
                    out.append(str(b.get("text") or b.get("content") or ""))
                else:
                    out.append(str(getattr(b, "text", "") or getattr(b, "input", "")))
    return "\n".join(out)


# -- the fork ------------------------------------------------------------------


def test_a_side_request_is_the_last_main_request_with_one_message_appended(ctx, store, view):
    """Byte for byte: model, system, tools, thinking, effort, output config, cache
    control, every message -- and then one user message. Anything else misses the
    cache, and a different system prompt or tool list invalidates the thinking blocks."""
    loop, fake = make_loop(ctx, store, view, [
        message([tool_block("bash", {"cmd": "ls"})], stop_reason="tool_use"),
        message([text_block("It is listing the case directory.")]),
        message([text_block("done")]),
    ])
    loop.say("mesh the duct")
    side = side_for(loop, view)
    asked_mid_turn(ctx, side, "what is it doing right now?")

    loop.run()

    main, aside = fake.calls[0], fake.calls[1]
    assert set(aside) == set(main)
    for key in main:
        if key != "messages":
            assert aside[key] == main[key], f"{key} differs from the main request"
    assert aside["tools"] == main["tools"], "the main tool list, all of it"
    assert aside["messages"][: len(main["messages"])] == main["messages"]
    (extra,) = aside["messages"][len(main["messages"]):]
    assert extra["role"] == "user"
    assert extra["content"][0]["text"] == _as_operator_text(SIDE_NOTE)
    assert extra["content"][1]["text"] == "what is it doing right now?"
    assert view.side_questions == ["what is it doing right now?"]
    assert view.side_answers == [("It is listing the case directory.", False)]


def test_the_harness_note_names_what_is_enabled_and_the_system_prompt_is_untouched():
    for name in ("read_file", "job_check", "send_to_agent"):
        assert name in SIDE_NOTE
    assert "not run" in SIDE_NOTE
    from openreynolds.prompt import FELIX_PROMPT, SYSTEM_PROMPT

    assert "side question" not in SYSTEM_PROMPT and "side question" not in FELIX_PROMPT


def test_a_side_question_and_its_answer_never_enter_the_main_thread(ctx, store, view):
    loop, fake = make_loop(ctx, store, view, [
        message([tool_block("bash", {"cmd": "ls"})], stop_reason="tool_use"),
        message([text_block("SIDE-ANSWER-7")]),
        message([text_block("done")]),
    ])
    loop.say("mesh the duct")
    side = side_for(loop, view)
    asked_mid_turn(ctx, side, "SIDE-QUESTION-7")

    loop.run()

    after = fake.calls[2]["messages"]
    assert "SIDE-QUESTION-7" not in _all_text(after)
    assert "SIDE-ANSWER-7" not in _all_text(after)
    assert "SIDE-QUESTION-7" not in _all_text(loop.messages)
    assert "SIDE-ANSWER-7" not in _all_text(loop.messages)
    rows = "\n".join(str(r.get("content")) for r in store.recent_messages(50))
    assert "SIDE-QUESTION-7" not in rows and "SIDE-ANSWER-7" not in rows
    assert view.interjections == []


def test_a_follow_up_builds_on_the_side_thread_and_keeps_the_prefix(ctx, store, view):
    """A second question on the same fork sends the first exchange as it was -- the
    cache prefix grows, it does not move -- and the new question after it."""
    loop, fake = make_loop(ctx, store, view, [
        message([text_block("finished")]),
        message([text_block("first answer")]),
        message([text_block("second answer")]),
    ])
    loop.say("go")
    loop.run()
    side = side_for(loop, view)

    side.ask("first?", wait=True)
    side.ask("and second?", wait=True)

    first, second = fake.calls[1]["messages"], fake.calls[2]["messages"]
    assert second[: len(first)] == first, "the first side request is the second's prefix"
    assert _all_text([second[len(first)]]) == "first answer"
    assert _texts(second[-1]["content"])[-1] == "and second?"
    assert [a for a, _ in view.side_answers] == ["first answer", "second answer"]


def test_once_the_turn_has_ended_its_last_words_ride_after_the_request(ctx, store, view):
    """Idle, the reply to the last request is complete -- "the drag is 0.31" -- and is
    what a side question is most often about, so it follows the request (same prefix,
    same prompt and tools; only its own tokens are new)."""
    loop, fake = make_loop(ctx, store, view, [
        message([text_block("The drag coefficient is 0.31.")]),
        message([text_block("0.31")]),
    ])
    loop.say("what is the drag?")
    loop.run()

    side_for(loop, view).ask("remind me of the number?", wait=True)

    main, aside = fake.calls[0]["messages"], fake.calls[1]["messages"]
    assert aside[: len(main)] == main
    assert aside[len(main)]["role"] == "assistant"
    assert "0.31" in _all_text([aside[len(main)]])
    assert len(aside) == len(main) + 2


def test_a_moved_fork_carries_earlier_side_answers_as_text(ctx, store, view):
    """Thinking made on one prefix is bound to it, so after the main thread has sent
    another request the side thread starts again -- with the earlier exchange in words."""
    loop, fake = make_loop(ctx, store, view, [
        message([text_block("one")]),
        message([text_block("side one")]),
        message([text_block("two")]),
        message([text_block("side two")]),
    ])
    loop.say("go")
    loop.run()
    side = side_for(loop, view)
    side.ask("q1", wait=True)
    loop.say("again")
    loop.run()

    side.ask("q2", wait=True)

    latest = fake.calls[3]["messages"]
    note = _texts(latest[-1]["content"])[0]
    assert "Q: q1" in note and "A: side one" in note
    assert not any(m["role"] == "assistant" and "side one" in _all_text([m]) for m in latest)


# -- the tools -----------------------------------------------------------------


def test_a_tool_not_enabled_in_a_side_question_is_not_run(ctx, store, view):
    loop, fake = make_loop(ctx, store, view, [
        message([text_block("idle")]),
        message([tool_block("bash", {"cmd": "rm -rf /work/case"}, "side_1")],
                stop_reason="tool_use"),
        message([text_block("I cannot run commands from here.")]),
    ])
    loop.say("go")
    loop.run()
    before = list(ctx.backend.execs)

    side_for(loop, view).ask("delete the case", wait=True)

    assert ctx.backend.execs == before, "nothing ran"
    result = fake.calls[2]["messages"][-1]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "side_1"
    assert result["content"] == NOT_ENABLED.format(name="bash")
    assert view.side_answers == [("I cannot run commands from here.", False)]


def test_an_enabled_read_only_tool_runs_and_leaves_the_main_bookkeeping_alone(ctx, store, view):
    ctx.backend.files["/work/case/log"] = b"Time = 0.4\n" * 300  # past the echo threshold
    loop, fake = make_loop(ctx, store, view, [
        message([text_block("idle")]),
        message([tool_block("read_file", {"path": "/work/case/log"}, "side_1")],
                stop_reason="tool_use"),
        message([text_block("It is at t = 0.4.")]),
    ])
    loop.say("go")
    loop.run()
    echoes, calls = dict(ctx.echoes), ctx.calls

    side_for(loop, view).ask("how far is it?", wait=True)

    result = fake.calls[2]["messages"][-1]["content"][0]
    assert "Time = 0.4" in str(result["content"])
    assert ctx.echoes == echoes and ctx.calls == calls, (
        "a side read is not in the main agent's record of what it has been shown")


def test_a_side_job_check_reads_the_log_without_moving_the_offset_or_waiting(ctx, store, view):
    job = ctx.backend.job_start("simpleFoam", name="solve")
    ctx.backend.logs[job] = b"Time = 1\nTime = 2\n"
    store.record_job(job, cmd="simpleFoam", name="solve")
    store.update_job(job, log_offset=9)
    loop, fake = make_loop(ctx, store, view, [
        message([text_block("idle")]),
        message([tool_block("job_check", {"job_id": job, "wait_s": 300}, "side_1")],
                stop_reason="tool_use"),
        message([text_block("At Time = 2.")]),
    ])
    loop.say("go")
    loop.run()

    side_for(loop, view).ask("where is the solve?", wait=True)

    result = str(fake.calls[2]["messages"][-1]["content"][0]["content"])
    assert "Time = 2" in result and "wait_s is not held" in result
    assert store.session.jobs[job].log_offset == 9, "the main agent's bookmark is untouched"


def test_a_side_question_is_bounded(ctx, store, view):
    loop, fake = make_loop(ctx, store, view, [
        message([text_block("idle")]),
        message([tool_block("read_file", {"path": "/work"}, "side_1")], stop_reason="tool_use"),
    ])
    ctx.backend.dirs["/work"] = ["case"]
    loop.say("go")
    loop.run()

    side_for(loop, view).ask("look around", wait=True)

    assert len(fake.calls) == 1 + desk.SIDE_STEPS
    assert view.side_answers and view.side_answers[0][1] is False


def test_the_main_agent_calling_send_to_agent_sends_nothing(ctx, store, view):
    loop, fake = make_loop(ctx, store, view, [
        message([tool_block("send_to_agent", {"note": "x"})], stop_reason="tool_use"),
        message([text_block("ok")]),
    ])
    loop.say("go")
    loop.run()

    result = fake.calls[1]["messages"][-1]["content"][0]
    assert "only for answering a side question" in str(result["content"])
    assert loop.take_forwarded() == []


# -- passing it on ---------------------------------------------------------------


def test_send_to_agent_delivers_the_persons_own_words_at_the_next_step(ctx, store, view):
    question = "Can you also export the wall shear stress?  (only on the walls)"
    loop, fake = make_loop(ctx, store, view, [
        message([tool_block("bash", {"cmd": "ls"})], stop_reason="tool_use"),
        message([tool_block("send_to_agent", {"note": "they asked mid-solve"}, "side_1")],
                stop_reason="tool_use"),
        message([text_block("Passed on.")]),
        message([text_block("done")]),
    ])
    loop.say("mesh the duct")
    side = side_for(loop, view)
    asked_mid_turn(ctx, side, question)

    loop.run()

    texts = _texts(fake.calls[3]["messages"][-1]["content"])
    assert question in texts, "the person's words, exactly as typed"
    note = next(t for t in texts if "side question" in t)
    assert note.startswith("[from the harness, not the user]")
    assert "The side chat's own note, not the person's: they asked mid-solve" in note
    assert texts.index(question) < texts.index(note)
    assert view.side_answers == [("Passed on.", True)]
    users = [r["content"] for r in store.recent_messages(50) if r["role"] == "user"]
    assert question in users, "recorded as the person's"


def test_send_to_agent_with_no_turn_running_waits_at_the_prompt_as_their_line(ctx, store, view):
    loop, fake = make_loop(ctx, store, view, [
        message([text_block("finished")]),
        message([tool_block("send_to_agent", {}, "side_1")], stop_reason="tool_use"),
        message([text_block("Passed on.")]),
    ])
    loop.say("go")
    loop.run()
    reader = ScriptedReader([])

    side_for(loop, view, reader).ask("please rerun it finer", wait=True)

    assert reader.get() == "please rerun it finer"
    assert any("side question" in p for p in loop._posted)


def test_send_to_agent_passes_it_on_once_per_question(ctx, store, view):
    loop, fake = make_loop(ctx, store, view, [
        message([text_block("finished")]),
        message([tool_block("send_to_agent", {}, "a"), tool_block("send_to_agent", {}, "b")],
                stop_reason="tool_use"),
        message([text_block("Passed on.")]),
    ])
    loop.say("go")
    loop.run()
    reader = ScriptedReader([])

    side_for(loop, view, reader).ask("do it", wait=True)

    assert reader.get() == "do it" and reader.get() is None


def test_a_forward_ends_a_held_wait_like_a_typed_message(ctx, store, view, monkeypatch):
    import time as _time

    from openreynolds import tools as tools_mod

    monkeypatch.setattr(tools_mod, "JOB_WAIT_POLL_S", 0.01)
    loop, _ = make_loop(ctx, store, view, [message([text_block("x")])])
    ctx.on_wait_input = loop.heard
    job = ctx.backend.job_start("simpleFoam", name="solve")
    store.record_job(job, cmd="simpleFoam", name="solve")
    loop.running = True
    assert loop.forward("is the inlet right?", "")

    began = _time.monotonic()
    out, _ = tools_mod.dispatch(ctx, "job_check", {"job_id": job, "wait_s": 30})

    assert _time.monotonic() - began < 5
    assert "the person wrote, so this answered early" in out
    assert loop._typed == ["is the inlet right?"]


def test_a_held_wait_asks_after_the_person_every_second_and_the_job_every_five(ctx, monkeypatch):
    """A message typed while the agent waits is read within a second; a status is a
    round trip to the workspace and is asked as often as before."""
    from openreynolds import tools as tools_mod

    clock = {"t": 0.0}
    monkeypatch.setattr(tools_mod.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(tools_mod.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    job = ctx.backend.job_start("simpleFoam", name="solve")
    status = ctx.backend.job_status(job)
    asked, looked = [], []
    real = ctx.backend.job_status
    ctx.backend.job_status = lambda jid: looked.append(clock["t"]) or real(jid)
    ctx.on_wait_input = lambda: asked.append(clock["t"]) or clock["t"] >= 7.0

    tools_mod._hold(ctx, job, status, 30)

    assert asked == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert looked == [5.0]


# -- routing ---------------------------------------------------------------------


class _Routing:
    def __init__(self, turn=None, error=None):
        self.turn, self.error, self.calls = turn, error, []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.turn


def _route_turn(solver, reason="it is incompressible and large"):
    from openreynolds.llm import Turn
    from openreynolds.llm.base import ToolUseBlock

    return Turn(content=[ToolUseBlock(id="r", name="choose_solver",
                                      input={"solver": solver, "reason": reason})],
                stop_reason="tool_use", tokens={"input": 50})


def test_routing_asks_once_with_its_own_prompt_and_one_tool():
    provider = _Routing(_route_turn("felix"))

    chosen = desk.route(provider, "claude-opus-5-5", "external aero of a car, 40M cells")

    (call,) = provider.calls
    assert call["system"] == desk.ROUTING_SYSTEM
    assert [t["name"] for t in call["tools"]] == ["choose_solver"]
    assert call["messages"] == [{"role": "user", "content": "external aero of a car, 40M cells"}]
    assert call["model"] == "claude-opus-5-5", "the study's own model"
    assert (chosen.solver, chosen.reason, chosen.failed) == (
        "felix", "it is incompressible and large", "")


@pytest.mark.parametrize("provider", [
    _Routing(error=ProviderError("boom", 500)),
    _Routing(_route_turn("fluent")),
])
def test_routing_that_does_not_choose_is_openfoam_and_says_why(provider):
    chosen = desk.route(provider, "m", "a pipe")
    assert chosen.solver == solvers.OPENFOAM and chosen.failed
    assert "OpenFOAM" in desk.routed_notice(chosen) and chosen.failed in desk.routed_notice(chosen)
    assert "by default" in desk.routed_line(chosen)


def test_routing_with_nothing_to_read_does_not_guess():
    provider = _Routing(_route_turn("felix"))
    chosen = desk.route(provider, "m", "   ")
    assert chosen.solver == solvers.OPENFOAM and chosen.failed and provider.calls == []
