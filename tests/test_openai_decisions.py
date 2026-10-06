import base64
import json

import httpx2
import llm
import pytest
from click.testing import CliRunner
from llm.cli import cli

from llm_openai_decisions import API_URL, DecisionsOptions

MODEL = "openai-decisions/gpt-6-luna"
OPTIONS = {
    "predicate": {},
    "choice": {"choices": {"billing": "Payments", "other": None}},
    "score": {"levels": ["None", {"label": "Severe", "description": "Blocked"}]},
}
ANSWERS = {
    "predicate": {"name": "evaluation", "type": "predicate", "probability": 0.8},
    "choice": {
        "name": "evaluation",
        "type": "choice",
        "choice": "billing",
        "confidence": 0.6,
        "probabilities": [
            {"value": "billing", "probability": 0.8},
            {"value": "other", "probability": 0.2},
        ],
    },
    "score": {
        "name": "evaluation",
        "type": "score",
        "score": 0.8,
        "confidence": 0.6,
        "probabilities": [
            {"value": 0, "label": "None", "probability": 0.2},
            {"value": 1, "label": "Severe", "probability": 0.8},
        ],
    },
}
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII="
)


def payload(kind="predicate", answers=None):
    return {
        "model": "gpt-6-luna",
        "answers": answers if answers is not None else [ANSWERS[kind]],
        "usage": {
            "input_tokens": 42,
            "output_tokens": 0,
            "total_tokens": 42,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def async_(request):
    return request.param


async def evaluate(async_, text="Input", **kwargs):
    model = llm.get_async_model(MODEL) if async_ else llm.get_model(MODEL)
    response = model.prompt(text, **kwargs)
    result = await response.text() if async_ else response.text()
    return response, json.loads(result)


def test_registration():
    for get_model in (llm.get_model, llm.get_async_model):
        model = get_model(MODEL)
        assert model.needs_key == "openai"
        assert not model.can_stream
        assert not model.supports_conversation
        assert not model.supports_schema
        assert not model.supports_tools
        with pytest.raises(llm.UnknownModelError):
            get_model("decisions")
    result = CliRunner().invoke(cli, ["models", "-m", MODEL, "--options"])
    assert result.exit_code == 0, result.output
    assert "aliases:" not in result.output
    assert "questions" in result.output


@pytest.mark.parametrize("kind", ANSWERS)
@pytest.mark.asyncio
async def test_evaluate(kind, async_, httpx2_mock):
    data = payload(kind)
    data["metadata"] = {"extra": "preserved"}
    httpx2_mock.add_response(url=API_URL, json=data)
    response, answer = await evaluate(
        async_,
        "Please refund my payment",
        system="Evaluate this message",
        answer_type=kind,
        **OPTIONS[kind],
    )
    assert answer == ANSWERS[kind]
    assert (await response.json() if async_ else response.json()) == data
    usage = await response.usage() if async_ else response.usage()
    assert (usage.input, usage.output) == (42, 0)
    assert usage.details == {
        k: data["usage"][k] for k in ("input_tokens_details", "output_tokens_details")
    }
    assert response.resolved_model == "gpt-6-luna"
    request = httpx2_mock.get_request()
    assert request.method == "POST"
    assert request.headers["Authorization"] == "Bearer test-openai-secret"
    body = json.loads(request.content)
    assert body["input"] == "Please refund my payment"
    assert body["model"] == "gpt-6-luna"
    question = body["questions"][0]
    assert question["name"] == "evaluation"
    assert question["instructions"] == "Evaluate this message"
    assert question["type"] == kind
    if kind == "choice":
        assert question["choices"] == [
            {"value": "billing", "description": "Payments"},
            {"value": "other"},
        ]
    elif kind == "score":
        assert question["levels"] == [
            {"label": "None"},
            {"label": "Severe", "description": "Blocked"},
        ]
    else:
        assert set(question) == {"type", "name", "instructions"}


@pytest.mark.asyncio
async def test_multiple_questions_and_refusal(async_, httpx2_mock):
    questions = [
        {"name": "refund", "type": "predicate", "instructions": "Refund?"},
        {
            "name": "severity",
            "type": "score",
            "instructions": "Severity?",
            "levels": [{"label": "Low"}, {"label": "High"}],
        },
    ]
    answers = [
        {"name": "refund", "type": "predicate", "probability": 0.8},
        {"name": "severity", "type": "refusal"},
    ]
    httpx2_mock.add_response(json=payload(answers=answers))
    _, result = await evaluate(async_, questions=json.dumps(questions))
    assert result == answers
    assert json.loads(httpx2_mock.get_request().content)["questions"] == questions


@pytest.mark.asyncio
async def test_single_refusal(async_, httpx2_mock):
    answer = {"type": "refusal", "name": "evaluation"}
    httpx2_mock.add_response(json=payload(answers=[answer]))
    _, result = await evaluate(async_, system="Question")
    assert result == answer


@pytest.mark.parametrize("source", ["content", "path", "url"])
@pytest.mark.asyncio
async def test_image_input(source, async_, httpx2_mock, tmp_path):
    if source == "path":
        path = tmp_path / "image.png"
        path.write_bytes(PNG)
        attachment = llm.Attachment(path=str(path))
    elif source == "url":
        httpx2_mock.add_response(
            url="https://example.com/image.png", method="GET", content=PNG
        )
        attachment = llm.Attachment(
            url="https://example.com/image.png", type="image/png"
        )
    else:
        attachment = llm.Attachment(content=PNG, type="image/png")
    httpx2_mock.add_response(url=API_URL, json=payload())
    await evaluate(async_, None, system="Visible damage?", attachments=[attachment])
    body = json.loads(httpx2_mock.get_request(url=API_URL).content)
    assert body["input"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,"
                    + base64.b64encode(PNG).decode(),
                }
            ],
        }
    ]


def test_fragments_and_explicit_messages(httpx2_mock):
    httpx2_mock.add_response(json=payload())
    llm.get_model(MODEL).prompt(
        messages=[
            llm.system("Refund?"),
            llm.user("Please refund"),
            llm.user("this payment"),
        ],
        fragments=["More evidence"],
    ).text()
    body = json.loads(httpx2_mock.get_request().content)
    assert "Please refund" in json.dumps(body["input"])
    assert "this payment" in json.dumps(body["input"])
    assert "More evidence" in json.dumps(body["input"])
    assert body["questions"][0]["instructions"] == "Refund?"


def test_system_fragments(httpx2_mock):
    httpx2_mock.add_response(json=payload())
    llm.get_model(MODEL).prompt(
        "Input", system="Refund?", system_fragments=["Apply these rules"]
    ).text()
    body = json.loads(httpx2_mock.get_request().content)
    assert body["questions"][0]["instructions"] == "Apply these rules\n\nRefund?"


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"system": " "},
        {"system": "Q", "answer_type": "bad"},
        {"system": "Q", "answer_type": "choice"},
        {"system": "Q", "answer_type": "score", "levels": ["Only one"]},
        {"system": "Q", "choices": {"a": None, "b": None}},
        {"system": "Q", "levels": ["a", "b"]},
        {
            "system": "Q",
            "answer_type": "choice",
            "choices": [{"value": "a"}, {"value": "a"}],
        },
        {
            "system": "Q",
            "answer_type": "choice",
            "choices": [{"value": 1}, {"value": 2}],
        },
        {"system": "Q", "answer_type": "choice", "choices": '{"a":"x","a":"y"}'},
        {"system": "Q", "levels": "[NaN, 1e999]"},
        {"system": "Q", "levels": "{broken"},
        {"system": "Q", "name": " "},
        {"questions": []},
        {"questions": [{"type": "predicate", "name": "q", "instructions": "Q"}] * 2},
        {"questions": [{"type": "predicate", "name": "q", "instructions": " "}]},
        {
            "questions": [
                {"type": "predicate", "name": "q", "instructions": "Q", "unknown": 1}
            ]
        },
        {
            "questions": [{"type": "predicate", "name": "q", "instructions": "Q"}],
            "system": "Conflict",
        },
        {
            "questions": [{"type": "predicate", "name": "q", "instructions": "Q"}],
            "answer_type": "score",
        },
    ],
)
@pytest.mark.asyncio
async def test_invalid_options(kwargs, async_, httpx2_mock):
    with pytest.raises(ValueError):
        await evaluate(async_, **kwargs)
    assert not httpx2_mock.get_requests()


def test_boolean_choices():
    options = DecisionsOptions(
        answer_type="choice",
        choices='[{"value":true},{"value":"true"},{"value":false}]',
    )
    assert [c["value"] for c in options.choices] == [True, "true", False]


@pytest.mark.parametrize("text", [None, "", " "])
@pytest.mark.asyncio
async def test_empty_input(text, async_, httpx2_mock):
    with pytest.raises(ValueError, match="Supply input"):
        await evaluate(async_, text, system="Q")
    assert not httpx2_mock.get_requests()


def test_unsupported_parts(httpx2_mock):
    for messages in (
        [llm.system(llm.Attachment(content=PNG, type="image/png")), llm.user("Text")],
        [llm.system("Q"), llm.user(llm.parts.ReasoningPart(text="Unsupported"))],
    ):
        with pytest.raises(ValueError):
            llm.get_model(MODEL).prompt(messages=messages).text()
    assert not httpx2_mock.get_requests()


def test_image_limit(httpx2_mock):
    with pytest.raises(ValueError, match="128 images"):
        llm.get_model(MODEL).prompt(
            system="Q",
            attachments=[llm.Attachment(content=PNG, type="image/png")] * 129,
        ).text()
    assert not httpx2_mock.get_requests()


@pytest.mark.parametrize("status", [302, 401, 422, 429, 500])
@pytest.mark.asyncio
async def test_http_errors(status, async_, httpx2_mock):
    httpx2_mock.add_response(
        status_code=status,
        json={"error": {"message": "Request failed"}},
        headers={"Location": "https://example.com/redirect"},
    )
    with pytest.raises(llm.ModelError, match=f"HTTP {status}.*Request failed"):
        await evaluate(async_, system="Q")
    assert len(httpx2_mock.get_requests()) == 1


@pytest.mark.asyncio
async def test_transport_error(async_, httpx2_mock):
    httpx2_mock.add_exception(httpx2.ConnectError("Connection refused"))
    with pytest.raises(llm.ModelError, match="Connection refused"):
        await evaluate(async_, system="Q")


@pytest.mark.parametrize(
    "bad", [b"{broken", b"\xff", b"{}", b"[]", b'{"model":"x","model":"y"}']
)
@pytest.mark.asyncio
async def test_invalid_json_response(bad, async_, httpx2_mock):
    httpx2_mock.add_response(content=bad)
    with pytest.raises(llm.ModelError, match="Invalid OpenAI Decisions response"):
        await evaluate(async_, system="Q")


@pytest.mark.parametrize(
    "change",
    [
        {"answers": []},
        {"answers": [None]},
        {"answers": [{"name": "evaluation", "type": "predicate"}]},
        {"answers": [{"name": "wrong", "type": "predicate", "probability": 0.5}]},
        {"answers": [{"name": "evaluation", "type": "predicate", "probability": 1.5}]},
        {"answers": [{"name": "evaluation", "type": "predicate", "probability": True}]},
        {"answers": [ANSWERS["score"]]},
        {"model": None},
        {"usage": {"input_tokens": True, "output_tokens": 0}},
    ],
)
@pytest.mark.asyncio
async def test_invalid_response_shape(change, async_, httpx2_mock):
    httpx2_mock.add_response(json={**payload(), **change})
    with pytest.raises(llm.ModelError, match="Invalid OpenAI Decisions response"):
        await evaluate(async_, system="Q")


@pytest.mark.asyncio
async def test_replies_rejected(async_, httpx2_mock):
    httpx2_mock.add_response(json=payload())
    response, _ = await evaluate(async_, system="Q")
    with pytest.raises(llm.ConversationNotSupported):
        if async_:
            await response.reply("Again")
        else:
            response.reply("Again")
    assert len(httpx2_mock.get_requests()) == 1


def test_cli_logging_and_templates(httpx2_mock):
    runner = CliRunner()
    saved = runner.invoke(cli, ["-m", MODEL, "-s", "Refund?", "--save", "refund"])
    assert saved.exit_code == 0, saved.output
    httpx2_mock.add_response(json=payload())
    result = runner.invoke(cli, ["-t", "refund"], input="Refund my payment")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == ANSWERS["predicate"]
    logs = runner.invoke(cli, ["logs", "--json"])
    assert logs.exit_code == 0, logs.output
    assert "gpt-6-luna" in logs.output
    assert "test-openai-secret" not in logs.output
    for args in [["-c", "Again"], ["chat", "-m", MODEL]]:
        result = runner.invoke(cli, args)
        assert result.exit_code != 0
        assert "does not support conversations" in result.output


@pytest.mark.parametrize("kind", ["choice", "score"])
def test_cli_json_options(kind, httpx2_mock):
    httpx2_mock.add_response(json=payload(kind))
    option, value = next(iter(OPTIONS[kind].items()))
    result = CliRunner().invoke(
        cli,
        [
            "-m",
            MODEL,
            "Input",
            "-s",
            "Q",
            "-o",
            "answer_type",
            kind,
            "-o",
            option,
            json.dumps(value),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == ANSWERS[kind]


def test_no_key(monkeypatch, httpx2_mock):
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(llm.NeedsKeyException):
        llm.get_model(MODEL).prompt("Input", system="Q").text()
    assert not httpx2_mock.get_requests()


def test_cli_multiple_questions_template_and_logging(httpx2_mock):
    questions = [{"type": "predicate", "name": "evaluation", "instructions": "Refund?"}]
    runner = CliRunner()
    saved = runner.invoke(
        cli,
        ["-m", MODEL, "-o", "questions", json.dumps(questions), "--save", "questions"],
    )
    assert saved.exit_code == 0, saved.output
    httpx2_mock.add_response(json=payload())
    result = runner.invoke(cli, ["-t", "questions"], input="Refund my payment")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == [ANSWERS["predicate"]]
    logs = runner.invoke(cli, ["logs", "--json"])
    assert logs.exit_code == 0, logs.output
    # Options written to logs must remain valid when LLM reconstructs a response.
    record = json.loads(logs.output)[0]
    stored_options = record["options_json"]
    if isinstance(stored_options, str):
        stored_options = json.loads(stored_options)
    assert DecisionsOptions(**stored_options).questions == questions
