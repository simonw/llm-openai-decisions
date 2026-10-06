"""Typed, single-turn evaluations using the OpenAI Decisions API."""

import asyncio
import json
from typing import Annotated, Literal

import httpx2
import llm
from llm.parts import AttachmentPart, TextPart
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    TypeAdapter,
    field_validator,
    model_validator,
)

API_URL = "https://api.openai.com/v1/decisions"


def parse_json(value):
    def unique_object(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = item
        return result

    result = json.loads(value, object_pairs_hook=unique_object)
    json.dumps(result, allow_nan=False)
    return result


class APIObject(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Choice(APIObject):
    value: StrictStr | StrictBool
    description: StrictStr | None = None


class Level(APIObject):
    label: StrictStr
    description: StrictStr | None = None


class Question(APIObject):
    name: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    type: Literal["predicate", "choice", "score"]
    choices: list[Choice] | None = None
    levels: list[Level] | None = None

    @field_validator("name", "instructions")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Must not be blank")
        return value

    @model_validator(mode="after")
    def valid_question(self):
        if self.type == "choice":
            if self.choices is None or len(self.choices) < 2:
                raise ValueError("choice questions need at least two choices")
            values = [(type(c.value), c.value) for c in self.choices]
            if len(set(values)) != len(values):
                raise ValueError("choice values must be unique")
        elif self.choices is not None:
            raise ValueError("choices are only supported for choice questions")
        if self.type == "score":
            if self.levels is None or len(self.levels) < 2:
                raise ValueError("score questions need at least two ordered levels")
        elif self.levels is not None:
            raise ValueError("levels are only supported for score questions")
        return self


class DecisionsOptions(llm.Options):
    answer_type: Literal["predicate", "choice", "score"] = Field(
        default="predicate",
        description="Single question type: predicate, choice, or score.",
    )
    name: str = Field(
        default="evaluation", min_length=1, description="Single question name."
    )
    choices: list[dict] | None = Field(
        default=None,
        description="JSON array of {value, description} choices, or an object mapping values to descriptions.",
    )
    levels: list[dict] | None = Field(
        default=None,
        description="JSON array of level labels or {label, description} objects, lowest to highest.",
    )
    questions: list[dict] | None = Field(
        default=None,
        description="JSON array of named questions; replaces the single system-prompt question.",
    )

    @field_validator("choices", "levels", "questions", mode="before")
    @classmethod
    def parse_options(cls, value, info):
        if isinstance(value, str):
            try:
                value = parse_json(value)
            except ValueError:
                raise ValueError(
                    "Expected valid JSON with unique keys and finite numbers"
                ) from None
        if info.field_name == "choices" and isinstance(value, dict):
            value = [{"value": k, "description": v} for k, v in value.items()]
        if info.field_name == "levels" and isinstance(value, list):
            value = [{"label": v} if isinstance(v, str) else v for v in value]
        if isinstance(value, list):
            item_type = {"choices": Choice, "levels": Level, "questions": Question}[
                info.field_name
            ]
            # LLM logs dict(options), so nested values must be JSON serializable.
            value = [
                item_type.model_validate(v).model_dump(exclude_none=True) for v in value
            ]
        return value

    @model_validator(mode="after")
    def valid_options(self):
        if self.questions is not None:
            if not self.questions:
                raise ValueError("questions must not be empty")
            if (
                self.answer_type != "predicate"
                or self.name != "evaluation"
                or self.choices is not None
                or self.levels is not None
            ):
                raise ValueError(
                    "questions cannot be combined with answer_type, name, choices, or levels"
                )
            names = [q["name"] for q in self.questions]
            if len(set(names)) != len(names):
                raise ValueError("Question names must be unique")
        else:
            self.question("Validate configuration")
        return self

    def question(self, instructions):
        return Question(
            name=self.name,
            instructions=instructions,
            type=self.answer_type,
            choices=self.choices,
            levels=self.levels,
        )


# Validate required result fields without discarding future provider metadata.
class Answer(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, allow_inf_nan=False)
    name: str


Probability = Annotated[float, Field(ge=0, le=1)]


class PredicateAnswer(Answer):
    type: Literal["predicate"]
    probability: Probability


class ChoiceProbability(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)
    value: StrictStr | StrictBool
    probability: Probability


class ChoiceAnswer(Answer):
    type: Literal["choice"]
    choice: StrictStr | StrictBool
    confidence: Probability
    probabilities: list[ChoiceProbability] = Field(min_length=2)


class LevelProbability(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)
    value: int
    label: str
    probability: Probability


class ScoreAnswer(Answer):
    type: Literal["score"]
    score: float
    confidence: Probability
    probabilities: list[LevelProbability] = Field(min_length=2)


class RefusalAnswer(Answer):
    type: Literal["refusal"]


ANSWER = TypeAdapter(
    Annotated[
        PredicateAnswer | ChoiceAnswer | ScoreAnswer | RefusalAnswer,
        Field(discriminator="type"),
    ]
)


class _Decisions:
    model_id = "openai-decisions/gpt-6-luna"
    needs_key = "openai"
    key_env_var = "OPENAI_API_KEY"
    can_stream = False
    supports_conversation = False
    supports_schema = False
    supports_tools = False
    attachment_types = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
    Options = DecisionsOptions

    def build_payload(self, prompt):
        inputs, instructions = [], []
        image_count = 0
        for message in prompt.messages:
            if message.role not in ("system", "user"):
                raise ValueError("Decisions supports single-turn evaluation only.")
            content = []
            for part in message.parts:
                if isinstance(part, TextPart):
                    if message.role == "system":
                        instructions.append(part.text)
                    elif part.text.strip():
                        content.append({"type": "input_text", "text": part.text})
                elif (
                    isinstance(part, AttachmentPart)
                    and message.role == "user"
                    and part.attachment
                ):
                    attachment = part.attachment
                    mime_type = attachment.resolve_type()
                    if mime_type not in self.attachment_types:
                        raise ValueError("Decisions accepts image attachments only.")
                    image_count += 1
                    if image_count > 128:
                        raise ValueError("Decisions accepts at most 128 images.")
                    content.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:{mime_type};base64,{attachment.base64_content()}",
                        }
                    )
                else:
                    raise ValueError(
                        "Decisions accepts user text and images, and system text only."
                    )
            if content:
                inputs.append({"role": "user", "content": content})
        if not inputs:
            raise ValueError("Supply input text or an image as the prompt or on stdin.")
        question_text = "\n\n".join(instructions)
        if prompt.options.questions is not None:
            if question_text.strip():
                raise ValueError(
                    "Do not combine questions with a system prompt; put instructions in each question."
                )
            questions = [Question.model_validate(q) for q in prompt.options.questions]
        else:
            if not question_text.strip():
                raise ValueError(
                    "Supply a question using -s/--system or system= in Python."
                )
            questions = [prompt.options.question(question_text)]
        # Keep text-only requests simple, while preserving boundaries in explicit messages.
        if len(inputs) == 1 and image_count == 0:
            inputs = "\n\n".join(p["text"] for p in inputs[0]["content"])
        return {
            "model": "gpt-6-luna",
            "input": inputs,
            "questions": [q.model_dump(exclude_none=True) for q in questions],
        }

    def finish_response(self, response, result, payload, multiple):
        if result.status_code != 200:
            raise llm.ModelError(
                f"OpenAI Decisions HTTP {result.status_code}: {result.text}"
            )
        try:
            data = parse_json(result.content)
            answers, model, usage = data["answers"], data["model"], data["usage"]
            if not isinstance(answers, list) or len(answers) != len(
                payload["questions"]
            ):
                raise ValueError("Expected one answer per question")
            for answer, question in zip(answers, payload["questions"]):
                parsed = ANSWER.validate_python(answer)
                if parsed.name != question["name"] or parsed.type not in (
                    question["type"],
                    "refusal",
                ):
                    raise ValueError("Answer does not match question")
            if not isinstance(model, str) or not model.strip():
                raise ValueError("Expected a model name")
            if not all(
                type(usage[k]) is int and usage[k] >= 0
                for k in ("input_tokens", "output_tokens")
            ):
                raise ValueError("Expected token counts")
        except (ValueError, KeyError, TypeError, UnicodeError):
            raise llm.ModelError(
                "Invalid OpenAI Decisions response: expected answers, model, and token usage."
            ) from None
        response.response_json = data
        details = {
            k: v
            for k, v in usage.items()
            if k not in ("input_tokens", "output_tokens", "total_tokens")
        }
        response.set_usage(
            input=usage["input_tokens"],
            output=usage["output_tokens"],
            details=details or None,
        )
        response.set_resolved_model(model)
        return json.dumps(
            answers if multiple else answers[0], ensure_ascii=False, allow_nan=False
        )


class Decisions(_Decisions, llm.KeyModel):
    def execute(self, prompt, stream, response, conversation, key=None):
        try:
            payload = self.build_payload(prompt)
            with httpx2.Client(timeout=60, follow_redirects=False) as client:
                result = client.post(
                    API_URL, json=payload, headers={"Authorization": f"Bearer {key}"}
                )
        except httpx2.HTTPError as ex:
            raise llm.ModelError(f"OpenAI Decisions request failed: {ex}") from None
        yield self.finish_response(
            response, result, payload, prompt.options.questions is not None
        )


class AsyncDecisions(_Decisions, llm.AsyncKeyModel):
    async def execute(self, prompt, stream, response, conversation, key=None):
        try:
            payload = await asyncio.to_thread(self.build_payload, prompt)
            async with httpx2.AsyncClient(timeout=60, follow_redirects=False) as client:
                result = await client.post(
                    API_URL, json=payload, headers={"Authorization": f"Bearer {key}"}
                )
        except httpx2.HTTPError as ex:
            raise llm.ModelError(f"OpenAI Decisions request failed: {ex}") from None
        yield self.finish_response(
            response, result, payload, prompt.options.questions is not None
        )


@llm.hookimpl
def register_models(register):
    register(Decisions(), AsyncDecisions())
