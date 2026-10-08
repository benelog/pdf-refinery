"""Tests for the engine that reads text with a model and places it with PaddleOCR."""

import base64
import json
import os
import subprocess
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pdf_refinery import llm_engine
from pdf_refinery.llm_engine import (
    ClaudeEngine,
    CodexEngine,
    ConsensusEngine,
    align_transcription,
    merge_readings,
    transcribe_with_claude,
    transcribe_with_codex,
)
from pdf_refinery.ocr_engine import OcrResult


class TestAlignTranscription:
    """Each PaddleOCR line gets the model's reading of the same text."""

    def test_misreads_are_replaced_line_by_line(self):
        lines = ["이 해는 나가오 카와 통슨이", "원자 모형을 제안한 해였다."]
        page = "이 해는 나가오 카와 톰슨이\n원자 모형을 제안한 해였다.\n"
        assert align_transcription(lines, page) == [
            "이 해는 나가오 카와 톰슨이", "원자 모형을 제안한 해였다.",
        ]

    def test_the_model_s_spacing_is_kept(self):
        # Spacing is most of what the model adds on Korean: PaddleOCR runs
        # the words of large type together.
        assert align_transcription(["A버튼(라이트버튼)"], "A 버튼 (라이트 버튼)") == [
            "A 버튼 (라이트 버튼)",
        ]

    def test_characters_only_the_model_read_open_the_next_line(self):
        # PaddleOCR's dictionary has no corner brackets, so they arrive as
        # insertions; the model's line break says which line they start.
        lines = ["이라는 제목을 붙였다.", "흐름의 진동에 관한 것]"]
        page = "이라는 제목을 붙였다.\n「흐름의 진동에 관한 것」"
        assert align_transcription(lines, page) == [
            "이라는 제목을 붙였다.", "「흐름의 진동에 관한 것」",
        ]

    def test_the_model_s_wrapping_does_not_decide_the_lines(self):
        # The boxes are PaddleOCR's; the model breaking a line elsewhere
        # must not move text between them.
        lines = ["first line of text", "second line"]
        page = "first line\nof text second\nline"
        assert align_transcription(lines, page) == ["first line of text", "second line"]

    def test_a_line_the_model_did_not_read_keeps_paddle_s_text(self):
        lines = ["알람소리와 함께", "시간 세팅 모드에 진입합니다."]
        assert align_transcription(lines, "시간 세팅 모드에 진입합니다.") == [
            None, "시간 세팅 모드에 진입합니다.",
        ]

    def test_a_speck_both_readings_call_empty_is_dropped(self):
        lines = ["사용 설명서", "#", "A 버튼"]
        assert align_transcription(lines, "사용 설명서\nA 버튼") == [
            "사용 설명서", "", "A 버튼",
        ]

    def test_punctuation_the_model_did_read_is_not_dropped(self):
        # A period wrapped onto a box of its own is real text.
        lines = ["그런 대로 끝났다", "."]
        assert align_transcription(lines, "그런 대로 끝났다\n.") == ["그런 대로 끝났다", "."]

    def test_an_alignment_that_no_longer_resembles_the_box_is_refused(self):
        # A model rewriting a line wholesale -- what GPT-6 Luna did on the
        # benchmark -- must not replace what PaddleOCR actually read.
        lines = ["밑에서 공부하며 이번에는 원자 구조에 관한 연구에 몰두하"]
        assert align_transcription(lines, "를 만나") == [None]

    def test_an_empty_transcription_changes_nothing(self):
        assert align_transcription(["some text"], "") == [None]


class TestCodexEngine:
    @staticmethod
    def _paddle_results():
        return [
            OcrResult("거두었다.단지", 0.9, [[0, 0], [10, 0], [10, 5], [0, 5]]),
            OcrResult("*", 0.6, [[0, 8], [2, 8], [2, 10], [0, 10]]),
        ]

    @patch("pdf_refinery.llm_engine.transcribe_with_codex")
    @patch("pdf_refinery.llm_engine.PaddleEngine")
    def test_boxes_come_from_paddle_and_text_from_the_model(self, paddle_cls, transcribe):
        paddle_cls.return_value.recognize.return_value = self._paddle_results()
        transcribe.return_value = "거두었다. 단지\n"

        results = CodexEngine(lang="korean").recognize(np.zeros((20, 20, 3), np.uint8))

        assert [r.text for r in results] == ["거두었다. 단지"]
        assert results[0].bbox == [[0, 0], [10, 0], [10, 5], [0, 5]]

    @patch("pdf_refinery.llm_engine.transcribe_with_codex")
    @patch("pdf_refinery.llm_engine.PaddleEngine")
    def test_a_failed_call_keeps_paddle_s_page(self, paddle_cls, transcribe, capsys):
        paddle_cls.return_value.recognize.return_value = self._paddle_results()
        transcribe.side_effect = RuntimeError("codex exited 1: not logged in")

        engine = CodexEngine(lang="korean")
        results = engine.recognize(np.zeros((20, 20, 3), np.uint8))

        assert [r.text for r in results] == ["거두었다.단지", "*"]
        assert engine.pages_kept_as_paddle == 1
        assert "not logged in" in capsys.readouterr().err

    @patch("pdf_refinery.llm_engine.transcribe_with_codex")
    @patch("pdf_refinery.llm_engine.PaddleEngine")
    def test_a_page_with_no_boxes_is_not_sent(self, paddle_cls, transcribe):
        paddle_cls.return_value.recognize.return_value = []
        CodexEngine(lang="korean").recognize(np.zeros((20, 20, 3), np.uint8))
        transcribe.assert_not_called()

    @patch("pdf_refinery.llm_engine.PaddleEngine")
    def test_paddle_options_reach_paddle(self, paddle_cls):
        CodexEngine(lang="korean", codex_model="gpt-6-luna", unwarp=True)
        paddle_cls.assert_called_once_with(lang="korean", unwarp=True)


class TestTranscribeWithCodex:
    @staticmethod
    def _fake_codex(answer: str | None, returncode: int = 0):
        def run(command, **kwargs):
            if answer is not None:
                out = command[command.index("-o") + 1]
                with open(out, "w", encoding="utf-8") as handle:
                    handle.write(answer)
            return MagicMock(returncode=returncode, stdout="", stderr="boom\n")
        return run

    def test_the_answer_file_is_returned(self):
        with patch("subprocess.run", side_effect=self._fake_codex("text\n")) as run:
            assert transcribe_with_codex(np.zeros((4, 4, 3), np.uint8)) == "text\n"
        command = run.call_args.args[0]
        assert command[:2] == ["codex", "exec"]
        assert command[command.index("-m") + 1] == "gpt-6-sol"
        assert command[command.index("-s") + 1] == "read-only"

    def test_a_failing_exit_is_reported(self):
        with patch("subprocess.run", side_effect=self._fake_codex(None, returncode=1)):
            with pytest.raises(RuntimeError, match="exited 1: boom"):
                transcribe_with_codex(np.zeros((4, 4, 3), np.uint8))

    def test_an_empty_answer_is_a_failure(self):
        with patch("subprocess.run", side_effect=self._fake_codex("  \n")):
            with pytest.raises(RuntimeError, match="empty"):
                transcribe_with_codex(np.zeros((4, 4, 3), np.uint8))

    def test_a_timeout_is_a_failure(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("codex", 1)):
            with pytest.raises(RuntimeError, match="did not answer"):
                transcribe_with_codex(np.zeros((4, 4, 3), np.uint8))


class TestMergeReadings:
    """Two models' readings of a line, with PaddleOCR settling disagreements."""

    def test_agreement_is_taken_as_it_is(self):
        assert merge_readings("이 해는 통슨이", "이 해는 톰슨이", "이 해는 톰슨이") == "이 해는 톰슨이"

    def test_a_line_only_one_model_read_takes_that_reading(self):
        # None is "this model had nothing usable for the line"; the other
        # model's reading is used, as that model's own engine would.
        assert merge_readings("A버튼", None, "A 버튼") == "A 버튼"
        assert merge_readings("A버튼", "A 버튼", None) == "A 버튼"

    def test_a_drop_is_overruled_by_a_model_that_read_text(self):
        assert merge_readings(".", "", ".") == "."
        assert merge_readings(".", ".", "") == "."

    def test_a_line_neither_model_read_keeps_paddle_s_text(self):
        assert merge_readings("밑에서 공부하며", None, None) is None

    def test_a_line_is_dropped_only_when_both_models_drop_it(self):
        assert merge_readings("#", "", "") == ""
        # One model has a reading that did not resemble the box, the other
        # would drop it: not both, so the box keeps what PaddleOCR read.
        assert merge_readings("#", "", None) is None
        assert merge_readings("#", None, "") is None

    def test_a_disagreement_goes_to_the_reading_paddle_supports(self):
        # Each model misread a different character; PaddleOCR read both of
        # them right, so each disagreement goes the other way.
        paddle = "원자 모형을 제안한 해였다."
        a = "원자 모형울 제안한 해였다."
        b = "원자 모형을 제안한 해었다."
        assert merge_readings(paddle, a, b) == "원자 모형을 제안한 해였다."
        assert merge_readings(paddle, b, a) == "원자 모형을 제안한 해였다."

    def test_what_both_models_agree_on_is_kept_even_against_paddle(self):
        # PaddleOCR has no corner brackets; both models do, so they stay,
        # while the one character the models disagree on is still voted.
        paddle = "흐름의 진동에 관한 것]"
        a = "「흐름의 진둥에 관한 것」"
        b = "「흐름의 진동에 관한 것」"
        assert merge_readings(paddle, a, b) == "「흐름의 진동에 관한 것」"

    def test_a_tie_goes_to_the_preferred_reading(self):
        # PaddleOCR read neither version, so it cannot decide.
        assert merge_readings("cat", "dog", "fog") == "dog"
        assert merge_readings("cat", "fog", "dog") == "fog"

    def test_spacing_is_decided_by_the_preferred_reading_not_by_paddle(self):
        # PaddleOCR runs the words of large type together. Letting it vote on
        # spaces would remove the ones a word search depends on.
        paddle = "A버튼(라이트버튼)"
        assert merge_readings(paddle, "A 버튼 (라이트 버튼)", "A버튼 (라이트버튼)") == (
            "A 버튼 (라이트 버튼)"
        )
        assert merge_readings(paddle, "A버튼 (라이트버튼)", "A 버튼 (라이트 버튼)") == (
            "A버튼 (라이트버튼)"
        )


def _stream(result: str | None = "page text\n", *, is_error=False,
            subtype="success") -> str:
    """What claude -p --output-format stream-json prints for one turn."""
    events = [
        {"type": "system", "subtype": "init", "model": "claude-opus-5-5"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "x"}]}},
    ]
    if result is not None or is_error:
        events.append({"type": "result", "subtype": subtype, "is_error": is_error,
                       "result": result})
    return "\n".join(json.dumps(e) for e in events) + "\n"


class TestTranscribeWithClaude:
    def test_the_result_event_is_returned(self):
        proc = MagicMock(returncode=0, stdout=_stream("page text\n"), stderr="")
        with patch("subprocess.run", return_value=proc) as run:
            assert transcribe_with_claude(np.zeros((4, 4, 3), np.uint8)) == "page text\n"

        command = run.call_args.args[0]
        assert command[:2] == ["claude", "-p"]
        assert command[command.index("--model") + 1] == "opus"
        assert command[command.index("--input-format") + 1] == "stream-json"
        assert command[command.index("--output-format") + 1] == "stream-json"
        # No tools, no session left behind, no user settings or MCP servers.
        assert command[command.index("--tools") + 1] == ""
        assert command[command.index("--setting-sources") + 1] == ""
        assert "--no-session-persistence" in command
        assert "--strict-mcp-config" in command
        assert "--verbose" in command

        kwargs = run.call_args.kwargs
        assert kwargs["cwd"] != os.getcwd()
        assert kwargs["timeout"] == llm_engine.CLAUDE_TIMEOUT_SECONDS
        message = json.loads(kwargs["input"])
        assert message["type"] == "user"
        image, text = message["message"]["content"]
        assert image["type"] == "image"
        assert image["source"]["media_type"] == "image/png"
        assert base64.b64decode(image["source"]["data"]).startswith(b"\x89PNG")
        assert text == {"type": "text", "text": llm_engine.TRANSCRIPTION_PROMPT}

    def test_the_model_is_passed_through(self):
        proc = MagicMock(returncode=0, stdout=_stream(), stderr="")
        with patch("subprocess.run", return_value=proc) as run:
            transcribe_with_claude(np.zeros((4, 4, 3), np.uint8), model="sonnet")
        command = run.call_args.args[0]
        assert command[command.index("--model") + 1] == "sonnet"

    def test_a_failing_exit_reports_the_result_text(self):
        # Claude Code puts "not logged in" and the like in the result event,
        # not on stderr.
        proc = MagicMock(
            returncode=1, stderr="",
            stdout=_stream("Not logged in", is_error=True, subtype="success"),
        )
        with patch("subprocess.run", return_value=proc):
            with pytest.raises(RuntimeError, match="exited 1: Not logged in"):
                transcribe_with_claude(np.zeros((4, 4, 3), np.uint8))

    def test_a_failing_exit_without_a_result_reports_stderr(self):
        proc = MagicMock(returncode=2, stdout="", stderr="error: unknown option\n")
        with patch("subprocess.run", return_value=proc):
            with pytest.raises(RuntimeError, match="exited 2: error: unknown option"):
                transcribe_with_claude(np.zeros((4, 4, 3), np.uint8))

    def test_an_error_result_is_a_failure_even_on_a_clean_exit(self):
        proc = MagicMock(returncode=0, stderr="", stdout=_stream(
            None, is_error=True, subtype="error_during_execution"))
        with patch("subprocess.run", return_value=proc):
            with pytest.raises(RuntimeError, match="error_during_execution"):
                transcribe_with_claude(np.zeros((4, 4, 3), np.uint8))

    def test_an_empty_answer_is_a_failure(self):
        proc = MagicMock(returncode=0, stdout=_stream("  \n"), stderr="")
        with patch("subprocess.run", return_value=proc):
            with pytest.raises(RuntimeError, match="empty"):
                transcribe_with_claude(np.zeros((4, 4, 3), np.uint8))

    def test_no_result_event_is_a_failure(self):
        proc = MagicMock(returncode=0, stdout=_stream(None), stderr="")
        with patch("subprocess.run", return_value=proc):
            with pytest.raises(RuntimeError, match="without a result"):
                transcribe_with_claude(np.zeros((4, 4, 3), np.uint8))

    def test_a_timeout_is_a_failure(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("claude", 1)):
            with pytest.raises(RuntimeError, match="did not answer"):
                transcribe_with_claude(np.zeros((4, 4, 3), np.uint8))


def _paddle_results():
    return [
        OcrResult("거두었다.단지", 0.9, [[0, 0], [10, 0], [10, 5], [0, 5]]),
        OcrResult("*", 0.6, [[0, 8], [2, 8], [2, 10], [0, 10]]),
    ]


class TestClaudeEngine:
    @patch("pdf_refinery.llm_engine.transcribe_with_claude")
    @patch("pdf_refinery.llm_engine.PaddleEngine")
    def test_boxes_come_from_paddle_and_text_from_the_model(self, paddle_cls, transcribe):
        paddle_cls.return_value.recognize.return_value = _paddle_results()
        transcribe.return_value = "거두었다. 단지\n"
        rendered = np.ones((20, 20, 3), np.uint8)

        results = ClaudeEngine(lang="korean", claude_model="sonnet").recognize(
            np.zeros((20, 20, 3), np.uint8), rendered=rendered,
        )

        assert [r.text for r in results] == ["거두었다. 단지"]
        assert results[0].bbox == [[0, 0], [10, 0], [10, 5], [0, 5]]
        assert transcribe.call_args.args[0] is rendered
        assert transcribe.call_args.kwargs["model"] == "sonnet"

    @patch("pdf_refinery.llm_engine.transcribe_with_claude")
    @patch("pdf_refinery.llm_engine.PaddleEngine")
    def test_a_failed_call_keeps_paddle_s_page(self, paddle_cls, transcribe, capsys):
        paddle_cls.return_value.recognize.return_value = _paddle_results()
        transcribe.side_effect = RuntimeError("claude exited 1: Not logged in")

        engine = ClaudeEngine(lang="korean")
        results = engine.recognize(np.zeros((20, 20, 3), np.uint8))

        assert [r.text for r in results] == ["거두었다.단지", "*"]
        assert engine.pages_kept_as_paddle == 1
        assert "Not logged in" in capsys.readouterr().err

    @patch("pdf_refinery.llm_engine.PaddleEngine")
    def test_paddle_options_reach_paddle(self, paddle_cls):
        ClaudeEngine(lang="korean", claude_model="sonnet", unwarp=True)
        paddle_cls.assert_called_once_with(lang="korean", unwarp=True)


@patch("pdf_refinery.llm_engine.transcribe_with_claude")
@patch("pdf_refinery.llm_engine.transcribe_with_codex")
@patch("pdf_refinery.llm_engine.PaddleEngine")
class TestConsensusEngine:
    @staticmethod
    def _lines():
        return [
            OcrResult("원자 모형을 제안한 해였다.", 0.9, [[0, 0], [10, 0], [10, 5], [0, 5]]),
            OcrResult("A버튼", 0.9, [[0, 8], [10, 8], [10, 12], [0, 12]]),
            OcrResult("#", 0.6, [[0, 14], [2, 14], [2, 16], [0, 16]]),
        ]

    def test_the_two_readings_are_voted_line_by_line(self, paddle_cls, codex, claude):
        paddle_cls.return_value.recognize.return_value = self._lines()
        codex.return_value = "원자 모형울 제안한 해였다.\nA 버튼\n"
        claude.return_value = "원자 모형을 제안한 해었다.\nA 버튼\n"

        engine = ConsensusEngine(lang="korean", codex_model="gpt-6-sol",
                                 claude_model="opus")
        results = engine.recognize(np.zeros((20, 20, 3), np.uint8))

        # Each model's misread is outvoted; the speck both skipped is dropped.
        assert [r.text for r in results] == ["원자 모형을 제안한 해였다.", "A 버튼"]
        assert codex.call_args.kwargs["model"] == "gpt-6-sol"
        assert claude.call_args.kwargs["model"] == "opus"

    def test_both_models_are_asked_at_the_same_time(self, paddle_cls, codex, claude):
        # Each call waits for the other to have started; called one after the
        # other, the first would wait out the barrier and fail.
        paddle_cls.return_value.recognize.return_value = self._lines()
        barrier = threading.Barrier(2, timeout=5)

        def answer(image, model):
            barrier.wait()
            return "원자 모형을 제안한 해였다.\nA 버튼\n"

        codex.side_effect = claude.side_effect = answer
        results = ConsensusEngine(lang="korean").recognize(np.zeros((20, 20, 3), np.uint8))
        assert [r.text for r in results] == ["원자 모형을 제안한 해였다.", "A 버튼"]

    def test_one_failed_call_leaves_the_other_model_alone(
        self, paddle_cls, codex, claude, capsys
    ):
        paddle_cls.return_value.recognize.return_value = self._lines()
        codex.side_effect = RuntimeError("codex exited 1: usage limit")
        claude.return_value = "원자 모형을 제안한 해었다.\nA 버튼\n"

        engine = ConsensusEngine(lang="korean")
        results = engine.recognize(np.zeros((20, 20, 3), np.uint8))

        # Exactly what the claude engine alone would have produced.
        assert [r.text for r in results] == ["원자 모형을 제안한 해었다.", "A 버튼"]
        assert engine.pages_read_by_one_model == 1
        assert engine.pages_kept_as_paddle == 0
        err = capsys.readouterr().err
        assert "usage limit" in err and "read by claude alone" in err

    def test_two_failed_calls_keep_paddle_s_page(self, paddle_cls, codex, claude, capsys):
        paddle_cls.return_value.recognize.return_value = self._lines()
        codex.side_effect = RuntimeError("codex exited 1: usage limit")
        claude.side_effect = RuntimeError("claude exited 1: Not logged in")

        engine = ConsensusEngine(lang="korean")
        results = engine.recognize(np.zeros((20, 20, 3), np.uint8))

        assert [r.text for r in results] == ["원자 모형을 제안한 해였다.", "A버튼", "#"]
        assert engine.pages_kept_as_paddle == 1
        err = capsys.readouterr().err
        assert "usage limit" in err and "Not logged in" in err
        assert "keeps PaddleOCR's text" in err

    def test_the_models_read_the_page_as_rendered(self, paddle_cls, codex, claude):
        paddle_cls.return_value.recognize.return_value = self._lines()
        codex.return_value = claude.return_value = "원자 모형을 제안한 해였다.\n"
        rendered = np.ones((20, 20, 3), np.uint8)

        ConsensusEngine(lang="korean").recognize(
            np.zeros((20, 20, 3), np.uint8), rendered=rendered
        )

        assert codex.call_args.args[0] is rendered
        assert claude.call_args.args[0] is rendered

    def test_a_page_with_no_boxes_is_not_sent(self, paddle_cls, codex, claude):
        paddle_cls.return_value.recognize.return_value = []
        ConsensusEngine(lang="korean").recognize(np.zeros((20, 20, 3), np.uint8))
        codex.assert_not_called()
        claude.assert_not_called()
