"""Tests for the engine that reads text with a model and places it with PaddleOCR."""

import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pdf_refinery.llm_engine import (
    CodexEngine,
    align_transcription,
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
