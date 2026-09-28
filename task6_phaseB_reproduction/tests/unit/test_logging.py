"""Log handlers may retain stderr after an earlier task's log has closed."""
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch
import logging
import multiprocessing
import sys

from task6_phaseb.common.io import terminal_log


class StrictHandler(logging.StreamHandler):
    def handleError(self, record):
        raise  # Make otherwise non-fatal logging errors visible to this regression.


def retained_logger():
    logger = logging.Logger("retained-stream-regression", level=logging.WARNING)
    logger.propagate = False
    logger.addHandler(StrictHandler(sys.stderr))
    return logger


def emit_child(logger, pipe):
    try:
        logger.warning("from-child")
        pipe.send("ok")
    except BaseException as error:
        pipe.send(repr(error))
    finally:
        pipe.close()


class LoggingTests(TestCase):
    def test_retained_handler_follows_next_task_without_reopening_old_log(self):
        with TemporaryDirectory() as temp, patch("sys.stdout", new=StringIO()), patch("sys.stderr", new=StringIO()) as console:
            first, second = Path(temp) / "first.log", Path(temp) / "second.log"
            with terminal_log(first):
                logger = retained_logger()
                logger.warning("first-task")
            first_text = first.read_text(encoding="utf-8")
            logger.warning("between-tasks")
            logger.handlers[0].flush()
            with terminal_log(second):
                logger.warning("second-task")
                print("normal-print")
            logger.warning("after-tasks")
            self.assertEqual(first.read_text(encoding="utf-8"), first_text)
            self.assertIn("second-task", second.read_text(encoding="utf-8"))
            self.assertIn("normal-print", second.read_text(encoding="utf-8"))
            self.assertNotIn("between-tasks", second.read_text(encoding="utf-8"))
            self.assertNotIn("after-tasks", second.read_text(encoding="utf-8"))
            for text in ("first-task", "between-tasks", "second-task", "after-tasks"):
                self.assertEqual(console.getvalue().count(text), 1)

    def test_exception_restores_streams_and_retained_handler_can_flush(self):
        with TemporaryDirectory() as temp, patch("sys.stdout", new=StringIO()) as out, patch("sys.stderr", new=StringIO()) as err:
            path = Path(temp) / "failed.log"
            with self.assertRaisesRegex(RuntimeError, "intentional-test"), terminal_log(path):
                logger = retained_logger()
                raise RuntimeError("intentional-test")
            self.assertIs(sys.stdout, out)
            self.assertIs(sys.stderr, err)
            self.assertIn("intentional-test", path.read_text(encoding="utf-8"))
            logger.warning("after-exception")
            logger.handlers[0].flush()
            self.assertIn("after-exception", err.getvalue())

    def test_nested_logs_restore_outer_destination(self):
        with TemporaryDirectory() as temp, patch("sys.stdout", new=StringIO()), patch("sys.stderr", new=StringIO()):
            outer, inner = Path(temp) / "outer.log", Path(temp) / "inner.log"
            with terminal_log(outer):
                with terminal_log(inner):
                    logger = retained_logger()
                    logger.warning("inside-inner")
                logger.warning("back-to-outer")
            self.assertEqual(outer.read_text(encoding="utf-8").count("inside-inner"), 1)
            self.assertEqual(outer.read_text(encoding="utf-8").count("back-to-outer"), 1)
            self.assertNotIn("back-to-outer", inner.read_text(encoding="utf-8"))

    @skipUnless("fork" in multiprocessing.get_all_start_methods(), "Linux fork-worker regression")
    def test_fork_worker_with_handler_retained_from_previous_task(self):
        with TemporaryDirectory() as temp, patch("sys.stdout", new=StringIO()), patch("sys.stderr", new=StringIO()):
            first, second = Path(temp) / "first.log", Path(temp) / "second.log"
            with terminal_log(first):
                logger = retained_logger()
                logger.warning("parent-first")
            context = multiprocessing.get_context("fork")
            receiver, sender = context.Pipe(duplex=False)
            with terminal_log(second):
                process = context.Process(target=emit_child, args=(logger, sender))
                process.start()
                sender.close()
                process.join(timeout=15)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    self.fail("Forked logging worker did not exit")
                self.assertEqual(process.exitcode, 0)
                self.assertTrue(receiver.poll(1))
                self.assertEqual(receiver.recv(), "ok")
                receiver.close()
            self.assertIn("from-child", second.read_text(encoding="utf-8"))
            self.assertNotIn("from-child", first.read_text(encoding="utf-8"))
