"""Neo is retired: every reasoning run says so once, in the form its caller reads."""

import io
import json

from neo.cli import RETIREMENT_NOTICE, announce_retirement


def test_a_human_run_gets_one_plain_line():
    stream = io.StringIO()
    announce_retirement(json_mode=False, stream=stream)
    assert stream.getvalue() == RETIREMENT_NOTICE + "\n"


def test_a_json_run_gets_a_parseable_notice_event():
    stream = io.StringIO()
    announce_retirement(json_mode=True, stream=stream)
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event == {"type": "notice", "message": RETIREMENT_NOTICE}
