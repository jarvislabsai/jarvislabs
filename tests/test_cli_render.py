from __future__ import annotations

from types import SimpleNamespace

from jarvislabs.cli import render


def test_service_url_rows_maps_builtin_and_custom_ports():
    inst = SimpleNamespace(
        url="https://abc123.notebooksn.jarvislabs.net/lab?token=test",
        vs_url="https://abc1230.notebooksn.jarvislabs.net",
        endpoints=[
            "https://abc1231.notebooksn.jarvislabs.net",
            "https://abc1232.notebooksn.jarvislabs.net",
            "https://abc1233.notebooksn.jarvislabs.net",
        ],
        http_ports="7860,8080",
    )

    rows = render._service_url_rows(inst)

    assert [field for field, _ in rows] == [
        "URL",
        "Port 7007",
        "Port 6006",
        "Port 7860",
        "Port 8080",
    ]
    assert "abc123.notebooksn.jarvislabs.net" in rows[0][1]
    assert "abc1230.notebooksn.jarvislabs.net" in rows[1][1]
    assert "abc1231.notebooksn.jarvislabs.net" in rows[2][1]
    assert "abc1232.notebooksn.jarvislabs.net" in rows[3][1]
    assert "abc1233.notebooksn.jarvislabs.net" in rows[4][1]
