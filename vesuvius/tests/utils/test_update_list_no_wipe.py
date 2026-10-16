"""update_list must not overwrite the packaged catalog with a failed or empty scrape."""

import pytest

for _module in ("aiohttp", "lxml"):
    pytest.importorskip(_module)

from vesuvius.data.paths import parser  # noqa: E402
from vesuvius.utils import catalog  # noqa: E402

PAGES = {
    "https://d/": '<a href="scrolls/">s</a>',
    "https://d/scrolls/": '<a href="1/">1</a>',
    "https://d/scrolls/1/": '<a href="volumes/">v</a>',
    "https://d/scrolls/1/volumes/": '<a href="54keV_7.91um.zarr/">z</a>',
    "https://c/": '<a href="cube_a/">c</a>',
}


def _fake_fetch(mode):
    async def fetch(session, url, failures=None):
        if mode == "fail" and failures is not None:
            failures.append(url)
        return PAGES.get(url, "<html></html>") if mode == "ok" else ""

    return fetch


@pytest.mark.parametrize("mode", ["fail", "empty"])
def test_failed_or_empty_scrape_keeps_files(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(catalog, "_config_dir", lambda: tmp_path)
    monkeypatch.setattr(parser, "fetch", _fake_fetch(mode))
    for name in ("scrolls.yaml", "cubes.yaml", "directory_structure.yaml"):
        (tmp_path / name).write_text("old: 1\n")

    with pytest.raises(RuntimeError, match="left unchanged"):
        catalog.update_list("https://d/", "https://c/")

    assert {p.read_text() for p in tmp_path.iterdir()} == {"old: 1\n"}


def test_successful_scrape_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "_config_dir", lambda: tmp_path)
    monkeypatch.setattr(parser, "fetch", _fake_fetch("ok"))

    catalog.update_list("https://d/", "https://c/")

    volume = "https://d/scrolls/1/volumes/54keV_7.91um.zarr/"
    assert catalog.list_files() == {"1": {"54": {"7.91": {"volume": volume, "segments": {}}}}}
    assert catalog.list_cubes() == {1: {54: {7.91: {"cube_a": "https://c/cube_a/"}}}}
