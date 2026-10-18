import json

from experiments.observer import analyze


def write_run(root, arm, seed, m1, refl=0.48, completed=100):
    d = root / f"{arm}-s{seed}"
    d.mkdir()
    comps = {"lists": m1, "question": m1, "length": m1}
    summary = {
        "arm": arm, "seed": seed, "completed": completed,
        "endpoints": {**comps, "M1": m1, "n": {}},
        "memory_stats": {"total": 100, "by_type": {"reflective": int(refl * 100), "episodic": 36}},
        "final_directives": [{"content": "x", "origin": "system"}],
        "crucial_after_40": 0, "important_to_note_after_42": 0, "seconds": 600,
    }
    (d / "summary.json").write_text(json.dumps(summary))


def test_primary_comparison_and_incomplete_runs_are_reported(tmp_path):
    for s in range(1, 9):
        write_run(tmp_path, "SELF_PLUS", s, 0.40)
        write_run(tmp_path, "PEER", s, 0.30)
    (tmp_path / "PEER-s9").mkdir()                       # crashed run
    write_run(tmp_path, "SELF", 1, 0.5, completed=60)    # partial run
    result = analyze.analyze(str(tmp_path))
    assert result["runs"] == 16
    assert sorted(result["incomplete"]) == ["PEER-s9", "SELF-s1"]
    q1 = result["comparisons"]["Q1"]
    assert q1["verdict"] == "PASS" and q1["wins"] == 8 and q1["seeds"] == list(range(1, 9))
    assert result["comparisons"]["Q2"]["n_pairs"] == 0


def test_q5_prediction(tmp_path):
    for s in (1, 2):
        write_run(tmp_path, "SELF", s, 0.4, refl=0.48)
        write_run(tmp_path, "PEER", s, 0.4, refl=0.47)
    assert analyze.analyze(str(tmp_path))["Q5"]["holds"]
    write_run(tmp_path, "OTHER", 1, 0.4, refl=0.30)
    assert not analyze.analyze(str(tmp_path))["Q5"]["holds"]


def test_report_prints(tmp_path, capsys):
    write_run(tmp_path, "SELF_PLUS", 1, 0.4)
    write_run(tmp_path, "PEER", 1, 0.3)
    analyze.main([str(tmp_path)])
    out = capsys.readouterr().out
    assert "Q1 SELF_PLUS - PEER" in out and (tmp_path / "analysis.json").exists()
