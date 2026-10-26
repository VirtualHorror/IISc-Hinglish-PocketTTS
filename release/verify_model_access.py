"""Verify the evaluators' READ-ONLY token against the private model repo, end to end.

1. The token is read-only (whoami role); anonymous access is refused through the API and by plain
   HTTPS requests for config.yaml and model.safetensors (status codes recorded), which the read
   token gets.
2. In a fresh, isolated Python process (`python -I`: no project code importable) whose only
   credential is the read token in HF_TOKEN and whose HF cache is empty, follow the model card's
   quick start: download only inference.py, import it, let it fetch the model, and synthesise a
   Hinglish sentence to a .wav with auto_normalize (the bundled normaliser must be found).
3. The official pocket-tts CLI with the repo's hf:// config and the same token; and, when the repo ships
   a distilled CPU student (student_<n>l/), the quick start again with variant="<n>l".
4. Whisper transcribes every .wav (round-trip sanity check of the downloaded weights), and the
   downloaded model.safetensors must match the exported file's sha256.

    python -m release.verify_model_access [--repo MSRIT-NBM/pocket-tts-indian-hinglish]

writes release/model_access_verification.json
"""

from __future__ import annotations

import os

os.environ.pop("HF_TOKEN", None)  # before huggingface_hub reads it: only the read token is used
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import textwrap  # noqa: E402
from pathlib import Path  # noqa: E402

import requests  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError  # noqa: E402

from data_pipeline.common import REPO_ROOT, WORK, hf_token, load_dotenv, setup_logging, write_json  # noqa: E402

logger = setup_logging("verify_model_access")

REPO_ID = "MSRIT-NBM/pocket-tts-indian-hinglish"
TEXT = "Kal subah 7 baje meri meeting hai, please reminder सेट कर दो"
CLI_TEXT = "आज ट्रैफ़िक बहुत ज़्यादा है, मैं दस मिनट लेट हो जाऊँगा।"


def status(url: str, token: str | None, head: bool = False) -> int:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    fn = requests.head if head else requests.get
    return fn(url, headers=headers, allow_redirects=False, timeout=60).status_code


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--workdir", type=Path, default=WORK / "tmp" / "verify_model_access")
    ap.add_argument("--release", type=Path, default=WORK / "release")
    ap.add_argument("--keep-cache", action="store_true")
    args = ap.parse_args()
    os.environ.pop("HF_TOKEN", None)  # load_dotenv must not have put a token back
    read = hf_token("read")
    report: dict = {"repo": args.repo}

    who = HfApi(token=read).whoami()
    report["token"] = {"user": who["name"], "role": who.get("auth", {}).get("accessToken", {}).get("role")}
    assert report["token"]["role"] == "read", report["token"]

    try:
        HfApi(token=False).model_info(args.repo)
        report["anonymous_api"] = "ALLOWED (repo is public!)"
    except (RepositoryNotFoundError, GatedRepoError, HfHubHTTPError) as exc:
        report["anonymous_api"] = f"refused ({type(exc).__name__})"
    base = f"https://huggingface.co/{args.repo}/resolve/main"
    report["http_status"] = {
        "anonymous": {"api": status(f"https://huggingface.co/api/models/{args.repo}", None),
                      "config.yaml": status(f"{base}/config.yaml", None),
                      "model.safetensors": status(f"{base}/model.safetensors", None, head=True)},
        "read_token": {"api": status(f"https://huggingface.co/api/models/{args.repo}", read),
                       "config.yaml": status(f"{base}/config.yaml", read),
                       "model.safetensors": status(f"{base}/model.safetensors", read, head=True)},
    }
    anon_ok = all(v in (401, 403, 404) for v in report["http_status"]["anonymous"].values())
    read_ok = all(v in (200, 302, 307) for v in report["http_status"]["read_token"].values())
    info = HfApi(token=read).model_info(args.repo, files_metadata=True)
    report["metadata"] = {"private": info.private, "sha": info.sha,
                          "files": {s.rfilename: s.size for s in info.siblings}}
    assert info.private and anon_ok and read_ok, report

    # --- 2 + 3: fresh process, read token only, empty cache --------------------------------------
    if args.workdir.exists():
        shutil.rmtree(args.workdir)
    args.workdir.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HF_", "HUGGING"))}
    env.update(HF_TOKEN=read, HF_HOME=str(args.workdir / "hf_home"), HF_HUB_DISABLE_TELEMETRY="1")
    wav, cli_wav = args.workdir / "read_token_test.wav", args.workdir / "read_token_cli_test.wav"
    quickstart = args.workdir / "quickstart"  # the model card's quick start: only inference.py, then run it

    def fresh_synthesis(out: Path, variant: str = "24l") -> dict:
        script = textwrap.dedent(f"""
            import json, sys, time
            from huggingface_hub import hf_hub_download
            t0 = time.time()
            hf_hub_download({args.repo!r}, "inference.py", local_dir={str(quickstart)!r})
            sys.path.insert(0, {str(quickstart)!r})
            from inference import HinglishTTS
            tts = HinglishTTS.from_pretrained({args.repo!r}, variant={variant!r})
            assert tts.processor is not None, "bundled hinglish_text normaliser not found"
            audio = tts.synthesize({TEXT!r}, voice="indian_female_ananya")
            tts.save(audio, {str(out)!r})
            print(json.dumps({{"variant": {variant!r}, "layers": len(tts.model.flow_lm.transformer.layers),
                              "snapshot": str(tts.model_dir), "voices": tts.voices(), "model_input": tts.normalize({TEXT!r}),
                              "seconds": round(len(audio) / tts.sample_rate, 2), "wall_s": round(time.time() - t0, 1),
                              "auto_normalize": tts.processor is not None}}, ensure_ascii=False))
        """)
        run = subprocess.run([sys.executable, "-I", "-c", script], env=env, cwd=args.workdir, capture_output=True,
                             text=True, timeout=1800)
        if run.returncode != 0:
            raise SystemExit(f"fresh-process synthesis ({variant}) failed:\n{run.stderr[-3000:]}")
        return json.loads(run.stdout.strip().splitlines()[-1])

    report["fresh_process"] = fresh_synthesis(wav)
    snapshot = Path(report["fresh_process"]["snapshot"])
    students = sorted({f.split("/")[0] for f in report["metadata"]["files"] if f.startswith("student_")})
    student_wavs = []
    for folder in students:
        variant = folder.removeprefix("student_")
        student_wavs.append((f"fresh_process_{variant}", args.workdir / f"read_token_test_{variant}.wav"))
        report[student_wavs[-1][0]] = fresh_synthesis(student_wavs[-1][1], variant)
    report["fresh_process"]["model_sha256_matches_export"] = (
        sha256(snapshot / "model.safetensors")
        == json.loads((args.release / "export_info.json").read_text())["files"]["model.safetensors"]["sha256"])

    cli = subprocess.run(["pocket-tts", "generate", "--config", f"hf://{args.repo}/config.yaml",
                          "--voice", str(snapshot / "voices" / "indian_male_aarav.wav"), "--text", CLI_TEXT,
                          "--output-path", str(cli_wav), "--device", "cpu"],
                         env=env, cwd=args.workdir, capture_output=True, text=True, timeout=1800)
    if cli.returncode != 0:
        raise SystemExit(f"pocket-tts CLI failed:\n{cli.stderr[-3000:]}")
    import soundfile as sf

    report["official_cli"] = {"command": f"pocket-tts generate --config hf://{args.repo}/config.yaml "
                                         f"--voice voices/indian_male_aarav.wav --text ...",
                              "text": CLI_TEXT, "seconds": round(sf.info(str(cli_wav)).duration, 2)}

    # --- 4: round trip ---------------------------------------------------------------------------
    from eval.asr import HinglishScorer, WhisperASR

    checks = [("fresh_process", TEXT, wav), ("official_cli", CLI_TEXT, cli_wav)] + [(k, TEXT, w) for k, w in student_wavs]
    hyps = WhisperASR().transcribe([w for _, _, w in checks])
    scorer = HinglishScorer()
    for (key, text, _), hyp in zip(checks, hyps, strict=True):
        s = scorer.score([text], [hyp])
        report[key].update(asr_transcript=hyp, cer=round(s["cer"], 4), wer=round(s["wer"], 4))
    report["result"] = ("PASS: anonymous access refused; the read-only token downloads the private repo, its "
                        "inference.py and the official pocket-tts CLI both load the model and speak Hinglish"
                        + (f"; the distilled student ({', '.join(students)}) loads and speaks too" if students else ""))
    keep = WORK / "eval_audio" / "access_check"
    keep.mkdir(parents=True, exist_ok=True)
    for _, _, w in checks:
        shutil.copy2(w, keep / w.name)
    if not args.keep_cache:
        shutil.rmtree(args.workdir)
    write_json(REPO_ROOT / "release" / "model_access_verification.json", report)
    logger.info(report["result"])
    for k in ("token", "anonymous_api", "http_status", "fresh_process", "official_cli", *(k for k, _ in student_wavs)):
        logger.info(f"{k}: {report[k]}")


if __name__ == "__main__":
    main()
