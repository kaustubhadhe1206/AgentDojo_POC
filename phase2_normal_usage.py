"""Phase 2: send one prompt to NVIDIA NIM through the plain OpenAI client (no AgentDojo yet),
so a later failure can be blamed on the AgentDojo wiring rather than the key or endpoint."""

import argparse
import time

import config
import tracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: sanity-check the LLM API")
    parser.add_argument("--model", default=config.SMALL_MODEL)
    args = parser.parse_args()

    prompt = "Say hi"
    start, reply, error = time.time(), "", ""
    try:
        r = config.make_client().chat.completions.create(model=args.model, messages=[{"role": "user", "content": prompt}])
        reply = r.choices[0].message.content or ""
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    if not error:
        print(reply)
        print(f"Phase 2 OK ({args.model})")
    tracker.record("normal", {"Model": args.model.rsplit("/", 1)[-1], "Prompt": prompt, "Response": reply,
                              "Latency (s)": round(time.time() - start, 1), "Error": error})
    if error:
        raise SystemExit(error)


if __name__ == "__main__":
    main()
