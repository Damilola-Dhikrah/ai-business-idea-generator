"""
AI Business Idea Generator (Streamlit)
- Local LLM via Ollama by default (llama3.2:3b) to avoid API costs/timeouts
- Generates 3 ideas in a single call using a strict JSON schema
- Clean, readable UI with spacing + TXT download

Key logic:
1) Strict JSON prompt: 3 ideas only, single-line fields, pricing hint, no slashes/newlines.
2) Local Ollama by default (fast + free); OpenAI optional via .env.
3) Parse JSON (fallback regex if wrapped) and sanitize fields before rendering.
"""

import os
import json
import re
import requests
import streamlit as st
from dotenv import load_dotenv

# ---------- Safe defaults to avoid timeouts ----------
DEFAULT_TOKENS = 1100       # good default for local 3B models
SAFE_MODEL = "llama3.2:3b"  # small, fast, decent quality

# ---------- Load .env for optional OpenAI usage ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # used only if provider = OpenAI

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Business Idea Generator", page_icon="💡", layout="centered")
st.title("💡 AI Business Idea Generator")

st.write(
    "Enter a **niche/industry** (e.g., “fitness for busy moms”, “African grocery e-commerce”, "
    "“B2B cybersecurity tooling”) and I’ll generate 3 concise startup ideas."
)

niche = st.text_input("Niche / Industry")
target = st.text_input("Target region/audience (optional)", placeholder="e.g., US market, Nigeria, Gen Z creators")
constraints = st.text_area("Constraints (optional)", placeholder="e.g., <$5k startup budget, no coding, must be remote-friendly")

with st.expander("Optional settings"):
    tone = st.selectbox("Tone", ["practical", "innovative", "scrappy"], index=0)
    include_validation = st.checkbox("Add quick validation steps", value=True)

    # Keep it simple: local model by default
    provider = st.selectbox("Model provider", ["Local - Ollama (free)", "OpenAI (requires credits)"], index=0)
    ollama_model = st.text_input("Ollama model name", SAFE_MODEL)

    # Friendly length control (capped to avoid timeouts)
    length_choice = st.selectbox(
        "Output length",
        ["Short (~700 tokens)", "Normal (~1100)", "Long (~1500)"],
        index=1
    )
    length_map = {"Short (~700 tokens)": 700, "Normal (~1100)": 1100, "Long (~1500)": 1500}
    max_tokens = length_map[length_choice]

def parse_json_lenient(raw: str):
    """
    Try strict JSON first. If that fails, extract the first [...] block,
    normalize quotes, and remove trailing commas before } or ].
    """
    try:
        return json.loads(raw)
    except Exception:
        pass

    # pull first JSON array
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        return None
    j = m.group(0)

    # normalize quotes / remove code fences
    j = j.replace("“", '"').replace("”", '"').replace("’", "'").replace("```", "")

    # remove trailing commas like ", ]" or ", }"
    j = re.sub(r",\s*([}\]])", r"\1", j)

    # collapse weird whitespace
    j = re.sub(r"\s+\n", "\n", j)

    try:
        return json.loads(j)
    except Exception:
        return None


# ---------- Provider calls ----------
def call_openai(system: str, user: str) -> str:
    """Calls OpenAI Chat Completions. Requires OPENAI_API_KEY."""
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in .env (or choose Local - Ollama).")
    from openai import OpenAI  # lazy import so local runs without openai installed
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.4,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content

def call_ollama(system: str, user: str, model: str = SAFE_MODEL, num_predict: int = DEFAULT_TOKENS) -> str:
    """
    Core LLM call to local Ollama server.
    - num_predict controls output length; tuned to avoid timeouts on small local models.
    - If first call is slow, 'warm' the model by running `ollama run llama3.2:3b` and saying 'hello' in another window.
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {
            "num_predict": int(num_predict),  # how much it can write
            "num_ctx": 8192,                  # prompt+output budget
            "temperature": 0.4
        },
        "keep_alive": "30m"                  # keep model warm between calls
    }
    r = requests.post(url, json=payload, timeout=600)  # 10 min cap is plenty for local 3B
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "").strip()

# ---------- Sanitizer (avoid stacked letters / ugly breaks) ----------
def one_line(s: str) -> str:
    """Collapse any linebreaks/extra spaces to a single clean line."""
    s = re.sub(r'[\r\n]+', ' ', str(s or ''))
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ---------- One-shot generator (3 ideas in JSON) ----------
def generate_three_ideas(
    provider: str,
    system: str,
    niche: str,
    target: str,
    constraints: str,
    tone: str,
    include_validation: bool,
    model: str,
    max_tokens: int
):
    """
    Ask the model for 3 ideas at once and return parsed JSON (list of dicts).
    We request a strict JSON array to make rendering deterministic.
    """
    comma = "," if include_validation else ""
    validation_schema = (
        '"Validation": ["fast test + success metric \u2264 12 words", '
        '"fast test + success metric \u2264 12 words", '
        '"fast test + success metric \u2264 12 words"]'
        if include_validation else ""
    )

    prompt = f"""
Generate exactly THREE startup ideas for the niche/industry below.

Output format: return ONLY a valid JSON array (no prose, no markdown). Each item must have:
[
  {{
    "Name": "short, brandable, unique",
    "Pitch": "<= 55 words; What it is. Why now: <driver>. Edge: <differentiator>.",
    "TargetAudience": "one line (no line breaks)",
    "RevenueModel": "one line with pricing example (e.g., $19 per month, 10% per transaction, $5 per order). Use '$', '%', or the word 'per'. Do NOT use slashes.",
    "Execution": ["action step + channel + metric <= 12 words",
                  "action step + channel + metric <= 12 words",
                  "action step + channel + metric <= 12 words"]{"," if include_validation else ""}
    {"\"Validation\": [\"fast test + success metric <= 12 words\", \"fast test + success metric <= 12 words\", \"fast test + success metric <= 12 words\"]" if include_validation else ""}
  }},
  {{ ... }},
  {{ ... }}
]

Rules:
- Use ONLY plain ASCII characters and straight double quotes.
- Names must be DISTINCT across the 3 ideas; vary segment/channel/revenue/product type.
- Keep language simple, specific, and practical—no buzzword salad.
- Do NOT include any text before or after the JSON.
- Do NOT include escaped newlines (\\n) or slashes (/) in single-line fields.
- Do NOT include trailing commas and do NOT include example ellipses ("...") in output.

Niche/Industry: {niche}
Target: {target or 'not specified'}
Constraints: {constraints or 'none'}
Tone: {tone}
"""
    raw = (
    	call_ollama(system, prompt, model=model, num_predict=max_tokens)
 	 if provider.startswith("Local")
    	else call_openai(system, prompt)
    )

    ideas = parse_json_lenient(raw)
    return ideas

# ---------- Button handler ----------
if st.button("Generate Ideas 🚀"):
    if not niche.strip():
        st.warning("Please enter a niche/industry.")
        st.stop()

    # Higher-quality system message (acts like an expert)
    system = (
        "You are an expert creative business operator and market strategist. "
        "Prioritize ideas that are feasible, testable within a week, and clearly differentiated. "
        "Be concrete and region-aware. Use US English, plain language, no emojis."
    )

    try:
        with st.spinner("Generating 3 ideas..."):
            ideas = generate_three_ideas(
                provider, system, niche, target, constraints, tone,
                include_validation, ollama_model, max_tokens
            )

        if not ideas or not isinstance(ideas, list):
            st.error("I couldn't parse the ideas. Try again or choose Short/Normal length.")
            st.stop()

        # Optional: warn if names are duped
        names = [str(x.get("Name", "")).lower().strip() for x in ideas]
        if len(set(names)) < len(names):
            st.info("Detected duplicate names; re-run once for variety or change Output length to Normal.")

        # Pretty print with clean spacing + build a download blob
        full_text_blocks = []
        for i, idea in enumerate(ideas, start=1):
            name = one_line(idea.get("Name", f"Idea {i}"))
            pitch = one_line(idea.get("Pitch", ""))
            ta   = one_line(idea.get("TargetAudience", ""))
            rev  = one_line(idea.get("RevenueModel", "")).replace('/', ' per ')
            steps = [one_line(s) for s in (idea.get("Execution", []) or [])]
            val   = [one_line(v) for v in (idea.get("Validation", []) or [])] if include_validation else []

            md = f"""### Startup Idea {i}

**Name:** {name}

**Pitch:** {pitch}

**Target Audience:** {ta}

**Revenue Model:** {rev}

**Execution Steps:**
{chr(10).join(f"- {s}" for s in steps)}
"""
            if include_validation and val:
                md += f"""\n**Validation:**\n{chr(10).join(f"- {v}" for v in val)}\n"""

            st.markdown(md)
            st.divider()

            # Prepare text block for download
            block_lines = [
                f"Startup Idea {i}:",
                f"Name: {name}",
                f"Pitch: {pitch}",
                f"Target Audience: {ta}",
                f"Revenue Model: {rev}",
                "Execution Steps:",
            ] + [f"- {s}" for s in steps]
            if include_validation and val:
                block_lines += ["Validation:"] + [f"- {v}" for v in val]
            full_text_blocks.append("\n".join(block_lines))

        download_text = "\n\n---\n\n".join(full_text_blocks)

        st.download_button(
            label="Download as TXT",
            data=download_text,
            file_name="business_ideas.txt",
            mime="text/plain",
        )

        st.success("Done!")

    except Exception as e:
        st.error(f"Error: {e}")
        st.info(
            "Tip: keep Output length = Normal (~1100) to avoid timeouts on local models. "
            "If the first local call is slow, warm the model with `ollama run llama3.2:3b` and say 'hello'."
        )
