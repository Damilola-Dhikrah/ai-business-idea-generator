AI Business Idea Generator

A simple Streamlit app: enter a niche/industry and it generates 3 concise startup ideas — each with a Pitch, Target Audience, Revenue Model, and Execution Steps (plus optional Validation).

Runs locally with Ollama (free) and can also use OpenAI if you add an API key.



Features

Three ideas in one shot using a strict JSON schema (stable output)



Clean layout with clear spacing and bullets



Optional settings: tone, include validation, output length (Short / Normal / Long)



Local by default (llama3.2:3b) — no API costs



TXT download of the results



Lenient JSON parsing \& sanitizing (prevents weird line breaks/slashes)



Demo (Screenshots \& Sample Output)

Screenshot:see example/screenshots()



Sample output: see examples/sample\_output.txt



How to Run Locally:



1\) Clone \& enter

bash

Copy

Edit

git clone https://github.com/<Damilola-Dhikrah>/ai-business-idea-generator.git

cd ai-business-idea-generator



2\) Python virtual environment

Windows (PowerShell):



bash

Copy

Edit

python -m venv .venv

.\\.venv\\Scripts\\Activate

macOS/Linux:



bash

Copy

Edit

python3 -m venv .venv

source .venv/bin/activate

3\) Install dependencies

bash

Copy

Edit

pip install -r requirements.txt

4\) (Optional) OpenAI setup

Create a .env file (do not commit this file):



ini

Copy

Edit

OPENAI\_API\_KEY=sk-...

OPENAI\_MODEL=gpt-4o-mini

5\) Install Ollama \& pull a local model (free)

Download: https://ollama.com/download



First run downloads the model (a few GB):



bash

Copy

Edit

ollama run llama3.2:3b     # then press Ctrl+C to exit once you see the prompt



6\) Run the app

bash

Copy

Edit

streamlit run app.py

Open http://localhost:8501 in your browser.



Usage

Enter a Niche / Industry (and optional Target and Constraints)



Click Generate Ideas 🚀



Use Optional settings to toggle Validation or adjust Output length



Click Download as TXT to save the results



Tip: For the smoothest first run, keep a warm model window open:



bash

Copy

Edit

ollama run llama3.2:3b

\# type: hello

\# leave this window open while using the app

Project Structure

bash

Copy

Edit

app.py            # Streamlit app (JSON prompt, local/OpenAI provider, rendering)

requirements.txt  # Python deps

README.md         # This file

.gitignore        # Keeps secrets/venv out of git

examples/

&nbsp; ├─ screenshot.png

&nbsp; └─ sample\_output.txt

Requirements

shell

Copy

Edit

streamlit

python-dotenv

requests

openai>=1.40.0





How It Works (Key Logic):



Builds a strict JSON-only prompt asking for 3 ideas.



Calls either local Ollama (default) or OpenAI (if .env provided).



Parses JSON with a lenient fallback and sanitizes fields to remove stray line breaks/slashes.



Renders ideas with clear spacing and provides a TXT download.



Troubleshooting:

Slow first response: Warm the model: ollama run llama3.2:3b → type hello and leave it open.



Timeouts: Set Output length to Normal (~1100) or Short (~700).



Parse error: Click Generate again, or switch to Short (~700). The lenient parser also auto-fixes common issues.



Using OpenAI: Ensure billing is enabled; otherwise you’ll see a 429 quota error.



Privacy \& Safety

No analytics, no uploads sent to external services when using Ollama (runs on your machine).



Keep your .env out of git (see .gitignore below).



.gitignore (already included)

markdown

Copy

Edit

.env

.venv/

\_\_pycache\_\_/

\*.pyc

.streamlit/

examples/\*.tmp



Meets the Assignment Criteria:



Functionality: Generates 3 ideas with one-paragraph pitch, target audience, revenue model, and execution steps (+ optional validation).



Creativity: Validation toggle, stable JSON approach, local/offline LLM, TXT export.



Code quality: Small, readable file with docstring and comments; clean repo.



Use of AI tools: Ollama (local) by default; OpenAI optional.



User experience: One-screen app, clear spacing, progress spinner, simple “Optional settings,” download button.





Credits

Built by <Sikirat Mustapha>.

