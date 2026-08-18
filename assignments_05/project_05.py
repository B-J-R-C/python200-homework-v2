# ==========================================
# Week 5 Mini-Project: project_05.py
# Job Application Helper
# ==========================================
# An AI job-application coach for career changers. It rewrites resume bullet
# points, drafts cover letter openings, and answers follow-up questions in a
# single conversation with memory -- with a moderation check on every input.

import json

from dotenv import load_dotenv
from openai import OpenAI

# ==========================================
# Task 1: Setup and System Prompt
# ==========================================

load_dotenv()
client = OpenAI()


def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    """Send a messages list to the model and return just the text reply."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400,
    )
    return response.choices[0].message.content


SYSTEM_PROMPT = """You are a job application coach who works specifically with career
changers -- people moving into a new field who have real, relevant experience but
struggle to describe it in the language their target industry uses.

How you help:
- You rewrite resume bullet points to be specific, results-oriented, and led by strong
  action verbs.
- You draft cover letter openings that connect the person's previous field to the role
  they are applying for.
- You answer questions about resumes, cover letters, job descriptions, and interview
  preparation.

Rules you always follow:
1. Stay on job application materials. If the user asks about something else, answer
   briefly if it is harmless, then steer back to their application.
2. Never invent facts. Do not add metrics, job titles, tools, degrees, or dates the
   user did not give you. If a bullet would be stronger with a number, ask the user
   for the number instead of making one up.
3. Close any drafted material by reminding the user to read it over and edit it before
   sending it anywhere. It is a draft, not a finished document.
4. Be explicit that you do not know the hiring norms of their specific industry,
   region, or company, and that their own judgment beats yours on tone and convention.
5. Be direct and concrete. Skip the filler encouragement -- useful specifics are more
   respectful of their time.
"""

# TASK 1 COMMENT: one deliberate choice in the system prompt, and why.
#
# The choice I care about most is rule 2 -- the explicit ban on inventing facts,
# combined with the instruction to *ask for a number* rather than supply one. My first
# draft of this prompt just said "make bullets results-oriented," and the model
# cheerfully turned "Helped customers with their problems" into "Resolved 200+ customer
# issues per month with a 95% satisfaction rating." Those numbers came from nowhere.
# For a tool whose output goes onto a real resume, a fabricated metric isn't a quality
# problem, it's something that can cost someone a job offer when it comes up in an
# interview. Making the constraint explicit -- and giving the model an alternative
# behaviour to fall back on -- cut it down substantially.
#
# I also deliberately narrowed the audience to career changers rather than job seekers
# in general. A specific brief produces more predictable behaviour: the model
# consistently reaches for transferable-skill framing instead of generic corporate
# advice.


# ==========================================
# Task 2: Bullet Point Rewriter
# ==========================================


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` fences the model sometimes adds around JSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]                      # drop the opening ``` or ```json
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]                 # drop the closing ```
        cleaned = "\n".join(lines).strip()
    return cleaned


def rewrite_bullets(bullets: list[str]) -> list[dict]:
    """Rewrite resume bullets and print the original next to the improved version."""
    if not bullets:
        print("No bullet points to rewrite.")
        return []

    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""You are a professional resume coach helping a career changer.

Rewrite each resume bullet point below so that it is more specific, results-oriented,
and compelling. Lead with a strong action verb. Name the skill or the outcome, not just
the activity.

Hard constraint: do not invent facts. Do not add numbers, percentages, team sizes,
tools, or job titles that are not stated or clearly implied by the original bullet. If
a bullet is vague, make the language stronger without fabricating detail.

Return ONLY a valid JSON list of objects. Each object must have exactly two keys:
  "original" - the bullet exactly as it was given to you
  "improved" - your rewritten version

Respond ONLY with valid JSON, no other text, and no markdown code fences.

Bullet points:
```
{bullet_text}
```"""

    messages = [{"role": "user", "content": prompt}]
    raw_response = get_completion(messages, temperature=0.5)

    try:
        parsed_bullets = json.loads(_strip_code_fences(raw_response))
    except json.JSONDecodeError as err:
        print("\nThe model did not return valid JSON, so I couldn't parse it.")
        print("Error:", err)
        print("Raw response:\n", raw_response)
        return []

    print("\n--- Bullet Point Revisions ---")
    for idx, item in enumerate(parsed_bullets, start=1):
        print(f"\nBullet {idx}:")
        print(f"  Original: {item.get('original')}")
        print(f"  Improved: {item.get('improved')}")
    print("\nThese are drafts -- read them over and make sure every claim is one you")
    print("can back up in an interview before putting them on your resume.\n")

    return parsed_bullets


# TASK 2 COMMENT: What makes these bullets weak, and what did the model change?
#
# The starter bullets are weak for three reasons. They lead with soft, low-information
# verbs ("Helped", "Made", "Worked with"), which describe participation rather than
# ownership. They name an activity but no outcome -- "made reports" tells a hiring
# manager nothing about whether the reports were used or what they were used for. And
# they're generic enough to belong to almost any job, which means they don't
# differentiate the candidate at all.
#
# The model's rewrites did two useful things: it swapped in verbs that imply ownership
# ("Resolved", "Produced", "Collaborated to deliver") and it made the *purpose*
# explicit, e.g. reports became reporting that supported management decisions. What it
# could not do -- correctly -- is supply scale. Once I added the no-inventing-facts
# constraint, the improved bullets stopped containing fake percentages, which means
# they're still missing numbers. That's the honest limit of the tool: the user has to
# supply "how many" and "how much." A good next iteration would have the bot ask a
# follow-up question for each vague bullet instead of just rewriting it.


# ==========================================
# Task 3: Cover Letter Generator
# ==========================================


def generate_cover_letter(job_title: str, background: str) -> str:
    """Generate a cover letter opening paragraph using few-shot prompting."""
    prompt = f"""You write strong cover letter opening paragraphs for career changers.
The paragraph should be 3-5 sentences: confident, specific, and free of clichés. Do not
claim any experience, credential, or skill that is not in the background provided.

Here are two examples of the style and tone you should match:

Example 1:
Role: Data Analyst at a healthcare nonprofit
Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
Opening: After seven years as a registered nurse, I've spent my career making decisions
under pressure using incomplete information -- which turns out to be excellent training for
data analysis. I recently completed a data analytics program where I built dashboards
tracking patient outcomes across departments. I'm excited to bring that combination of
clinical context and technical skill to [Company]'s mission-driven work.

Example 2:
Role: Junior Software Engineer at a fintech startup
Background: Ten years in retail banking operations, self-taught Python developer for two years.
Opening: I spent a decade on the operations side of banking, watching technology decisions
get made by people who had never processed a wire transfer or resolved a failed ACH batch.
That frustration turned into curiosity, and two years of self-teaching Python later, I'm
ready to be on the other side of those decisions. I'm applying to [Company] because your
work on payment infrastructure is exactly where my domain expertise and new technical skills
intersect.

Now write an opening paragraph for this person:
Role: {job_title}
Background: {background}
Opening:"""

    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages, temperature=0.7)


# TASK 3 COMMENT: Why those examples, and what does few-shot control here?
#
# I picked those two because they share a structure I want copied, but come from
# different fields so the model generalizes the structure instead of the subject matter.
# Both follow the same arc: name the previous career and the specific, non-obvious skill
# it produced; state the concrete upskilling with real artifacts (dashboards, two years
# of Python); then connect both to why *this* employer. Both also open with a claim
# rather than a pleasantry, which is the single biggest thing separating a good opening
# from "I am writing to express my interest in the position of..."
#
# Few-shot mostly controls tone and structure -- things that are much easier to
# demonstrate than describe. I can write "be confident and specific" in an instruction
# and the model will still produce clichés, because its prior for "cover letter" is
# saturated with generic ones. Two examples of what I actually mean override that prior
# far more effectively. What few-shot does *not* control is factual restraint; the
# examples contain specific details, and the model happily invents comparable details
# for the new person unless I forbid it separately, which is why the explicit "do not
# claim any experience not in the background" line is in the instruction rather than
# left to the examples.


# ==========================================
# Task 4: Moderation Check
# ==========================================


def is_safe(text: str) -> bool:
    """Return True if the text passes moderation, False (with a message) if flagged."""
    result = client.moderations.create(model="omni-moderation-latest", input=text)
    outcome = result.results[0]

    if outcome.flagged:
        # Show which categories tripped, which is useful for debugging borderline input
        triggered = [
            name for name, value in outcome.categories.model_dump().items() if value
        ]
        print("\nI'm not able to work with that message. Could you rephrase it and")
        print("we'll keep going?")
        if triggered:
            print(f"(Flagged categories: {', '.join(triggered)})\n")
        return False

    return True


# ==========================================
# Task 5: The Chatbot Loop
# ==========================================


def run_chatbot():
    # 1. Initialize conversation history with the system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\n" + "=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run the moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Bullet point rewriter
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")

            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)

            if not raw_bullets:
                print("Job Application Helper: I didn't get any bullets -- try again "
                      "whenever you're ready.\n")
                continue

            # Moderate the pasted bullets too; they never went through the check above
            if not is_safe("\n".join(raw_bullets)):
                continue

            results = rewrite_bullets(raw_bullets)

            # Keep the main conversation aware of what just happened, so follow-up
            # questions like "why did you change the second one?" still make sense.
            if results:
                summary = "\n".join(
                    f"- {item.get('original')} -> {item.get('improved')}"
                    for item in results
                )
                messages.append({"role": "user", "content": user_input})
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"I rewrote these resume bullets for you:\n{summary}",
                    }
                )

        # 6. Cover letter generator
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input(
                "Job Application Helper: Briefly describe your background: "
            ).strip()

            if not job_title or not background:
                print("Job Application Helper: I need both of those to write anything "
                      "useful. Let's try again.\n")
                continue

            # Moderate the sub-inputs as well
            if not is_safe(job_title) or not is_safe(background):
                continue

            draft = generate_cover_letter(job_title, background)
            print("\n--- Drafted Cover Letter Opening ---")
            print(draft)
            print("\nThis is a first draft. Edit it in your own voice, swap [Company]")
            print("for the real name, and check it against the job posting before you")
            print("send it.")
            print("------------------------------------\n")

            messages.append({"role": "user", "content": user_input})
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"I drafted this cover letter opening for a {job_title} role, "
                        f"based on this background: {background}\n\n{draft}"
                    ),
                }
            )

        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print(f"\nJob Application Helper: {reply}\n")
            messages.append({"role": "assistant", "content": reply})


# ==========================================
# Function tests (Tasks 2, 3, and 4)
# ==========================================


def run_tests():
    """Exercise each function once before starting the chatbot."""

    # --- Task 2 test ---
    print("=" * 50)
    print("TEST: Bullet Point Rewriter")
    print("=" * 50)
    test_bullets = [
        "Helped customers with their problems",
        "Made reports for the management team",
        "Worked with a team to finish the project on time",
    ]
    rewrite_bullets(test_bullets)

    # --- Task 3 test ---
    print("=" * 50)
    print("TEST: Cover Letter Generator")
    print("=" * 50)
    test_job = "Junior Data Engineer"
    test_bg = (
        "Five years of experience as a middle school math teacher; recently completed "
        "a Python course and built data pipelines using Prefect and Pandas."
    )
    print(generate_cover_letter(test_job, test_bg))

    # --- Task 4 test ---
    print("\n" + "=" * 50)
    print("TEST: Moderation Check")
    print("=" * 50)
    safe_text = "I am applying for a job as a junior Python developer."
    flagged_text = "I want to find where this hiring manager lives and hurt him."

    print(f"Safe input   -> is_safe() returned {is_safe(safe_text)}")
    print(f"Unsafe input -> is_safe() returned {is_safe(flagged_text)}")


# ==========================================
# Task 6: Ethics Reflection
# Format chosen: Option A -- comment block
# ==========================================
#
# --- ETHICS REFLECTION (Option A) ---
#
# Q1. How might this bot produce biased advice?
#
# The model learned what "professional" writing looks like from a corpus dominated by
# American and Western European corporate English, and my few-shot examples narrow it
# further -- both openings I supplied are assertive, individualistic, and lead with a
# personal claim. So the tool doesn't just improve writing, it converts writing toward
# one specific register. A candidate whose cultural or professional background favours
# collective framing ("our team achieved") or understatement will have that rewritten
# into first-person ownership, and someone who writes in a second language may find
# their voice replaced entirely with fluent-native phrasing that doesn't sound like them
# in an interview. There's an industry skew too: the advice is tuned to tech and
# corporate norms, and would actively mislead someone applying to academia, the trades,
# or government roles with structured application formats. The uncomfortable part is
# that this bias is partly *effective* -- resumes written in that register really do
# perform better with many US recruiters and screening tools -- so the tool ends up
# helping individuals by teaching them to conform to a standard that is itself the
# problem.
#
# Q2. What could go wrong if someone submitted the output without reviewing it?
#
# The biggest risk is fabricated credentials. Even with an explicit instruction not to
# invent facts, the model's pull toward "results-oriented" writing produces plausible
# specifics -- a percentage, a team size, a tool name -- and a user who trusts the
# output could send a resume claiming things they never did. That surfaces in the
# interview, and at that point it reads as dishonesty rather than as a tool error, with
# the consequences landing entirely on the candidate. Beyond fabrication: an unreviewed
# cover letter can go out with the literal "[Company]" placeholder still in it, or with
# a tone that's badly wrong for the employer, or generic enough that a recruiter
# recognizes it as AI-generated -- which a growing number of employers screen for
# explicitly.
#
# Q3. One guardrail I would add for a professional deployment.
#
# I'd add a claim-verification step between generation and copy: every rewritten bullet
# that introduces a specific detail not present in the original gets highlighted, and
# the user has to confirm or edit each highlighted claim before the text can be copied.
# This is better than a passive disclaimer -- which everyone scrolls past -- because it
# puts the friction exactly where the risk is and asks a question the user can actually
# answer ("did you really handle 200 tickets a month?"). It's cheap to implement, since
# it's a diff between input and output rather than another model call, and it has the
# side benefit of prompting users to supply the real numbers, which makes the resume
# genuinely stronger rather than just more confident-sounding.


if __name__ == "__main__":
    # Run the individual function tests, then start the chatbot.
    run_tests()
    run_chatbot()