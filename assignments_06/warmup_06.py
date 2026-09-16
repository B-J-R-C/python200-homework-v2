# ==========================================
# Week 6 Warmup: warmup_06.py
# ==========================================
# Covers: LLM augmentation strategies, keyword RAG, semantic RAG concepts,
# and building/evaluating a LlamaIndex pipeline.

import os
import string
from pathlib import Path

from dotenv import load_dotenv

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI

# Look for .env at the repo root (one level up from assignments_06/) so the key
# is found no matter which directory the script is launched from.
SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR.parent / ".env"

if load_dotenv(ENV_PATH) or load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# ==========================================
# --- RAG Concepts ---
# ==========================================

# Concepts Q1 -- which augmentation strategy fits each scenario?
#
# Scenario A (legal team, hundreds of PDFs, updated quarterly): RAG.
# The deciding factor is that the documents change every quarter. Fine-tuning
# would bake the policies into the model's weights, so every update would mean
# retraining -- and the model still couldn't cite which document an answer came
# from. RAG keeps the documents in an index that you re-build cheaply when
# policies change, and it can point back to the source, which a legal team is
# going to insist on.
#
# Scenario B (startup, 3,000 examples of a distinctive brand voice): Fine-tuning.
# Style is the thing fine-tuning is genuinely good at, and 3,000 examples is a
# realistic dataset size. The key detail is that the voice "does not appear much
# online" -- so it is weakly represented in the base model's training data, and
# no amount of describing it in a prompt will reproduce it reliably. Fine-tuning
# shifts the weights so the style becomes the default rather than something the
# model has to be talked into every call.
#
# Scenario C (analyst, one two-page report, one-off): Prompt engineering.
# Two pages fits comfortably in the context window, so she can paste the whole
# thing into the prompt. Building an index or fine-tuning would be real work for
# a task that ends when she closes the file. The rule of thumb: if the documents
# fit in context and you don't need to reuse the system, skip the infrastructure.


# Concepts Q2 -- why is a confident wrong answer more harmful?
#
# An answer that hedges hands the user a job: go verify this. A confidently
# wrong answer removes that prompt entirely -- the user has no signal that
# checking is necessary, so the error travels downstream unexamined. The damage
# isn't that the model was wrong; it's that it was wrong in a way that
# suppressed the verification that would have caught it.
#
# Example: an AI clinical assistant asked about a paediatric medication dose. If
# it returns "Consult the current dosing guidelines, I'm not certain," the nurse
# opens the reference. If it returns "0.5 mg/kg every 6 hours" in the same flat,
# authoritative register it uses for everything else, that number can go into a
# patient. The wrong dose and the uncertain answer are equally incorrect; only
# one of them gets administered.
#
# On tone: we read fluency and confidence as proxies for competence, because in
# humans they usually correlate -- people who know a subject tend to speak about
# it more fluidly. LLMs break that correlation completely. Fluency is what the
# model optimises for, and it's produced identically whether the underlying
# claim is well-grounded or invented. So the cue we instinctively use to gauge
# reliability is exactly the cue that carries no information here.


# Concepts Q3 -- the RAG pipeline in order.
#
# INDEXING (happens once, ahead of time):
#   1. "Extract text from source documents"
#      Pull raw text out of PDFs, Word files, HTML, etc.
#   2. "Split text into chunks"
#      Break long documents into passages small enough to retrieve precisely
#      and to fit in a prompt alongside other chunks.
#   3. "Convert text chunks into embeddings"
#      Run each chunk through an embedding model to get a vector representing
#      its meaning, and store it in the index.
#
# QUERY TIME (happens on every question):
#   4. "Receive the user's query"
#      The user asks a question.
#   5. "Embed the user's query"
#      Embed it with the SAME model used for the chunks -- vectors from
#      different models aren't comparable.
#   6. "Retrieve the most relevant chunks"
#      Compare the query vector against the stored vectors (cosine similarity)
#      and take the top k.
#   7. "Inject retrieved chunks into the prompt"
#      Build a prompt containing the retrieved text plus the question, usually
#      with an instruction to answer only from the provided context.
#   8. "Generate a response from the LLM"
#      The LLM writes the final answer grounded in that injected context.
#
# The split matters: steps 1-3 are a batch job you re-run when documents
# change, and steps 4-8 run per question. Confusing the two is how people end
# up re-embedding an entire corpus on every query.


# ==========================================
# --- Keyword RAG ---
# ==========================================


def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}


# Keyword Q1
print("\n--- Keyword Q1 ---")
query_1 = "What are your hours on weekends?"
result_1 = simple_keyword_retrieval(query_1, documents, verbose=True)
print(f"Selected document: {result_1[0][0]}")

# Keyword Q1 Comment:
# The function selected 'loyalty.txt' -- the WRONG document. I expected
# 'hours.txt' and was surprised, so I traced through the scoring.
#
# After stopword removal the query tokens are ['hours', 'weekends', 'what',
# 'your']. Scores come out as:
#   menu.txt    0
#   hours.txt   1  -> {'weekends'}
#   hiring.txt  1  -> {'your'}
#   loyalty.txt 1  -> {'your'}
#
# Two separate problems combine here. First, 'your' is a meaningless pronoun
# but it isn't in the stopword list (the list has "you" and "our", not "your"),
# so it scores a point in any document containing it -- and both hiring.txt and
# loyalty.txt contain "your". Meanwhile 'hours' scores nothing, because
# hours.txt never uses the word "hours"; it says "open Monday through Friday
# from 7am to 7pm". So the only real signal, 'weekends', gets exactly one point
# -- the same as the noise.
#
# Second, the tie-break is arbitrary. scores.sort(reverse=True) sorts tuples of
# (score, name, content); with all three scores tied at 1 it falls through to
# comparing the filename string, descending. Since 'loyalty.txt' sorts after
# 'hours.txt' alphabetically, the reverse sort puts it first -- so loyalty.txt
# wins on nothing but its name.
#
# This is a better illustration of keyword RAG's weakness than the failure I
# expected. It doesn't just miss synonyms -- it can rank an irrelevant document
# above a relevant one, and then report that result with total confidence,
# because "1 overlapping keyword" looks identical whether the keyword was
# 'weekends' or 'your'. The score carries no notion of which words matter.


# Keyword Q2
print("\n--- Keyword Q2 ---")
query_2 = "Do you have anything without caffeine?"
result_2 = simple_keyword_retrieval(query_2, documents, verbose=True)
print(f"Selected document: {result_2[0][0]}")

# Keyword Q2 Comment:
# Which document was selected: none. Every document scored 0 and the function
# returned its "No overlapping keywords found" fallback.
#
# Did keyword RAG get this right? No -- and it failed in the most complete way
# possible. menu.txt is plainly the relevant document, but the word "caffeine"
# appears in none of the four documents. The menu talks about espresso, lattes,
# cappuccinos, and cold brew, all of which a human instantly connects to
# caffeine, but that connection lives in world knowledge, not in shared
# characters. Set intersection has no access to it. (Worth noting: the real
# Groundwork corpus does contain the word "decaf" in wholesale_catering.txt --
# so even a system that found a lexical match here would have retrieved a
# document about catering orders rather than the menu.)
#
# What would do better: semantic retrieval. Embedding the query puts "without
# caffeine" near "decaf", "herbal tea", and the drink names in vector space,
# because the embedding model learned those associations from context during
# training. It would surface menu.txt without needing a single shared word.


# Keyword Q3
#
# Keyword Q3 Prediction (written before running):
# I predict 'None found'. The user says "rewards" and "sign up"; loyalty.txt
# says "loyalty program", "join", "earn", and "points". Conceptually identical,
# lexically disjoint -- there is no shared token for the intersection to find.

print("\n--- Keyword Q3 ---")
query_3 = "How do I sign up for rewards?"
result_3 = simple_keyword_retrieval(query_3, documents, verbose=True)
print(f"Selected document: {result_3[0][0]}")

# Keyword Q3 Result Reflection:
# Prediction confirmed. Query tokens after filtering are ['do', 'how', 'i',
# 'rewards', 'sign', 'up'], and all four documents score 0.
#
# This is the classic vocabulary mismatch problem: the user and the document
# author described the same thing using different words. It's not an edge case
# -- it's the normal condition, because the person asking the question has
# never read the document and doesn't know its terminology. Note the contrast
# with Q1: there, noise words produced a confident wrong answer; here, the
# absence of shared words produces no answer at all. Both are failures, but the
# silent one is arguably safer, since at least it doesn't mislead.


# ==========================================
# --- Semantic RAG Concepts ---
# ==========================================

# Semantic Q1
#
# 1. What is a vector embedding?
# It's a list of numbers -- typically several hundred to a couple thousand of
# them -- that encodes the meaning of a piece of text as a position in a
# high-dimensional space. The model producing it was trained such that texts
# used in similar contexts land near each other, so the geometry of that space
# ends up carrying semantic structure: direction and distance between points
# correspond to relationships in meaning.
#
# 2. Cosine similarity of 0.85 vs 0.30 -- which is more relevant?
# The 0.85 chunk, by a wide margin. Cosine similarity measures the angle
# between two vectors, ignoring their magnitude, on a scale from -1 to 1. At
# 0.85 the two vectors point in nearly the same direction, meaning the chunk
# and the query occupy almost the same region of meaning-space -- likely the
# same topic answered directly. At 0.30 they're closer to perpendicular, which
# indicates the texts are largely unrelated; they might share a general domain
# but the chunk almost certainly doesn't answer the question. One caution I
# want to record: these numbers are only comparable *within* one embedding
# model and one corpus. A raw 0.30 isn't universally "bad" -- some models
# compress most real-world text into a narrow band -- so thresholds have to be
# calibrated against your own data, not assumed.
#
# 3. Why can semantic search match a chunk with no shared words?
# Because the comparison happens after both texts have been converted to
# vectors, and that conversion is driven by meaning rather than spelling. The
# embedding model was trained on enormous amounts of text where words like
# "rewards", "loyalty", and "points" appear in interchangeable contexts, so it
# learned to place them close together. By the time the search runs, the
# original characters are gone -- all that's left is position. Matching happens
# in that space, which is why "How do I sign up for rewards?" can retrieve a
# passage about joining a loyalty program despite sharing zero tokens.


# Semantic Q2
#
# | Feature                    | Keyword RAG                       | Semantic RAG                                      |
# |----------------------------|-----------------------------------|---------------------------------------------------|
# | What is compared?          | Exact word overlap                | Vector embeddings -- proximity in meaning-space   |
# | What is retrieved?         | Full document                     | Individual chunks (passages) of documents         |
# | Can it handle synonyms?    | No                                | Yes -- synonyms embed to nearby vectors           |
# | Storage format             | Plain text dictionary             | Vector store / index of embeddings + metadata     |
# | Relevance score            | Number of overlapping keywords    | Cosine similarity, a continuous value in [-1, 1]  |
#
# Two things the table doesn't capture. Keyword scores are integers with no
# upper bound and no calibration -- "3 overlaps" means nothing without knowing
# the query length -- whereas cosine similarity is continuous and bounded,
# which is what makes threshold filtering possible at all. And retrieving
# chunks rather than whole documents is what lets the prompt hold results from
# several documents at once; that's a precision advantage, not just a storage
# detail. Semantic RAG isn't strictly better, though: keyword search still wins
# on exact identifiers like part numbers, error codes, or surnames, where you
# want the literal string and not something that means roughly the same. Real
# production systems often run both and merge the results (hybrid search).


# ==========================================
# --- LlamaIndex ---
# ==========================================
#
# Path note: the PDFs live in assignments_06/brightleaf_pdfs/ in this repo.
# Paths are resolved relative to THIS FILE rather than the current working
# directory, so the script works no matter where it is launched from and still
# works after someone else clones the repo.

PDF_DIR = str(SCRIPT_DIR / "brightleaf_pdfs")

# Fall back to the lesson folder if the PDFs were not copied into the repo
if not os.path.exists(PDF_DIR):
    lesson_copy = SCRIPT_DIR.parent.parent / "06_AI_augmentation" / "brightleaf_pdfs"
    if lesson_copy.exists():
        PDF_DIR = str(lesson_copy)

if not os.path.exists(PDF_DIR):
    print(f"\nCould not find the BrightLeaf PDFs. Looked at: {PDF_DIR}")
    print("Update PDF_DIR to point at brightleaf_pdfs/ and re-run.")
else:
    # ---------- LlamaIndex Q1 ----------
    print("\n--- LlamaIndex Q1 ---")
    documents_llama = SimpleDirectoryReader(PDF_DIR).load_data()
    print(f"Loaded {len(documents_llama)} document objects from {PDF_DIR}")

    # Sanity check on PDF text extraction.
    #
    # This bit me on my first run, and the failure is worth recording because it
    # is completely silent. If pypdf is not installed, SimpleDirectoryReader does
    # not raise -- it falls back to reading each PDF as raw bytes. Those bytes get
    # chunked and embedded like any other text, so the pipeline runs to completion
    # and produces confident-looking output. The retrieval scores even look
    # healthy (~0.79), because the filename metadata still carries topic signal.
    # What you actually get are answers like "BrightLeaf offers a variety of
    # employee benefits" -- fluent, contentless, and describing the file rather
    # than its contents.
    #
    # I check two things: that the parser is importable, and that the loaded text
    # does not look like PDF internals.
    # The PDF reader does NOT live in llama-index-core. It ships in the separate
    # llama-index-readers-file package, and having pypdf installed is not enough
    # -- core never imports pypdf directly, it imports llama_index.readers.file.
    try:
        import llama_index.readers.file  # noqa: F401
    except ImportError:
        raise SystemExit(
            "\nERROR: llama-index-readers-file is not installed, so SimpleDirectoryReader\n"
            "has no PDF reader registered and falls back to reading the files as raw\n"
            "bytes. Nothing raises -- you just get binary garbage embedded.\n"
            "Fix:  python -m pip install llama-index-readers-file\n"
        )

    sample_text = documents_llama[0].text[:400]
    if any(m in sample_text for m in ("%PDF", "endobj", "/Type", "ReportLab")):
        raise SystemExit(
            "\nERROR: the loaded documents look like raw PDF structure rather than\n"
            "extracted text. Check that pypdf is working and that the files in\n"
            f"{PDF_DIR} are real PDFs.\n"
            f"\nWhat was loaded:\n{sample_text[:200]!r}\n"
        )
    print(f"Text extraction OK. Sample: {sample_text[:80]}...")

    index = VectorStoreIndex.from_documents(documents_llama)
    query_engine_3 = index.as_query_engine(similarity_top_k=3)

    questions = [
        "What employee benefits does BrightLeaf offer?",
        "What are BrightLeaf's security policies?",
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        response = query_engine_3.query(q)
        print(f"Answer: {response.response}\n")

        for i, node in enumerate(response.source_nodes, start=1):
            source = node.node.metadata.get("file_name", "unknown")
            print(f"  Node {i} | {source} | score={node.score:.4f}")
            print(f"  Text: {node.node.text[:150]}...\n")

    # LlamaIndex Q1 Comment:
    #
    # BROKEN RUNS (before llama-index-readers-file was installed) -- worth
    # recording because the failure was instructive and completely silent:
    #
    # Root cause, which took a while to find: llama-index-core does NOT contain
    # a PDF reader. It lives in a separate package, llama-index-readers-file.
    # Core tries `from llama_index.readers.file import PDFReader, ...` inside a
    # try/except ImportError; when that fails it returns an empty reader dict, so
    # ".pdf" is never registered. SimpleDirectoryReader then falls through to its
    # generic branch, which opens the file and decodes it as text with
    # errors="ignore" -- which is why it produced a plausible-looking string
    # rather than raising UnicodeDecodeError. Having pypdf installed made no
    # difference: core never imports pypdf directly.
    #
    # Retrieval put the right FILE at the top for both queries --
    # employee_benefits.pdf at 0.7923 for the benefits question,
    # security_policy.pdf at 0.8165 for the security question. But the chunk
    # text was raw PDF binary (';FT"h:K.ai,f?S69JlS[McJNtaQhM...' and
    # '%PDF-1.4 % ReportLab Generated PDF document'), so nothing usable reached
    # the model.
    #
    # The answers reflected that without admitting it:
    #   "BrightLeaf offers employee benefits outlined in the PDF document
    #    located at the specified file path."
    #   "BrightLeaf's security policies are outlined in the PDF document located
    #    at the file path: ...\security_policy.pdf"
    #
    # Both are grammatical, confident, and completely contentless -- the model
    # described the file rather than its contents, because the filename was the
    # only readable signal it had. Two things I want to remember from this:
    #
    # (a) Similarity scores near 0.79-0.82 looked perfectly healthy while the
    #     embedded text was garbage, and the right FILE still ranked top for each
    #     query. I cannot tell from the output what the embeddings keyed on --
    #     possibly the readable metadata and filename, possibly incidental
    #     structure in the bytes -- but the lesson holds either way: a high score
    #     is not evidence that retrieval worked. You have to read the chunk text.
    # (b) Nothing in the response's tone signalled the problem. This is the same
    #     confident register the model uses when it has real context, which is
    #     exactly the hallucination-adjacent failure the Concepts Q2 answer is
    #     about: fluency is uncorrelated with grounding.
    #
    # WORKING RUN (llama-index-readers-file installed, PDFs parsing correctly):
    #
    # Are the retrieved chunks relevant? The top chunk, yes -- decisively.
    # employee_benefits.pdf scored 0.9090 on the benefits question and
    # security_policy.pdf scored 0.8821 on the security question, both a clear
    # margin above everything else. The chunk text is now real prose
    # ("Introduction / BrightLeaf Solar views employee well-being as inseparable
    # from long-term innovation...").
    #
    # But nodes 2 and 3 are NOT relevant, and this is the more interesting part.
    # For the benefits query they were mission_statement.pdf (0.8164) and
    # security_policy.pdf (0.8155); for the security query, employee_benefits.pdf
    # (0.8390) and mission_statement.pdf (0.8217). The two queries retrieved
    # essentially the same three documents in swapped order.
    #
    # The reason is chunk granularity. I checked the source PDFs: each is a
    # single page of roughly 2,200-3,300 characters, about 550-830 tokens, which
    # is under LlamaIndex's default 1024-token chunk size. So each document
    # becomes ONE node, and "retrieve the top 3 chunks" is really "retrieve the
    # top 3 documents" -- there is no sub-document precision available. Since
    # every document shares BrightLeaf corporate vocabulary, the runners-up score
    # 0.81-0.84 on topics they have nothing to do with.
    #
    # Those are high absolute numbers -- higher than several correct top-1 hits
    # elsewhere in this assignment -- which is the reminder worth keeping:
    # similarity scores are only meaningful relative to the other candidates in
    # the same query, never against a fixed scale.
    #
    # Tone: confident and specific, and this time correctly so. The benefits
    # answer named the Wellness Reimbursement Plan, the 401(k) company match, and
    # the Learning Hub; the security answer named 90-day credential rotation,
    # NIST 800-61, and ISO 27001. Compare that with the pre-fix run above, where
    # the identical register produced "BrightLeaf offers employee benefits
    # outlined in the PDF document located at the specified file path." Same
    # confidence, one grounded and one hollow. The tone genuinely carries no
    # information about whether retrieval worked.
    #
    # Anything unexpected? Two things. First, the model ignored the irrelevant
    # nodes cleanly -- nothing from the mission statement or security policy
    # leaked into the benefits answer, which is better behaviour than I expected
    # given those chunks scored above 0.81. Second, the pypdf warnings
    # ("incorrect startxref pointer", "parsing for Object Streams") show the PDFs
    # are slightly malformed, yet extraction still worked. The "■" characters
    # visible in the security chunk are mangled hyphens from the PDF's encoding,
    # so extraction is imperfect even when it succeeds.

    # ---------- LlamaIndex Q2 ----------
    print("\n--- LlamaIndex Q2 ---")
    q2 = "What employee benefits does BrightLeaf offer?"

    query_engine_1 = index.as_query_engine(similarity_top_k=1)
    res_1 = query_engine_1.query(q2)
    print("\ntop_k=1 answer:", res_1.response)
    print(f"top_k=1 node score: {res_1.source_nodes[0].score:.4f}")

    query_engine_5 = index.as_query_engine(similarity_top_k=5)
    res_5 = query_engine_5.query(q2)
    print("\ntop_k=5 answer:", res_5.response)
    for i, node in enumerate(res_5.source_nodes, start=1):
        source = node.node.metadata.get("file_name", "unknown")
        print(f"  top_k=5 node {i} | {source} | score={node.score:.4f}")

    # LlamaIndex Q2 Comment:
    #
    # BROKEN RUNS (before llama-index-readers-file, so the chunks were binary):
    #   top_k=1: "BrightLeaf offers a variety of employee benefits."  (score 0.7923)
    #   top_k=5: "The employee benefits offered by BrightLeaf are not specified
    #             in the provided context information."
    #   top_k=5 scores: 0.7923, 0.7786, 0.7684, then 0.7367 (earnings_report.pdf)
    #                   and 0.7362 (partnerships.pdf)
    #
    # Even through the binary failure, the k comparison showed something real.
    # At k=1 the model produced a vague but affirmative-sounding claim; at k=5
    # it declined outright. More context made it MORE likely to admit it had
    # nothing -- the opposite of "more context is always better," and a useful
    # reminder that adding low-relevance chunks changes model behaviour rather
    # than simply adding information.
    #
    # The score list also shows the top-k structure clearly: the three
    # employee_benefits.pdf chunks cluster around 0.77-0.79, then nodes 4 and 5
    # drop to different documents entirely (earnings_report, partnerships) at
    # ~0.736. Those last two are only loosely on-topic, retrieved because
    # top-k always returns k results regardless of quality.
    #
    # WORKING RUN (PDFs parsing correctly) -- and this result surprised me.
    #
    #   top_k=1 (score 0.9090): a complete, correct answer listing medical
    #     insurance, vision, wellness programs, the Wellness Reimbursement Plan,
    #     life and disability insurance, 401(k) with company match, parental
    #     leave, work flexibility, professional development stipend, mentorship,
    #     the DEI Council, and the Learning Hub.
    #   top_k=5 (0.9090, 0.8164, 0.8155, 0.8096, 0.7858): a similarly complete
    #     answer, slightly more detailed in places (it added "preventive care,
    #     hospitalization, telemedicine, and mental health services" under
    #     health, and named fitness memberships and nutrition counselling), but
    #     it dropped the DEI Council that k=1 mentioned.
    #
    # I expected k=1 to be visibly thinner and it was not. The reason is the
    # corpus, not the parameter: employee_benefits.pdf is a single page of roughly
    # 2,600 characters, well under the default 1024-token chunk size, so the
    # whole document is a single node. That one node already contained every benefit
    # category, leaving k=1 nothing to miss. k=1 is only a handicap when the
    # answer spans chunk boundaries -- which requires documents long enough to
    # be split, and these are not.
    #
    # That makes the k=5 comparison a clean cost demonstration. Nodes 2-5 were
    # mission_statement, security_policy, partnerships, and earnings_report --
    # every one irrelevant to employee benefits. So k=5 sent four extra documents
    # of context to the model, paid for those tokens, took noticeably longer on the
    # completion call (visible in the HTTP request timestamps in the log), and
    # returned an answer that was not better -- it traded one detail for another.
    #
    # Is more context always better? No, and this run shows the specific reason:
    # top-k retrieval has no relevance floor, so raising k does not "add more
    # relevant context" -- it adds the next-best candidates whether or not they
    # are any good. Here that meant four irrelevant documents.
    #
    # The costs of a larger k: you pay for every retrieved token on every query,
    # so spend and latency scale directly with k; low-relevance chunks dilute the
    # prompt and can pull the model off-topic, and with long contexts models tend
    # to attend less to material in the middle; and a bigger k raises the chance
    # of sweeping in a chunk that contradicts the good ones, which the model has
    # no reliable way to adjudicate.
    #
    # The right k therefore depends on how information is distributed in the
    # corpus -- answers spread across sections need a higher k, self-contained
    # ones do better with a low k. That is an empirical question about your own
    # data, not a number to copy from a tutorial.

    # ---------- LlamaIndex Q3 ----------
    print("\n--- LlamaIndex Q3 ---")
    q3_hard = "Who is the CEO's favorite band?"
    res_hard = query_engine_3.query(q3_hard)
    print(f"\nQuestion: {q3_hard}")
    print("Answer:", res_hard.response)
    for i, node in enumerate(res_hard.source_nodes, start=1):
        source = node.node.metadata.get("file_name", "unknown")
        print(f"  Node {i} | {source} | score={node.score:.4f}")
        print(f"  Text: {node.node.text[:100]}...")

    # LlamaIndex Q3 Comment:
    #
    # What I expected: a refusal. The BrightLeaf documents are corporate
    # material -- financials, benefits, security, products, partnerships,
    # mission -- and none of them names a CEO at all, let alone their musical
    # taste. I chose this deliberately as a question with no possible grounding.
    #
    # What happened: retrieval returned three chunks anyway, from
    # security_policy.pdf (0.7107), employee_benefits.pdf (0.7092), and
    # mission_statement.pdf (0.6936). The model answered "I do not have
    # information regarding the CEO's favorite band based on the provided
    # context."
    #
    # Two things stand out. First, the structural point -- a top-k retriever
    # ALWAYS returns k results. It has no concept of "nothing here is relevant";
    # it ranks whatever exists and hands back the best of a bad set. The model
    # declining is a property of the generator, not the retriever.
    #
    # Second, and more useful: compare the score PROFILE against the working
    # queries. Here the three nodes are 0.7107 / 0.7092 / 0.6936 -- a near
    # three-way tie, top result barely ahead of third. On the benefits query the
    # profile was 0.9090 / 0.8164 / 0.8155, a decisive winner with a 0.09 gap to
    # second place. That gap, not the absolute value, is what distinguishes
    # "found the answer" from "found nothing in particular". A flat spread means
    # the retriever is returning the least-bad options from a corpus that has no
    # match, which is exactly the situation where the generator is most likely to
    # improvise. Worth noting that an absolute threshold would have struggled
    # here: 0.7107 is not far below the 0.7858 that legitimately appeared as
    # node 5 in the Q2 top_k=5 run.
    #
    # So the guardrail that saved this query was the generator, not the
    # retriever -- and that's a thinner defence than it looks. The model
    # declined here because the gap between "solar panel specifications" and
    # "favourite band" is enormous and obvious. A question whose plausible-
    # looking answer sits adjacent in the retrieved text is a much harder case,
    # and there the same pipeline is far more likely to reach for the nearby
    # detail and present it confidently.
    #
    # What I'd change: add a similarity floor via SimilarityPostprocessor, so
    # chunks below a calibrated cutoff are dropped before the LLM sees them; if
    # nothing survives, skip generation and return "no relevant documents
    # found". That turns a silent low-confidence retrieval into an explicit
    # refusal and saves the generation call. The threshold has to be tuned
    # against real queries on this corpus rather than picked out of the air --
    # set it too high and legitimate questions start getting refused.

    # ---------- LlamaIndex Q4 ----------
    print("\n--- LlamaIndex Q4 ---")

    judge_llm = OpenAI(model="gpt-4o-mini")
    faithfulness = FaithfulnessEvaluator(llm=judge_llm)
    relevancy = RelevancyEvaluator(llm=judge_llm)

    # Evaluate the good query
    good_q = "What employee benefits does BrightLeaf offer?"
    print(f"\nEvaluating (expected good): {good_q}")
    good_resp = query_engine_3.query(good_q)

    f_eval = faithfulness.evaluate_response(response=good_resp)
    r_eval = relevancy.evaluate_response(query=good_q, response=good_resp)
    print(f"  Faithfulness score: {f_eval.score}")
    print(f"  Relevancy score:    {r_eval.score}")

    # Evaluate a query whose answer is not in the documents
    bad_q = "Who is the CEO's favorite band?"
    print(f"\nEvaluating (expected poor): {bad_q}")
    bad_resp = query_engine_3.query(bad_q)

    f_eval_bad = faithfulness.evaluate_response(response=bad_resp)
    r_eval_bad = relevancy.evaluate_response(query=bad_q, response=bad_resp)
    print(f"  Faithfulness score: {f_eval_bad.score}")
    print(f"  Relevancy score:    {r_eval_bad.score}")

    # LlamaIndex Q4 Comment:
    #
    # 1. What does faithfulness 1.0 mean, and what would 0.0 mean?
    # Faithfulness asks a single question: is every claim in the generated
    # answer supported by the retrieved context? A score of 1.0 means yes --
    # the answer is fully grounded, nothing was added from the model's own
    # parametric knowledge or invented outright. A 0.0 means the answer
    # contains claims the context does not support: the model hallucinated, or
    # answered from training data while ignoring the documents it was given.
    # Crucially, faithfulness says nothing about whether the answer is *useful*
    # or even whether it addresses the question -- only whether it stayed
    # inside its evidence.
    #
    # 2. What does relevancy measure, and how does it differ?
    # Relevancy asks whether the response actually addresses the user's query.
    # The two are independent, which is the point of running both. An answer
    # can be perfectly faithful and completely irrelevant -- if I ask about
    # holiday policy and the system faithfully summarises the encryption
    # standard, faithfulness is 1.0 and relevancy is 0.0. The reverse also
    # happens: a fluent, on-topic answer built from invented facts scores well
    # on relevancy and badly on faithfulness. Faithfulness catches
    # hallucination; relevancy catches retrieval that fetched the wrong thing.
    #
    # 3. Did the scores change between the two queries?
    #
    # These runs happened while the PDF reader was broken (see the note in Q1),
    # so every chunk was raw PDF binary. I am keeping the numbers because the
    # evaluators behaved informatively even on garbage input.
    #
    #   Run 1:
    #     "What employee benefits does BrightLeaf offer?"  faithfulness 0.0, relevancy 1.0
    #     "Who is the CEO's favorite band?"                faithfulness 0.0, relevancy 1.0
    #   Run 2 (identical code and data):
    #     "What employee benefits does BrightLeaf offer?"  faithfulness 0.0, relevancy 0.0
    #     "Who is the CEO's favorite band?"                faithfulness 0.0, relevancy 1.0
    #
    # Three things worth recording.
    #
    # (a) Faithfulness was 0.0 across the board, correctly. No answer could be
    #     grounded in its context, because the context was binary noise. This is
    #     the metric earning its keep: the generated answers read as fluent and
    #     confident ("BrightLeaf offers a variety of employee benefits"), and
    #     nothing in the text itself revealed that retrieval had completely
    #     failed. Reading the answers would not have caught this; the score did.
    #
    # (b) The benefits query's relevancy flipped between two runs of identical
    #     code -- 1.0 then 0.0. Nothing changed except the sampling in the judge
    #     LLM's own generation. That is a concrete demonstration that
    #     LLM-as-a-judge is stochastic, not a deterministic measurement. A single
    #     evaluation run is a noisy sample, which is why real evaluation averages
    #     over a set of queries rather than trusting one number.
    #
    # (c) The CEO query held relevancy 1.0 in both of these runs -- but see the
    #     third run below, where it dropped to 0.0 on a nearly identical refusal.
    #     Across all three runs the refusal was scored 1.0, 1.0, then 0.0, which
    #     means the judge is inconsistent about whether "I don't have that
    #     information" counts as relevant to the question. Two runs would have
    #     convinced me it was stable; three showed it wasn't.
    #
    # WORKING RUN (PDFs parsing correctly) -- the scores finally separate:
    #
    #   "What employee benefits does BrightLeaf offer?"  faithfulness 1.0, relevancy 1.0
    #   "Who is the CEO's favorite band?"                faithfulness 0.0, relevancy 0.0
    #
    # The benefits query now scores 1.0 on both, as it should: the answer is
    # detailed, entirely supported by employee_benefits.pdf, and addresses what
    # was asked. Faithfulness moving from 0.0 to 1.0 purely because the retrieval
    # bug was fixed -- with no change to the prompt, model, or question -- is a
    # clean demonstration that the metric tracks grounding rather than answer
    # quality in the abstract.
    #
    # The CEO query scoring 0.0 on faithfulness is worth thinking about, because
    # I predicted the opposite. My reasoning was that "I do not have information
    # regarding the CEO's favorite band" is a truthful statement about the
    # context, so it should be perfectly faithful. The judge disagreed. The most
    # likely explanation is that FaithfulnessEvaluator asks whether the response
    # is SUPPORTED BY the context -- and a refusal asserts nothing that the
    # retrieved chunks about network security and employee benefits can support.
    # A statement about the absence of information is not the same as a statement
    # derived from the information. So a correct refusal scores 0.0, exactly like
    # a hallucination would.
    #
    # That is a real limitation to record: these evaluators cannot distinguish
    # "appropriately declined" from "made something up". Both come back 0.0. If I
    # were using faithfulness to monitor a production RAG system, refusals would
    # look identical to fabrications in the metrics, and I would need to filter
    # them out separately before the numbers meant anything.
    #
    # The general point stands either way: a single "accuracy" number would be
    # misleading here, because a correct refusal and a confident hallucination
    # are both "wrong" to a naive metric while being opposite behaviours.
    #
    # 4. What is "LLM-as-a-judge", and why use it for RAG?
    # It means using a language model to grade another model's output against
    # rubric-style criteria -- here, feeding the query, retrieved context, and
    # generated answer to gpt-4o-mini and asking whether the answer is
    # supported and on-topic. It's used because free-text answers have no
    # single correct string to compare against. Exact-match or F1 would punish
    # a correct answer for using different wording than a reference, and there
    # is no reference answer here at all. Judging whether a paragraph is
    # entailed by a source document requires understanding the language, which
    # is what makes an LLM a workable evaluator.
    # The obvious caveat: the judge is a language model with the same failure
    # modes as the one it's grading, so these scores are a useful signal rather
    # than ground truth. Standard practice is to spot-check judge verdicts by
    # hand and use a stronger model as judge than as generator where budget
    # allows.

print("\n=== End of warmup_06.py ===")