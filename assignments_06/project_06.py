# ==========================================
# Week 6 Mini-Project: project_06.py
# Groundwork Coffee Co. Q&A Assistant
# ==========================================
# A RAG-powered assistant that answers questions about Groundwork Coffee Co.
# from their own documents, using LlamaIndex for chunking, embedding, indexing,
# and retrieval.

from pathlib import Path

from dotenv import load_dotenv

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# ==========================================
# Step 1: Setup
# ==========================================

# Paths are resolved relative to THIS FILE rather than the current working
# directory, so the script behaves the same no matter where it is launched from
# and still works after someone else clones the repo.
SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR.parent / ".env"

if load_dotenv(ENV_PATH) or load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# The assert below stops the script immediately with a clear message rather than
# letting SimpleDirectoryReader fail later with something more cryptic.
DOCS_DIR = SCRIPT_DIR / "resources" / "groundwork_docs"

# Fall back to a copy sitting directly in assignments_06/
if not DOCS_DIR.exists() and (SCRIPT_DIR / "groundwork_docs").exists():
    DOCS_DIR = SCRIPT_DIR / "groundwork_docs"

assert DOCS_DIR.exists(), (
    f"Document directory not found: {DOCS_DIR.resolve()}\n"
    "Update DOCS_DIR to point at the groundwork_docs folder."
)
print(f"Using documents from: {DOCS_DIR}")


# ==========================================
# Step 2: Load the Documents
# ==========================================

print("\n--- Loading Documents ---")
documents = SimpleDirectoryReader(str(DOCS_DIR)).load_data()

print(f"Total documents loaded: {len(documents)}")
for doc in documents:
    print(f"  Loaded: {doc.metadata.get('file_name', 'unknown')}")

# Note: len(documents) counts Document objects, not files. SimpleDirectoryReader
# creates one Document per file for .txt, but multi-page formats like PDF
# produce one Document per page -- so this number matching the file count is
# specific to this corpus being plain text.


# ==========================================
# Step 3: Build the Index and Query Engine
# ==========================================

print("\n--- Building Index ---")
# This is the step that costs money: every chunk is sent to the OpenAI
# embeddings API. The index lives in memory, so it is rebuilt on every run.
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)
print("Index built successfully. Ready to answer questions.")


# ==========================================
# Step 4: Query the Assistant
# ==========================================

print("\n--- Querying Assistant ---")
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print(f"A: {response.response}")

    top_node = response.source_nodes[0]
    filename = top_node.node.metadata.get("file_name", "unknown")
    print(f"   Top source: [{filename}] | score={top_node.score:.4f}")
    print(f"   Chunk preview: {top_node.node.text[:200]}...")

# Step 4 Reflection:
#
# All five answers were accurate and confidently phrased. Retrieval routed each
# question sensibly: hours and loyalty to faq.txt (0.8080, 0.7670), the founding
# story to our_story.txt (0.8938 -- the highest score of the run), catering to
# wholesale_catering.txt (0.8515).
#
# The dairy-free question is the one worth dwelling on, and it did surprise me.
# It is the exact query that broke keyword RAG in the warmup, and semantic RAG
# answered it correctly -- "All dairy-free options are available at no extra
# charge, including oat milk, almond milk, and soy milk." But look at the top
# source: seasonal_specials.txt at 0.7761, NOT menu.txt. The milk list lives in
# menu.txt; seasonal_specials.txt just happens to use the exact phrase
# "Dairy-free" twice as a tag on individual drinks.
#
# So the highest-scoring chunk was not the one containing the answer. The
# correct answer came from a lower-ranked chunk that similarity_top_k=3 happened
# to include. That is a useful lesson about reading top-1 scores as a proxy for
# retrieval quality: the ranking was arguably wrong while the overall result was
# right, and only the k=3 window saved it. With similarity_top_k=1 this question
# would very likely have produced a worse answer, or one drawn only from the
# seasonal drinks.
#
# The loyalty answer was also more complete than I expected -- it picked up that
# the program is free to join and that you sign up at the register or online,
# not just the points mechanics. The chunk boundary kept the whole loyalty
# section of faq.txt together; with a smaller chunk size that answer could
# easily have come back partial.
#
# One weaker result: "Do you offer catering or wholesale orders?" returned just
# "Yes, catering and wholesale orders are available." That is correct but thin
# given the source document contains the 20-person minimum, the 72-hour notice
# requirement, and the contact email. The question was phrased as a yes/no, and
# the model answered it as one.


# ==========================================
# Step 5: Find a Failure
# ==========================================

print("\n--- Testing Failure Mode ---")
bad_query = "What is the manager's name and how much do you charge for a large vanilla latte?"
bad_resp = query_engine.query(bad_query)

print(f"\nQ: {bad_query}")
print(f"A: {bad_resp.response}\n")

for i, node in enumerate(bad_resp.source_nodes, start=1):
    filename = node.node.metadata.get("file_name", "unknown")
    print(f"Source {i}: [{filename}] | score={node.score:.4f}")
    print(f"Preview: {node.node.text[:200]}...\n")

# Step 5 Comment:
#
# 1. What I asked and why I expected it to be hard.
# I asked a two-part question where neither part has an answer in the corpus:
# the name of a manager, and the price of a "large vanilla latte". I chose it
# specifically because both halves have *near misses* in the documents rather
# than being absent outright. That is a harder test than asking something
# wholly unrelated, because the retrieved context will look superficially
# useful.
#
# 2. What went wrong. THE MODEL HALLUCINATED A PRICE.
#
# The actual response was:
#   "The manager's name is not provided in the context information.
#    The price for a large vanilla latte is $5.00."
#
# There is no large vanilla latte on any Groundwork menu, and no size tiers
# exist anywhere in the corpus. Retrieved chunks were seasonal_specials.txt
# (0.7952), menu.txt (0.7783), and wholesale_catering.txt (0.7623).
#
# Tracing where "$5.00" could have come from, the top chunk alone offers two
# candidates: the Iced Lavender Lemonade at $5.00 and the Honey Cardamom Latte
# at $5.00 -- the latter being an actual latte at exactly the quoted price.
# menu.txt adds a Matcha latte at $5.00 and a plain Latte at $4.50. The word
# "vanilla" appears exactly once in the whole corpus, as an ingredient in the
# $5.50 Horchata Latte. I cannot tell from the output which of these the model
# drew on, and I should not pretend to -- what is verifiable is that every
# ingredient of the answer exists somewhere in the retrieved context while the
# combination it asserts does not.
#
# That is the essence of the failure: the model produced a confident, specific,
# fabricated fact by recombining true fragments. A price that is real for other
# drinks, a flavour word that is real for a different drink, and a size category
# that exists nowhere. Notably, a naive check for "does this number appear in
# the retrieved context?" would pass -- $5.00 is genuinely there. Only checking
# that the number is attached to the right entity catches it. I produced this on
# the first attempt with an ordinary customer question.
#
# The split behaviour within a single response is the striking part. Asked two
# things it could not answer, the model correctly refused the first ("the
# manager's name is not provided") and fabricated the second -- in consecutive
# sentences, in the same flat register. Refusing is clearly within its
# repertoire; it just did not apply that behaviour consistently. My read is that
# the difference is what the context afforded: nothing in the retrieved chunks
# resembled a person's name, but the chunks were full of drink names and dollar
# amounts, so a plausible-looking price was easy to assemble. Hallucination
# tracked the availability of adjacent-looking material, not the model's actual
# knowledge of whether it had the answer.
#
# 3. Did the model's tone change?
# No -- and that is the finding that matters most. Compare the two sentences:
# "The manager's name is not provided in the context information" (true refusal)
# and "The price for a large vanilla latte is $5.00" (fabrication). Same
# sentence structure, same confidence, same neutral register, no hedging on
# either. There is no stylistic signal separating the sentence I can trust from
# the one I cannot.
#
# The retrieval scores did not warn me either. At 0.7952 / 0.7783 / 0.7623 they
# sit comfortably inside the range of the five successful Step 4 queries
# (0.7670 to 0.8938). The dairy-free question scored 0.7761 at top-1 and gave a
# correct answer; this question scored 0.7952 and produced a fabrication. So on
# this corpus a healthy-looking score does not distinguish a good answer from an
# invented one, and a threshold tuned to catch this would also have rejected
# legitimate questions.
#
# The practical conclusion: you cannot audit a RAG system by reading its
# answers, and on this evidence you cannot fully audit it by reading its scores
# either. The only reliable check was reading the retrieved chunks myself and
# confirming the claimed fact appeared in them -- which is what printing the
# source text is for, and what a real deployment would need to automate.
#
# 4. What I would change.
#
# My original plan here was a similarity floor via SimilarityPostprocessor, and
# running the experiment changed my mind about it. The hallucinating query
# scored 0.7952 at top-1, higher than the dairy-free question (0.7761) that was
# answered correctly. Any threshold that would have blocked the fabrication
# would also have blocked a good answer, so a similarity floor does not solve
# this failure. It would still be worth adding for the truly-unrelated case
# (the warmup's "CEO's favourite band" retrieved 0.7107 / 0.7092 / 0.6936 --
# note how little separates first from third, which is the real signal that
# nothing matched), but it is not the fix for this one. Revised list:
#
# (a) A stricter prompt template: answer only from the context, and for a
#     multi-part question address each part separately, saying explicitly when
#     one cannot be answered. The model already demonstrated it can refuse -- it
#     did so for the manager's name in the same response -- so the goal is
#     making that behaviour consistent rather than teaching it something new.
#     This is the cheapest change and targets the observed failure directly.
#
# (b) An entity-level grounding check on numeric claims. The naive version --
#     "does this figure appear in the retrieved chunks?" -- provably fails here,
#     because $5.00 appears four separate times in the corpus (Iced Lavender
#     Lemonade, Honey Cardamom Latte, Matcha latte, and the Pour-over). The
#     check has to verify
#     that the number is bound to the entity the answer attaches it to, so
#     "large vanilla latte = $5.00" fails because no such item exists on any
#     line. That is harder to build, but numbers are where fabrication does the
#     most damage and are still the most mechanically checkable part of an
#     answer.
#
# (c) Answering with citations by default -- show the source chunk beside each
#     claim. This does not prevent hallucination, but it makes it visible to the
#     person reading, who can see that the quoted price belongs to a lemonade.
#     Given that neither tone nor score distinguished the bad answer here, moving
#     verification to the user is a realistic mitigation.
#
# (d) A fallback pointing to hello@groundworkcoffee.com when the assistant
#     cannot answer, so a refusal ends somewhere useful.


# ==========================================
# Step 6: Reflection
# ==========================================
#
# 1. Lines of code: manual semantic RAG vs LlamaIndex.
# The manual version in the lesson needed roughly 60-80 lines to do what this
# project does in three: reading files, splitting text into chunks with
# overlap, batching chunks to the embeddings API, storing the vectors, writing
# a cosine similarity function, ranking results, assembling the prompt, and
# calling the LLM. Here it is SimpleDirectoryReader(...).load_data(),
# VectorStoreIndex.from_documents(...), and .as_query_engine(similarity_top_k=3).
#
# What that tells me is genuinely two-sided. The framework absorbs a lot of
# fiddly, error-prone plumbing -- off-by-one bugs in chunk overlap and
# mismatched embedding models between index and query are easy mistakes that
# LlamaIndex simply prevents. But the three lines hide every decision that
# determines retrieval quality: chunk size, overlap, which embedding model,
# how similarity is computed, what the prompt template says. I only understand
# what to tune -- and the fact that top-k always returns k results, which is
# the root of the Step 5 failure -- because the lesson made me build it by
# hand first. The framework is the right tool for production; building it
# manually once is what makes the framework's defaults legible rather than
# magic.
#
# 2. A different use case where this adds genuine value.
# Hospital clinical policy lookup. A large hospital has hundreds of protocol
# documents -- infection control, medication administration, escalation
# pathways, equipment procedures -- revised constantly and scattered across
# shared drives and binders. A nurse at 3am needing the current protocol for a
# specific situation is not going to find it quickly, and the cost of using an
# outdated version is real.
#
# This is a good RAG fit for the same reason the legal scenario in the warmup
# was: the documents change often, so fine-tuning is wrong, and there are far
# too many to paste into a prompt. It's also a domain where RAG's ability to
# cite its source matters more than the answer itself -- a clinician needs to
# see *which* protocol, revised *when*, not just a paragraph of prose. And it
# is precisely where the Step 5 lesson bites hardest: a confidently worded
# answer assembled from a superseded protocol would be more dangerous than no
# answer, so it would need a similarity floor, mandatory source display with
# revision dates, and scoping to answering-with-citation rather than
# free-form advice.
#
# 3. One failure mode RAG cannot fully prevent, even when retrieval works.
# Stale or wrong source documents. RAG grounds answers in the corpus, but it
# has no way to know whether the corpus is currently true. If Groundwork
# changed weekend closing to 4pm and faq.txt still says 5pm, the system
# retrieves the right document, generates a faithful answer, and confidently
# tells the user something false. Every quality signal reads green -- high
# similarity, faithfulness 1.0, correct citation -- because faithfulness
# measures agreement with the retrieved text, not agreement with reality.
#
# That is the failure mode I find most unsettling, because it is invisible to
# the evaluation tools from the warmup and it gets worse as the system is
# trusted more. It isn't fixable with better retrieval or a better model; it's
# a data governance problem -- document ownership, review dates, removing
# superseded material from the index. RAG moves the trust question from "is
# the model reliable?" to "is the corpus maintained?", which is more tractable,
# but only if someone is actually assigned to maintain it.
#
# A second one, which I did not fully appreciate until Step 5 produced it:
# recombination. Even with correct retrieval and a well-maintained corpus, the
# model can assemble a false statement entirely out of true fragments. My
# hallucinated "$5.00 large vanilla latte" used a real price, a real flavour
# word, and a size category from nowhere -- every ingredient traceable to the
# retrieved context, the combination invented. Retrieval quality cannot prevent
# this, because retrieval did its job; the failure happened downstream, in
# generation. It is also the hardest kind to catch automatically, since a naive
# check for "does this number appear in the context?" passes.
#
# A third: RAG struggles with questions requiring synthesis across many chunks,
# or aggregate reasoning ("which of our documents mention pricing?"). Top-k
# returns a handful of passages; it cannot reason over the corpus as a whole, so
# a question whose answer requires reading everything gets answered confidently
# from a fraction of the evidence.

print("\n=== End of project_06.py ===")