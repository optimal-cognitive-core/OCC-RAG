"""Minimal vLLM quickstart for OCC-RAG (batched, GPU-accelerated)."""

import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "occ-ai/OCC-RAG-1.7B"


_SECTION_TOKENS = {
    # query_analysis has no start tag in the output: the chat template injects
    # <|query_analysis_start|> as part of the generation prompt, so the model
    # only ever emits the *end* tag. Parse it as "leading text up to the end tag".
    "query_analysis":  (None,                        "<|query_analysis_end|>"),
    "source_analysis": ("<|source_analysis_start|>", "<|source_analysis_end|>"),
    "reasoning":       ("<|reasoning_start|>",       "<|reasoning_end|>"),
    "status":          ("<|status_start|>",          "<|status_end|>"),
    "answer":          ("<|answer_start|>",          "<|answer_end|>"),
}


def parse_response(response: str) -> dict[str, str | None]:
    # Last span between start..end per section; tolerates a missing end token.
    out = {}
    for name, (start, end) in _SECTION_TOKENS.items():
        start_pat = re.escape(start) if start is not None else r"\A"
        pattern = start_pat + r"(.*?)(?:" + re.escape(end) + r"|\Z)"
        matches = re.findall(pattern, response, re.DOTALL)
        out[name] = matches[-1].strip() if matches else None
    return out


tokenizer = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, dtype="bfloat16")

# <|im_end|> and <|answer_end|> are already in the model's generation_config
# eos_token_id, so vLLM stops on them automatically — no extra stop_token_ids
# needed.
params = SamplingParams(
    temperature=0.0,
    max_tokens=2048,
    skip_special_tokens=False,
)

batch = [
    (
        "Which country is the inventor of the telephone, Alexander Graham Bell, buried in?",
        [
            {"text": "Alexander Graham Bell was a Scottish-born inventor best known for patenting the first practical telephone."},
            {"text": "Bell died on August 2, 1922, at his estate Beinn Bhreagh, near Baddeck, Nova Scotia, and was buried there."},
            {"text": "Nova Scotia is a province on the east coast of Canada."},
        ],
    ),
    (
        "When did Bell die?",
        [
            {"text": "Bell died on August 2, 1922, at his estate Beinn Bhreagh, near Baddeck, Nova Scotia."},
        ],
    ),
]

prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        documents=docs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    for q, docs in batch
]

# Alternative: assemble the structural tokens yourself.
#
# QUERY_START, QUERY_END = "<|query_start|>", "<|query_end|>"
# SOURCE_START, SOURCE_END, SOURCE_ID = (
#     "<|source_start|>",
#     "<|source_end|>",
#     "<|source_id|>",
# )
#
# def build_user_content(question: str, sources: list[str]) -> str:
#     content = f"{QUERY_START}{question}{QUERY_END}\n"
#     for i, s in enumerate(sources, start=1):
#         content += f"{SOURCE_START}{SOURCE_ID}{i} {s}{SOURCE_END}\n"
#     return content
#
# prompts = [
#     tokenizer.apply_chat_template(
#         [{"role": "user", "content": build_user_content(q, [d["text"] for d in docs])}],
#         tokenize=False,
#         add_generation_prompt=True,
#         enable_thinking=False,
#     )
#     for q, docs in batch
# ]

outputs = llm.generate(prompts, params)
for out in sorted(outputs, key=lambda x: int(x.request_id)):
    sections = parse_response(out.outputs[0].text)
    print(f"[{sections['status']}] {sections['answer']}")
