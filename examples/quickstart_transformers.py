"""Minimal Transformers quickstart for OCC-RAG."""

import re

from transformers import AutoModelForCausalLM, AutoTokenizer

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
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype="auto", device_map="auto"
)

question = "Which country is the inventor of the telephone, Alexander Graham Bell, buried in?"
documents = [
    {"text": "Alexander Graham Bell was a Scottish-born inventor best known for patenting the first practical telephone."},
    {"text": "Bell died on August 2, 1922, at his estate Beinn Bhreagh, near Baddeck, Nova Scotia, and was buried there."},
    {"text": "Nova Scotia is a province on the east coast of Canada."},
]

text = tokenizer.apply_chat_template(
    [{"role": "user", "content": question}],
    documents=documents,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

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
# sources = [d["text"] for d in documents]
# text = tokenizer.apply_chat_template(
#     [{"role": "user", "content": build_user_content(question, sources)}],
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=False,
# )

inputs = tokenizer([text], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048)
response = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=False
)
print(response)

sections = parse_response(response)
print("\nStatus:", sections["status"])   # -> ANSWERABLE
print("Answer:", sections["answer"])     # -> Canada
