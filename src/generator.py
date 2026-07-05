import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from src.chunking import Chunk


class RAGGenerator:
    """Wraps a causal language model for retrieval-augmented generation (RAG).

    Loads the model and tokenizer once at construction time and exposes a
    single ``generate_answer`` method that combines retrieved context chunks
    with a user question to produce a grounded answer.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Load the tokenizer and causal language model.

        Args:
            model_name: HuggingFace model identifier to load.  Defaults to
                ``"Qwen/Qwen3-0.6B"``.  The model is placed on CUDA when
                available, otherwise on CPU.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            f"Loading {self.model_name} on {self.device}"
            " (This might take a minute the first time)..."
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load model efficiently
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=(
                torch.float16 if torch.cuda.is_available() else torch.float32
            ),
            device_map=("auto" if torch.cuda.is_available() else None),
            low_cpu_mem_usage=True,
        )

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Chunk],
        max_new_tokens: int = 512,
    ) -> str:
        """Generate an answer grounded in the provided context chunks.

        Context chunks are concatenated up to a 10 000-character limit to stay
        within the model's effective token budget.  The model is instructed to
        answer solely from the supplied context and to respond with a fixed
        refusal message when the context is insufficient.

        Args:
            question: The natural-language question to answer.
            retrieved_chunks: Ordered list of ``Chunk`` objects to use as
                context.  Chunks beyond the character budget are silently
                dropped.
            max_new_tokens: Maximum number of new tokens to generate
                (default: 512).

        Returns:
            The model's response as a stripped string.
        """
        # Combine chunks into context; cap to avoid exceeding token budget.
        context_parts: List[str] = []
        current_len = 0

        for c in retrieved_chunks:
            chunk_str = f"Source: {c.file_path}\n{c.text}\n"
            # 10 000 chars ≈ 2 000 tokens — stay within model budget.
            if current_len + len(chunk_str) > 10000:
                break
            context_parts.append(chunk_str)
            current_len += len(chunk_str)

        context_text = "\n".join(context_parts)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful coding assistant. "
                    "Answer the user's question based ONLY on the provided"
                    " context. "
                    "If the context does not contain enough information to"
                    " answer the question, respond with exactly: "
                    "'I don't have enough information in the provided context"
                    " to answer this question.' "
                    "Do NOT use your training knowledge to fill gaps. "
                    "Do NOT guess or infer beyond what is explicitly stated"
                    " in the context. "
                    "Keep your answer clear, self-contained, and faithful"
                    " to the source."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context information is below.\n"
                    f"---------------------\n"
                    f"{context_text}\n"
                    f"---------------------\n"
                    f"Given the context information, answer the"
                    f" following question: {question}"
                ),
            },
        ]

        # Qwen uses chat templates
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Ensure attention mask is returned; move tensors to correct device.
        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        model_inputs = {
            name: tensor.to(self.device)
            for name, tensor in tokenized.items()
        }

        generated = self.model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

        # Trim the prompt tokens from the outputs
        input_len = model_inputs["input_ids"].shape[1]
        generated_ids = [output[input_len:].tolist() for output in generated]

        response: str = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return response.strip()
