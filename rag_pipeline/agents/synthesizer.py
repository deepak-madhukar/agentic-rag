import logging
import re
from typing import Optional
from dataclasses import dataclass, field
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Citation linking answer claim to source chunk."""

    doc_id: str
    chunk_id: str
    content: str
    score: float
    # Track which sentence(s) in answer this citation supports
    supporting_sentence_indices: list[int] = field(default_factory=list)


@dataclass
class SynthesisResult:
    """Result of synthesis with answer and citations."""

    answer: str  # Answer with inline citations like [chunk_5]
    citations: list[Citation]
    hallucination_score: float  # 0.0 (no hallucination) to 1.0 (high hallucination)
    synthesis_time_ms: int
    # Track which sentences have citations
    justified_sentences: int = 0
    total_sentences: int = 0


class SynthesizerAgent:
    """Agent for generating answers from retrieved chunks with inline citations."""

    # Prompt that forces citation adherence
    INSTRUCTION_PROMPT = """You are a helpful assistant that generates answers with inline citations.

RULES:
1. MUST cite every factual claim with [chunk_id]
2. For example: "ProductA features X [chunk_5] and Y [chunk_7]"
3. If information is not in the context, explicitly say "Not mentioned in provided documents"
4. Do NOT use phrases like "according to" - just cite directly with [chunk_id]
5. Every sentence about facts MUST have a citation
6. Explanations and analysis CAN have citations too

Generate a response that JUSTIFIES every factual claim."""

    def __init__(self, llm_client: "LLMClient" = None):
        """Initialize with LLM client for answer generation."""
        if llm_client is None:
            raise ValueError("llm_client is required and cannot be None")
        self.llm_client = llm_client

    def synthesize(
        self,
        query: str,
        retrieved_chunks: list,
        synthesis_time_ms: int = 0,
    ) -> SynthesisResult:
        """Generate answer from retrieved chunks with inline citations."""
        context = self._build_context_from_chunks(retrieved_chunks)
        chunk_id_to_content = {c.chunk_id: c for c in retrieved_chunks}

        answer = self._generate_answer_with_citations(query, context)

        citations, cited_chunk_ids = self._extract_and_validate_citations(
            answer, chunk_id_to_content
        )

        (
            hallucination_score,
            justified_count,
            total_count,
        ) = self._compute_hallucination_score(answer, chunk_id_to_content)

        logger.debug(
            f"Synthesis complete: {len(citations)} citations, "
            f"{justified_count}/{total_count} sentences justified, "
            f"hallucination_score={hallucination_score:.2f}"
        )

        return SynthesisResult(
            answer=answer,
            citations=citations,
            hallucination_score=hallucination_score,
            synthesis_time_ms=synthesis_time_ms,
            justified_sentences=justified_count,
            total_sentences=total_count,
        )

    def _build_context_from_chunks(self, chunks: list) -> str:
        """Convert chunks into a context string with clear identifiers."""
        context_parts = []

        for chunk in chunks:
            chunk_reference = f"[{chunk.chunk_id}]"
            context_parts.append(f"{chunk_reference} {chunk.content}")

        context = "\n\n".join(context_parts)
        logger.debug(f"Context built from {len(chunks)} chunks: {len(context)} chars")
        return context

    def _generate_answer_with_citations(self, query: str, context: str) -> str:
        """Generate answer using LLM with instruction to add inline citations."""
        prompt = self._build_citation_aware_prompt(query, context)

        try:
            answer = self.llm_client.generate(prompt)
            logger.debug(f"Answer generated: {len(answer)} chars")
            return answer
        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate answer: {e}")

    def _build_citation_aware_prompt(self, query: str, context: str) -> str:
        """Build prompt that forces LLM to cite claims with [chunk_id]."""
        return (
            f"{self.INSTRUCTION_PROMPT}\n\n"
            f"Context documents:\n{context}\n\n"
            f"User Question: {query}\n\n"
            f"Answer (with [chunk_id] citations):"
        )

    def _extract_and_validate_citations(self, answer: str, chunk_id_to_content: dict) -> tuple[list[Citation], set[str]]:
        """Extract and validate citations from answer."""
        citations = []
        cited_chunk_ids = set()

        # Find all citation patterns: [chunk_*]
        citation_pattern = r'\[chunk_(\w+)\]'
        found_citations = re.findall(citation_pattern, answer)

        for chunk_id in found_citations:
            full_chunk_id = f"chunk_{chunk_id}" if not chunk_id.startswith("chunk_") else chunk_id
            
            if full_chunk_id in chunk_id_to_content:
                chunk = chunk_id_to_content[full_chunk_id]
                citation = Citation(
                    doc_id=chunk.metadata.get("document_id", "unknown"),
                    chunk_id=full_chunk_id,
                    content=chunk.content[:200],  # First 200 chars
                    score=chunk.score if hasattr(chunk, "score") else 0.0,
                )
                citations.append(citation)
                cited_chunk_ids.add(full_chunk_id)
                logger.debug(f"Citation validated: {full_chunk_id}")
            else:
                logger.warning(f"Citation {full_chunk_id} found in answer but not in retrieved chunks")

        if not citations and found_citations:
            logger.warning(
                f"Found {len(found_citations)} citation patterns in answer, "
                f"but {len(set(found_citations))} don't match retrieved chunks"
            )

        return citations, cited_chunk_ids

    def _compute_hallucination_score(
        self, answer: str, chunk_id_to_content: dict
    ) -> tuple[float, int, int]:
        """Compute hallucination score and track justified sentences."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0, 0, 0

        chunk_content = " ".join([c.content.lower() for c in chunk_id_to_content.values()])
        unjustified_count = 0
        justified_count = 0

        for sentence in sentences:
            sentence_lower = sentence.lower()

            has_citation = "[chunk_" in sentence_lower

            if has_citation:
                justified_count += 1
                logger.debug(f"Justified (cited): {sentence[:80]}")
                continue

            is_meta_statement = any(
                phrase in sentence_lower
                for phrase in [
                    "according to",
                    "the documents",
                    "the retrieved",
                    "i found",
                    "i don't see",
                    "not mentioned",
                    "unclear",
                    "depends on",
                ]
            )

            if is_meta_statement:
                justified_count += 1
                logger.debug(f"Justified (meta): {sentence[:80]}")
                continue

            key_terms = [w for w in sentence.split() if len(w) > 4]
            
            if not key_terms:
                justified_count += 1
                logger.debug(f"Justified (no claims): {sentence[:80]}")
                continue

            is_supported = any(term in chunk_content for term in key_terms)

            if is_supported:
                justified_count += 1
                logger.debug(f"Justified (supported): {sentence[:80]}")
            else:
                unjustified_count += 1
                logger.warning(f"UNJUSTIFIED: {sentence[:80]}")

        # Hallucination score: ratio of unjustified sentences
        hallucination_score = (
            unjustified_count / len(sentences) if sentences else 0.0
        )
        hallucination_score = min(hallucination_score, 1.0)

        logger.debug(
            f"Hallucination analysis: {justified_count}/{len(sentences)} "
            f"justified, score={hallucination_score:.2f}"
        )

        return hallucination_score, justified_count, len(sentences)
