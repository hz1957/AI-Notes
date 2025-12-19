# Practical Lessons in Agent Development

> Based on real-world engineering experiences in Clinical Data ETL and Multi-Agent Systems.

## 1. Framework Selection: AutoGen vs. LangGraph

### AutoGen (Multi-Agent Conversation)
*   **Use Case**: "Deep Research" systems, literature review, multi-role brainstorming.
*   **Experience**:
    *   Good for simulating conversation between roles (e.g., "User" vs "Analyst").
    *   Built a system testing GPT-4o, Qwen, and Gemma.
    *   **Pro**: Easy to set up multi-agent dialogue.
    *   **Con**: Hard to control precise state transitions or enforce strict output formats for downstream engineering tasks.

### LangGraph (State Machine / Workflows)
*   **Use Case**: TSP Pricing Systems, Complex ETL Pipelines.
*   **Experience**:
    *   Better suited for defined business logic (Rules + AI Hybrid).
    *   Implemented workflow: `Markdown Parsing -> Info Extraction -> DB Query -> Formula calc`.
    *   **Key Advantage**: Fine-grained control over state (Graph State) and transitions.

## 2. Case Study: Agentic ETL System

Building an AI system that converts natural language requirements into executable ETL JSON configurations.

### Architecture Evolution
1.  **Linear Chain**: Initially assumed simple linear steps. Failed to handle complex dependencies.
2.  **DAG-based Planner**:
    *   **Planner**: Generates a high-level plan.
    *   **State Management**: Moved to a **DAG (Directed Acyclic Graph)** structure.
    *   **Mechanism**: Tracks "Available Variables" and "Upstream Nodes" dynamically. This allows the Agent to know exactly which fields are available at any step of the transformation.
    *   **Debug**: Implemented "snapshotting" (`dag_state.json`) to trace the agent's reasoning process.

### Engineering Challenges
*   **Hallucinations in SQL**: Agents would select columns that don't exist or hallucinate SQL syntax.
    *   *Fix*: Explicitly feed the current node's schema output as part of the prompt state.
    *   *Fix*: Added **"Watchdogs"** (Guardrails) to validate SQL/JSON structure before execution.
*   **JSON Output Stability**: Large ETL configurations (>1300 lines) caused JSON truncation or syntax errors.
    *   *Fix*: Strict prompt constraints (forbidding Markdown blocks).
    *   *Fix*: "Few-shot" examples specifically for complex operators like Window Functions.

## 3. RAG vs. Context Engineering

*   **Medical Knowledge Graph**: Integrated **Neo4j** for structured entity mapping (SDTM standards), enabling precise queries (CQL) where Vector Search (FAISS) was too fuzzy/inaccurate.
*   **Context Optimization**:
    *   Explored "Context Compression" similar to **Claude Code**: Instead of dumping all files into context, simulate a "grep" process where the agent explores the file tree first, then reads specific files.

## 4. Prompt Engineering Patterns for Reliability

*   **Double-Layer Interaction**:
    *   **Layer 1 (Clarification)**: Agent converses with user to clarify requirements (Ambiguity Resolution).
    *   **Layer 2 (Execution)**: Once requirements are locked, `Planner` and `Executor` take over to generate code.
*   **Structured Constraints**:
    *   Instead of letting the LLM generate the full final JSON directly, split it into atomic "Actions" or "Steps" that a deterministic code engine then assembles.
