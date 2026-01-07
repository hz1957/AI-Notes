# Agent Architecture Patterns

## 1. “685B + 小模型协同”架构 (For SQL / ETL)

### 核心目标
让 **685B** 只做最值钱的推理，把高频、低难度、结构化的部分交给 **小模型/规则/工具**，显著减少：
1.  685B 的调用次数。
2.  每次调用的上下文长度。
3.  长上下文 Prefill 次数。

---

### 推荐架构图
```mermaid
flowchart LR
  UI[User Chat UI] --> ORCH[Orchestrator<br/>Conversation State + Budget]
  ORCH --> SM[Small Model (7B~32B)<br/>Intent + Plan Draft + Slot Fill]
  SM --> RET[Schema Retriever<br/>(BM25/Embedding)]
  RET --> PLAN[Schema-aware Plan JSON<br/>tables/fields/joins]
  PLAN --> UI_CONFIRM[User Confirm / Edit]
  UI_CONFIRM -->|approved| BIG[DeepSeek 685B<br/>Final SQL + Hard Reasoning]
  BIG --> VALID[SQL Validator<br/>Parser + Static checks]
  VALID --> EXEC[DB Sandbox<br/>EXPLAIN / LIMIT 10]
  EXEC --> FIX[Error/Trace to Fix Loop<br/>(Small model first)]
  FIX -->|needs deep reasoning| BIG
  FIX -->|simple fix| SM
  EXEC --> OUT[Result + Explanation]

  subgraph "Key Principle: Offload Context"
    SM
    RET
    PLAN
  end
```

### 核心策略 (Key Principles)

#### 1. "Schema-aware Plan" 放在小模型侧
不要直接让 685B 从全量 Schema 里挑表。
*   **做法**：让小模型 + Retriever 生成一个 `Plan JSON`（包含相关的 top-K 表、字段建议）。
*   **收益**：685B 永远只看到经过筛选的、极小的 Schema 子集，**避免了每次都 Prefill 40k+ tokens**。

#### 2. “确认后”再调用大模型
把“用户需求确认”前置。
*   **做法**：小模型生成初步计划后，让用户确认/修改（UI 层面）。只有用户点了“Proceed”，才把清洗好的 Context 喂给 685B。
*   **收益**：消除了大量因需求模糊导致的 685B 无效调用。

#### 3. 分级修错 Loop
*   **90% 的错误**：字段名拼错、SQL 语法微小错误、Join 键类型不匹配。
    *   **处理**：交给小模型看 Error Log 即可修好，无需动用 685B。
*   **10% 的错误**：逻辑错误、业务理解偏差（窗口函数用错、嵌套层级不对）。
    *   **处理**：才 escalate 给 685B。

### 落地建议（立刻能做的两步）
1.  **降并非**：把并发 agent 从 20 降到 **2~4**（针对长上下文调用 685B 的那一段）。
2.  **Schema 预处理**：把“选表”任务剥离给小模型 + 检索，685B 只在用户确认 Plan 后介入。

这两步通常就能把“没排队但很慢”的现象明显改善。
