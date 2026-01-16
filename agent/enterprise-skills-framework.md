# Enterprise Agent Skills Framework
> **Core Philosophy**: Standardizing and modularizing enterprise business processes not just as AI Prompts, but as executable assets that are "business-defined, developer-maintained, and audit-compliant".

## 1. What is an Enterprise Skill?
A Skill is not merely a prompt; it is a **standardized engineering package** designed to endow AI with specific business execution capabilities.

### 📂 Standard Structure
A standard Skill package typically includes the following components:

*   **`SKILL.md` (Core Documentation)**
    *   **Metadata**: Name, version, owner.
    *   **Operating Manual**: Detailed step-by-step logic.
    *   **Business Rules**: Decision criteria, exception handling, compliance requirements.
*   **`scripts/` (Execution Scripts)**
    *   Python/Bash scripts for API calls, data validation, or complex calculations.
    *   *Purpose: To compensate for the instability of pure text reasoning and ensure precise execution.*
*   **`resources/` (Business Resources)**
    *   **Templates**: Contract templates, email templates.
    *   **Sample Data**: Few-shot examples (positive/negative samples).
    *   **Checklists**: SOP (Standard Operating Procedure) checklists.
*   **`tests/` (Optional)**
    *   Test cases for validating Skill logic.

---

## 2. Roles & Responsibilities

### 👔 Business Experts — Responsible for "What & Why"
*Similar to "Product Managers" or "Business Analysts"*
*   **Define Processes**: Write business logic and judgment criteria in `SKILL.md` (e.g., 5 red lines for contract approval).
*   **Provide Samples**: Organize real-world best/worst cases into the `resources` directory.
*   **Acceptance Testing**: Confirm that AI execution results meet business standards.

### 💻 Developers/Engineers — Responsible for "How & Stability"
*Responsible for engineering, toolchain, and auditing*
*   **Script Development**: Write code in `scripts/` to interface with internal APIs or databases.
*   **Code Review**: Audit Skills for logical flaws and security issues (preventing Prompt Injection).
*   **Version Management**: Maintain Git repositories and manage Skill version iteration and rollbacks.

---

## 3. Orchestration & Workflow

Enterprises should treat Skills as **software assets**, incorporating them into standard R&D/Audit lifecycles:

### 🔄 Lifecycle Management
1.  **Drafting**
    *   Business personnel fill out or submit Markdown documentation online.
2.  **Review**
    *   **Business Review**: Senior experts confirm process compliance.
    *   **Technical Review**: Developers perform Pull Request Reviews to ensure safety.
3.  **Deployment**
    *   Deploy approved Skills to the AI platform/knowledge base via CI/CD pipelines.
4.  **Execution**
    *   AI Agents dynamically load the corresponding Skill modules to execute tasks based on user intent.

### 🧩 Composition & Reuse
*   **Atomic**: Each Skill does only one thing (e.g., "Check Inventory", "Contract Comparison").
*   **Chaining**: Through this framework, multiple atomic Skills can be chained (e.g., Extract Info Skill -> Risk Assessment Skill -> Generate Report Skill) to form complex business loops.
