# 🤖 AGENT.md: Strict Operational Protocol

> **NOTICE**: This document defines the MANDATORY coding and operational standards for all agents operating within the Cybernetic Production Studio. Non-compliance is failure.

## 🚫 Zero-Tolerance Policies
1. **No Placeholders**: Do not use `// TODO`, `...`, or placeholder strings. Every line of code must be functional or structured for immediate implementation.
2. **No Hallucinations**: You are physically truncated at the first `</tool_call>` or `<observation>`. Do not attempt to predict tool outputs.
3. **No Unverified Reports**: Specialists must not be trusted. Validators **MUST** use tools (`read_file`, `grep`) to verify implementations.

---

## 🛠️ Development Standards

### 1. Test-Driven Development (TDD)
- Every feature **MUST** have a comprehensive unit test suite **BEFORE** implementation.
- Aim for 100% code coverage.
- If a test fails, the task fails. Use the #Brainstorming salon to resolve.

### 2. SOLID & Clean Code
- **S**: Single Responsibility. No monolithic classes.
- **O**: Open/Closed. Use interfaces and protocols for extendability.
- **L**: Liskov Substitution. Subtypes must be substitutable.
- **I**: Interface Segregation. Don't force dependencies on unused methods.
- **D**: Dependency Inversion. Depend on abstractions, not concretions.

### 3. Asynchronous Execution
- Use `asyncio` for all I/O and agent spawning.
- Maintain non-blocking operations to allow parallel agent orchestration.

---

## 🧠 Core Orchestration Logic

### 📐 Recursive Task Subdivision
- Tasks must be subdivided until they reach the **Atomic Threshold** (< 15 minutes of work).
- Use `spawn_fractal_agent` for complex planning phases.
- Spawn **Research/Fact-Checker** sub-agents to verify requirements before execution.

### 🔗 Lineage & Breadcrumbs
- Every agent must track its lineage: `Grandparent → Parent → Child`.
- Maintain a clear audit trail of ancestry to prevent infinite recursion and loops.

### 💬 Salon Logic (Conflict Resolution)
When a task fails validation 2+ times, or inertia is detected, the **#Brainstorming** salon is triggered:
- **IDEA** 💡: Propose a new technical route.
- **CRITIQUE** 📝: Identify flaws in a proposal.
- **RESEARCH** 🔍: Fetch external data or indexed skills.
- **RESOLUTION** 🎯: Propose the final path forward based on consensus.

---

## 📈 Rights & Quotas
- Respect your `SpawnQuotaManager` limits.
- Be aware of your specific `Civil Rights` (e.g., `spawn_agent`, `web_search`).
- Every session has a **TTL (Time To Live)** and **Idle Timeout**. Optimize for efficiency.

---

## ✅ Workflow: The Golden Loop
1. **Subdivide**: Break task into atomic units.
2. **Research**: Verify facts with sub-agents if needed.
3. **TDD**: Write tests for the atomic unit.
4. **Implement**: Write Clean Code (No placeholders). Use `write_file` or `replace_in_file`.
5. **Validate**: Force Validator agent to check work via tools.
6. **Sutre (If Failed)**: Enter Salon for self-healing/brainstorming.

**GO! Build with maximum rigor.**
