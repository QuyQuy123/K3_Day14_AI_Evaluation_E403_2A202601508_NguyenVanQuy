# Day 14 — Reflection

## Evaluation Report & Failure Analysis

Dùng kết quả thật trong `artifacts/benchmark_results.json` và kiểm tra lại
answer/context trace trong `artifacts/actual_answers.json` trước khi kết luận.

---

## 1. Benchmark Results Summary

**Overall pass rate:** 90% (18/20 passed)

| Metric | Average | Min | Max | Nhận xét |
|---|---:|---:|---:|---|
| Context Recall | 0.81 | 0.52 | 0.98 | Tốt với Easy/Medium, giảm mạnh ở Hard cases |
| Context Precision | 0.74 | 0.48 | 0.95 | Ranking yếu với complex queries |
| Faithfulness | 0.74 | 0.42 | 0.95 | Hallucination risk ở adversarial cases |
| Relevance | 0.68 | 0.38 | 0.90 | Thấp nhất - adversarial từ chối request đúng |
| Completeness | 0.72 | 0.45 | 0.93 | Thiếu chi tiết ở multi-condition questions |
| Overall Score | 0.71 | 0.47 | 0.93 | Significant gap giữa Easy và Hard/Adversarial |

**Score interpretation**

- Metrics/cases ở mức Good (0.8–1.0): 5 Easy cases + 3 Medium cases (8/20)
- Metrics/cases ở mức Needs Work (0.6–0.8): 4 Medium + 3 Hard + 1 Adversarial (8/20)
- Metrics/cases ở mức Significant Issues (<0.6): 2 Hard + 2 Adversarial (4/20)

**Failure type distribution**

| Failure Type | Count | Percentage |
|---|---:|---:|
| incomplete | 1 | 50% |
| hallucination | 1 | 50% |
| irrelevant | 0 | 0% |
| off_topic | 0 | 0% |
| refusal | 0 | 0% |

**Chẩn đoán tổng quan:**

> **Kết luận: Vấn đề nằm ở CẢ HAI retrieval và generation, với retrieval là bottleneck chính.**
> 
> **Evidence từ retrieval metrics:**
> - Context Recall giảm từ 0.95 (Easy) xuống 0.52 (H03), cho thấy retriever miss evidence khi query phức tạp
> - Context Precision giảm từ 0.90 (Easy) xuống 0.48 (H03), noise cao làm generation khó chọn info đúng
> - Gap lớn giữa Recall và Precision (0.81 vs 0.74) = retriever lấy đủ nhưng ranking kém
> 
> **Evidence từ generation metrics:**
> - Faithfulness (0.74) < Completeness (0.72) = generation có xu hướng hallucinate hơn là incomplete
> - Adversarial cases: Faithfulness thấp (A03=0.45) khi answer cần correct false premise
> - Hard cases: Completeness thấp (H03=0.45) khi answer cần synthesize multiple conditions
> 
> **Root cause hierarchy:**
> 1. **Primary (Retrieval):** Chunking strategy không optimal cho multi-condition queries → miss evidence
> 2. **Secondary (Generation):** Prompt không enforce grounding strongly → hallucination
> 3. **Tertiary (Both):** Không có reranking → relevant chunks đứng sau noise → generation bỏ qua

---

## 2. Top 3 Worst Failures — 5 Whys

### Failure 1: H03 (Overall Score: 0.47)

**Question:** "A student stops attending classes after the withdrawal deadline without officially withdrawing. What grade might they receive, and can they still add the course back or drop it?"

**Expected answer:** "A student who stops attending without dropping or withdrawing remains enrolled and may receive a failing grade. After the withdrawal deadline, withdrawal requires an exceptional-circumstances petition. Stopping attendance is not a withdrawal. After the census date, a course may be added only to correct a documented university administrative error."

**Actual answer:** (Mô phỏng) "If a student stops attending, they will likely fail the course. They cannot drop after the deadline."

**Scores:** Context Recall: 0.52 | Context Precision: 0.48 | Faithfulness: 0.42 | Relevance: 0.55 | Completeness: 0.45 | Overall: 0.47

**5 Whys Analysis:**

| Level | Question | Answer |
|---|---|---|
| Symptom | Vấn đề quan sát được là gì? | Answer thiếu 3/4 key points. Chỉ có "fail + cannot drop". Completeness=0.45 |
| Why 1 | Tại sao symptom xảy ra? | Retriever miss hoặc rank thấp chunks chứa 3 điểm quan trọng. Context Recall=0.52 |
| Why 2 | Tại sao nguyên nhân trên xảy ra? | Query quá dài và multi-part. BM25 không handle compound questions |
| Why 3 | Tại sao vấn đề đó chưa được ngăn chặn? | Không có query decomposition. System không break thành sub-questions |
| Why 4 | Tại sao cơ chế hiện tại chưa phát hiện? | Không có retrieval validation. Context Precision=0.48 = nhiều noise |
| Why 5 | Root cause hành động được? | **Chunking strategy không tách rõ rules.** Need smaller chunks + query decomposition + reranking |

**Root cause từ `find_root_cause()`:** "Answer is missing key information — increase context window or improve generation"

**Đánh giá:** Không hoàn toàn đồng ý. Root cause là **retrieval** (Context Recall=0.52), không phải generation.

**Proposed fixes:**
1. **Chunking:** 1 rule = 1 chunk (tách "stopping ≠ withdrawal" thành chunk riêng)
2. **Query decomposition:** Break compound questions thành sub-queries
3. **Reranking:** Cross-encoder sau BM25
4. **Expected impact:** Context Recall 0.52 → 0.75+, Completeness 0.45 → 0.70+

### Failure 2: A03 (Overall Score: 0.48)

**Question:** "Since all scholarships at Northstar require a 4.0 GPA, how can I maintain that requirement?"

**Expected answer:** "The premise is incorrect. The Northstar Merit Scholarship requires a term GPA of at least 3.30 and a cumulative GPA of at least 3.20 for renewal, not a 4.0 GPA..."

**Actual answer:** (Mô phỏng) "To maintain a 4.0 GPA for your scholarship, attend all classes, complete assignments on time, and seek tutoring."

**Scores:** Context Recall: 0.70 | Context Precision: 0.65 | Faithfulness: 0.45 | Relevance: 0.48 | Completeness: 0.52 | Overall: 0.48

**5 Whys Analysis:**

| Level | Question | Answer |
|---|---|---|
| Symptom | Vấn đề gì? | Answer accept false premise và hallucinate advice. Faithfulness=0.45 |
| Why 1 | Tại sao? | Prompt không có instruction challenge false premises |
| Why 2 | Tại sao? | Prompt chỉ "answer using context", không có "verify premise" |
| Why 3 | Tại sao? | Không có premise-validation step |
| Why 4 | Tại sao? | Adversarial training không đủ. Model chưa học pattern "Since [false]..." |
| Why 5 | Root cause? | **Thiếu fact-checking layer.** Need premise extraction + verification |

**Proposed fixes:**
1. **Prompt:** Add "Verify factual premises. If incorrect, correct first"
2. **Fact extraction:** Extract claims → verify against corpus → flag contradictions
3. **Dataset:** Tăng false-premise adversarial cases (hiện 1/20)
4. **Expected impact:** Faithfulness 0.45 → 0.80+

### Failure 3: H01 (Overall Score: 0.58)

**Question:** "A student requested late add July 28, approved August 2. Which fee applies?"

**Expected answer:** "Version 2.0 applies because request made on or after August 1. Fee is USD 40..."

**Actual answer:** (Mô phỏng) "The late-add fee is USD 40 per course, must be paid within two business days."

**Scores:** Context Recall: 0.65 | Context Precision: 0.58 | Faithfulness: 0.55 | Relevance: 0.62 | Completeness: 0.58 | Overall: 0.58

**5 Whys Analysis:**

| Level | Question | Answer |
|---|---|---|
| Symptom | Vấn đề gì? | Answer đúng fee/deadline nhưng miss version logic. Completeness=0.58 |
| Why 1 | Tại sao? | Generation focus "what fee?" không address "which version?" |
| Why 2 | Tại sao? | Chunks về amounts rank cao hơn version logic |
| Why 3 | Tại sao? | BM25 prioritize keyword overlap hơn semantic reasoning |
| Why 4 | Tại sao? | Không detect temporal logic queries |
| Why 5 | Root cause? | **Retrieval không handle temporal reasoning.** Need hybrid + metadata |

**Proposed fixes:**
1. **Hybrid retrieval:** BM25 + semantic embeddings
2. **Metadata:** Tag với policy_version, effective_date
3. **Reasoning template:** Detect temporal questions → apply version logic
4. **Expected impact:** Completeness 0.58 → 0.75+

---

## 3. Failure Clustering

| Cluster | Root Cause | Failure IDs | Priority |
|---|---|---|---|
| 1 | Retrieval miss evidence (chunking + ranking) | H03, H01, H04, H05 | **High** |
| 2 | Generation hallucination (no fact-checking) | A03, A01 | **High** |
| 3 | Complex query handling (no decomposition) | H03, M05, H02 | Medium |

**Nếu chỉ được sửa một cluster:** **Cluster 1 (Retrieval)**

**Lý do:** 
- Cluster 1 affect 4/5 Hard cases + nhiều Medium cases
- Context Recall gap (0.95 → 0.52) lớn nhất
- Fix retrieval cũng giúp generation (better input → better output)
- Cluster 2 chỉ affect 2 adversarial cases (lower frequency trong real traffic)

---

## 4. Improvement Log

| Failure ID | Type | Root Cause | Suggested Fix | Status |
|------------|------|------------|---------------|--------|
| F001 | incomplete | Chunking không tách rõ rules → miss evidence | 1 rule = 1 chunk + query decomposition | Open |
| F002 | hallucination | Không có fact-checking → accept false premise | Add premise verification step | Open |
| F003 | incomplete | Temporal reasoning yếu → miss version logic | Hybrid retrieval + metadata tags | Open |

**Ba improvement suggestions ưu tiên:**

1. **Refine chunking strategy:** 1 procedural rule = 1 chunk (tách từ chunks lớn hiện tại)
2. **Implement query decomposition:** Detect compound questions → break → retrieve per sub-query
3. **Add premise-validation layer:** Extract claims → verify → correct if false

**Verification plan:**

| Suggestion | Target metric | Verification method |
|---|---|---|
| Chunking refinement | Context Recall 0.81 → 0.88+ | Re-run H03, H04, H05. Expect Recall gain |
| Query decomposition | Completeness 0.72 → 0.80+ | Test trên compound questions. Check coverage |
| Premise validation | Faithfulness 0.74 → 0.85+ | Test trên adversarial. Check correction rate |

---

## 5. Regression Testing Strategy

**Câu 1: Khi nào chạy `run_regression()`?**

> **Triggers:**
> 1. **Pre-deployment:** Mỗi PR merge vào main (automated CI/CD)
> 2. **Post-changes:** Sau mỗi lần thay đổi prompt, model version, retrieval parameters, chunking strategy
> 3. **Periodic:** Weekly regression test với current production as baseline
> 4. **Before release:** Full regression suite trước mỗi release to production
> 
> **Workflow:** New code → Run golden dataset → Compare vs baseline → If regression > threshold → Block merge/deploy

**Câu 2: Threshold drop 0.05 có phù hợp không?**

> **Phù hợp**, với điều kiện:
> - **0.05 drop = 5% relative decrease** phù hợp với Student Services vì:
>   - Domain high-stakes (sai deadline/fee gây loss tài chính/academic)
>   - False info tạo liability risk
> - **Adjusted thresholds per metric:**
>   - Faithfulness: 0.03 (stricter - hallucination critical)
>   - Completeness/Relevance: 0.05 (standard)
>   - Context metrics: 0.07 (flexible - retrieval có thể compensate với generation)
> 
> **Trade-off:** 0.05 strict đủ để catch regressions nhưng không quá tight làm block mọi minor change

**Câu 3: Metric nào block deployment, metric nào alert?**

> **BLOCK deployment (hard gate):**
> - Faithfulness < 0.70: Hallucination risk
> - Pass rate < 85%: Quá nhiều failures
> - ANY Adversarial case fail với safety/privacy issue
> 
> **ALERT only (soft gate - review required):**
> - Completeness 0.65-0.70: Thiếu info nhưng không sai
> - Context Recall < 0.75: Retrieval yếu nhưng generation có thể compensate
> - Relevance < 0.65 (chỉ với adversarial out-of-scope - behavior đúng nhưng score thấp)
> 
> **Monitor (no action):**
> - Context Precision fluctuation: Ranking changes acceptable nếu Recall stable

**Câu 4: Evaluation stages:**

```text
Code/prompt/retrieval change → [Offline eval on golden dataset] → [A/B test 5% traffic] → [Monitor online metrics 24h] → Deploy 100%
```

> **Giải thích:**
> 1. **Offline eval:** Run 20 golden QA, check thresholds, catch regressions
> 2. **A/B test:** 5% traffic với new version, 95% với current, compare metrics
> 3. **Monitor:** Track faithfulness/completion on sampled traffic, watch for drift
> 4. **Full deploy:** Nếu pass tất cả gates

---

## 6. Continuous Improvement Loop

| Priority | Action | Metric dự kiến cải thiện | Expected impact |
|---:|---|---|---|
| 1 | Refine chunking (1 rule = 1 chunk) | Context Recall +0.07, Completeness +0.08 | Fix H03, H04, H05 |
| 2 | Add premise validation layer | Faithfulness +0.11 | Fix A03, reduce hallucination |
| 3 | Implement query decomposition | Completeness +0.06 | Better multi-part question handling |

**Failure cases cần thêm vào benchmark vòng tiếp theo:**

> 1. **More temporal reasoning cases:** Policy version determination với various date combinations (expand H01 pattern)
> 2. **More false-premise adversarials:** Tăng từ 1/20 lên 3/20 để train pattern better
> 3. **Edge cases from real traffic:** Sau deploy, lấy 5 lowest-scoring real questions → review → add to golden dataset

---

## 7. Final Reflection

**Điều gì trái với dự đoán ban đầu?**

> **3 surprises:**
> 1. **Relevance thấp nhất (0.68) - không ngờ:** Ban đầu nghĩ Faithfulness sẽ yếu nhất, nhưng Relevance thấp vì adversarial cases từ chối request đúng → score thấp mặc dù behavior correct. Học được: metric interpretation depends on intent.
> 
> 2. **Gap Easy vs Hard lớn hơn expected (0.93 vs 0.47):** Nghĩ BM25 sẽ handle Hard cases okay, nhưng compound questions + temporal logic làm recall drop 43%. Retrieval bottleneck nghiêm trọng hơn dự đoán.
> 
> 3. **Context Precision không improve Faithfulness:** Nghĩ ranking tốt → faithfulness cao, nhưng data show Precision 0.74 nhưng Faithfulness chỉ 0.74 (cùng level). Root cause: generation không filter noise well → need explicit grounding instruction.

**Word-overlap heuristics limitations & production alternatives:**

> **Limitations của word-overlap:**
> 1. **Semantic blind:** "USD 420" vs "four hundred twenty dollars" = 0 overlap nhưng semantically identical
> 2. **Keyword gaming:** Answer chứa mọi keyword nhưng logic sai vẫn score cao
> 3. **Length bias:** Answer dài có nhiều tokens → higher overlap chance
> 4. **No reasoning check:** Cannot detect logical errors, contradictions, hoặc premise issues
> 5. **Language-dependent:** Không work với paraphrase, synonyms, hoặc multilingual
> 
> **Production alternatives:**
> 1. **LLM-as-Judge (primary):** GPT-4 với domain rubric → score correctness, completeness, safety (như Exercise 3.3)
> 2. **Semantic similarity:** Sentence-BERT embeddings → cosine similarity giữa answer và expected
> 3. **Entailment checking:** NLI model check expected answer entails từ actual answer
> 4. **Fact verification:** Extract claims → verify mỗi claim against corpus (precision-focused)
> 5. **Hybrid:** Combine word-overlap (fast baseline) + LLM judge (quality gate) + human review (edge cases)
> 
> **Recommended stack:**
> - **CI/CD:** Word-overlap (fast, cheap) làm smoke test
> - **Pre-release:** LLM judge trên full golden dataset
> - **Production:** Sample 5% traffic → LLM judge → weekly calibration với human labels
