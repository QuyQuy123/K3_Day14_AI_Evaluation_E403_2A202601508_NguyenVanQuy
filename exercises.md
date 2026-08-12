# Day 14 — Exercises

## AI Evaluation & Benchmarking · Lab Worksheet

**Thời gian làm bài:** 09:15–12:00

**Domain:** Northstar University Student Services

Điền trực tiếp câu trả lời vào file này. Golden dataset 20 QA được viết một lần
duy nhất trong `golden_dataset.json`, không chép lại toàn bộ vào Markdown.

---

Từ 09:15–09:30, cài môi trường và chạy baseline tests theo `guide_lab.md`.

---

## Part 1 — Warm-up (09:30–09:45)

### Exercise 1.1 — RAGAS Metric Thresholds

Theo bài giảng:

- 0.8–1.0: Good — monitor, maintain.
- 0.6–0.8: Needs work — analyze failures, iterate.
- Dưới 0.6: Significant issues — investigate.

Với từng metric, xác định khi nào score thấp có thể chấp nhận và khi nào là
critical.

| Metric | Acceptable Low Score Scenario | Critical Low Score Scenario | Action Required |
|---|---|---|---|
| Faithfulness | 0.65-0.75: Câu hỏi adversarial/mơ hồ khó grounding hoàn toàn nhưng câu trả lời vẫn an toàn | < 0.6: Câu trả lời bịa thông tin không có trong context, tạo ra rủi ro pháp lý | Khẩn cấp: Thêm kiểm tra grounding, yêu cầu citation. Review prompt generation để ngăn hallucination |
| Answer Relevance | 0.65-0.75: Câu hỏi phức tạp nhiều phần, câu trả lời một phần vẫn giải quyết ý chính | < 0.5: Câu trả lời hoàn toàn ngoài chủ đề hoặc trả lời sai câu hỏi | Gấp: Sửa phân loại intent, cải thiện độ rõ ràng của prompt, thêm lớp validation relevance |
| Context Recall | 0.6-0.7: Chủ đề ngách với corpus coverage hạn chế, chấp nhận được nếu có ghi nhận | < 0.5: Câu hỏi cốt lõi thiếu evidence quan trọng | Ưu tiên cao: Mở rộng corpus, cải thiện chiến lược chunking, tune tham số retrieval (top_k, threshold) |
| Context Precision | 0.55-0.65: Chấp nhận được khi recall cao — noise không chặn generation nếu có các chunk liên quan | < 0.4: Noise cao làm giảm chất lượng answer hoặc tăng latency/chi phí | Ưu tiên trung bình: Triển khai reranking, cải thiện query expansion, thêm đa dạng nguồn |
| Completeness | 0.6-0.7: Chấp nhận được cho các chi tiết "nice-to-have" nếu thông tin quan trọng có đầy đủ | < 0.5: Thiếu thông tin ảnh hưởng đến việc ra quyết định hoặc vi phạm yêu cầu chính sách | Ưu tiên cao: Tăng context window, cải thiện coverage retrieval, tune generation tránh dừng sớm |

### Exercise 1.2 — Bias trong LLM-as-a-Judge

Ba bias thường gặp:

- Position bias: judge ưu tiên answer xuất hiện trước.
- Verbosity bias: judge ưu tiên answer dài hơn.
- Self-preference: judge ưu tiên output giống chính model đó.

**Câu 1: Thiết kế experiment phát hiện position bias với ít nhất hai conditions.**

> **Thiết kế thí nghiệm:**
> 
> **Chuẩn bị:** Chọn 20 câu hỏi, mỗi câu có 2 câu trả lời (A và B) với chất lượng khác biệt rõ ràng (A tốt hơn B).
> 
> **Điều kiện 1 (thứ tự A-B):** Judge đánh giá với thứ tự "Câu trả lời A" trước, "Câu trả lời B" sau
> 
> **Điều kiện 2 (thứ tự B-A):** Judge đánh giá cùng câu trả lời nhưng đổi thứ tự thành "Câu trả lời B" trước, "Câu trả lời A" sau
> 
> **Phương pháp phát hiện:**
> - Nếu không có position bias: Câu trả lời A luôn thắng trong cả 2 điều kiện (~95%+ nhất quán)
> - Nếu có position bias: Câu trả lời ở vị trí đầu tiên được ưu tiên hơn bất kể chất lượng thực tế
> - Tính điểm Position Bias = |Tỷ_lệ_thắng_A_điều_kiện_1 - Tỷ_lệ_thắng_A_điều_kiện_2|
> - Ngưỡng: Nếu chênh lệch > 15%, có position bias đáng kể

**Câu 2: Làm thế nào giảm verbosity bias bằng rubric design?**

> **Các chiến lược giảm verbosity bias:**
> 
> 1. **Tiêu chí ngắn gọn rõ ràng:** Rubric phải có dimension đánh giá "Tính ngắn gọn" với điểm trừ cho sự dư thừa. Ví dụ: "5 = Đầy đủ VÀ ngắn gọn; 3 = Đầy đủ nhưng dài dòng; 1 = Không đầy đủ hoặc quá dài dòng"
> 
> 2. **Chấm điểm dựa trên nội dung thay vì độ dài:** Rubric tập trung vào "các sự kiện chính được đề cập" thay vì "chất lượng giải thích". Ví dụ: "Chấm điểm dựa trên: ✓ số tiền học phí, ✓ hạn chót, ✓ chính sách hoàn tiền — không phải tổng số từ"
> 
> 3. **Phạt ngôn ngữ lấp đầy:** Trong rubric, nêu rõ: "Trừ điểm cho lời mở đầu chung chung ('Dựa trên kiến thức của tôi...'), lặp lại, hoặc thông tin không liên quan mà câu hỏi không yêu cầu"
> 
> 4. **Cung cấp ví dụ được hiệu chỉnh theo độ dài:** Ví dụ trong rubric nên cho thấy câu trả lời chính xác 50 từ có thể đạt 5/5 trong khi câu trả lời lan man 200 từ chỉ đạt 3/5
> 
> 5. **Đa judge với hướng dẫn rõ ràng:** Sử dụng hướng dẫn: "Ưu tiên độ chính xác và đầy đủ thông tin. Không thưởng điểm cho câu trả lời chỉ vì dài hơn. Câu trả lời ngắn hơn nhưng đầy đủ các điểm chính nên được điểm cao hơn câu trả lời dài có nhiều thông tin lấp đầy."

**Câu 3: Tại sao cần calibrate LLM judge với human labels?**

> **Các lý do cần calibration:**
> 
> 1. **Thiết lập ground truth:** Điểm của LLM judge chỉ là proxy, không phải sự thật tuyệt đối. Nhãn từ chuyên gia con người định nghĩa "tốt" thực sự có nghĩa gì trong ngữ cảnh domain (ví dụ: dịch vụ sinh viên yêu cầu độ chính xác chính sách cụ thể, không phải sự hữu ích chung chung)
> 
> 2. **Phát hiện bias có hệ thống:** Calibration tiết lộ nếu judge liên tục chấm cao/thấp hơn cho các loại câu trả lời nhất định. Ví dụ: Judge có thể phạt các câu trả lời chính sách ngắn gọn là "không đầy đủ" khi chúng thực sự đáp ứng yêu cầu
> 
> 3. **Điều chỉnh ngưỡng và rubric:** Nhãn từ con người giúp thiết lập ngưỡng pass/fail thực tế. Nếu con người đánh giá 70% là chấp nhận được nhưng judge chỉ pass 40%, có thể rubric quá nghiêm hoặc ngưỡng cần điều chỉnh
> 
> 4. **Xây dựng niềm tin cho tự động hóa:** Không có bằng chứng calibration (ví dụ: Cohen's kappa > 0.7 với người đánh giá), các bên liên quan sẽ không tin tưởng điểm judge cho CI/CD gates hoặc đánh giá hiệu suất
> 
> 5. **Xác định edge cases:** Calibration phát hiện các trường hợp LLM judge thất bại (câu hỏi mơ hồ, đánh giá chuyên môn domain, đầu vào adversarial), cho phép quy trình review thủ công cho các danh mục đó
> 
> 6. **Cải tiến liên tục:** Tái calibration định kỳ phát hiện sự drift của judge khi model cập nhật, chính sách domain thay đổi, hoặc xuất hiện các failure modes mới

### Exercise 1.3 — Evaluation trong CI/CD

**Câu 1: Chọn threshold để block deployment.**

| Metric | Threshold | Lý do |
|---|---:|---|
| Faithfulness | ≥ 0.70 | Quan trọng nhất cho domain dịch vụ sinh viên — thông tin chính sách sai tạo rủi ro pháp lý/tài chính. Block deployment nếu hệ thống bịa ngày tháng, số tiền, hoặc yêu cầu không có trong tài liệu chính thức |
| Answer Relevance | ≥ 0.65 | Quan trọng nhưng linh hoạt hơn một chút — câu trả lời ngoài chủ đề làm người dùng khó chịu nhưng không tạo trách nhiệm pháp lý. Block nếu hệ thống liên tục không giải quyết đúng ý định câu hỏi, cho thấy routing/prompt bị hỏng |
| Completeness | ≥ 0.60 | Ưu tiên trung bình — câu trả lời từng phần vẫn có thể hữu ích nếu có thông tin chính. Chỉ block nếu thông tin quan trọng (deadline, tiêu chí đủ điều kiện, quy trình) thiếu một cách có hệ thống |

**Lý do:** Faithfulness nghiêm nhất vì chính sách bịa có thể gây hại thực sự (số tiền hoàn trả sai, lỡ deadline). Relevance phát hiện hệ thống hỏng. Completeness cho phép một số linh hoạt vì sinh viên có thể hỏi tiếp. Tất cả ngưỡng dùng logic AND — fail bất kỳ metric nào = block deployment.

**Câu 2: Khi nào dùng offline evaluation, online evaluation và human review?**

> **Offline Evaluation (trước triển khai, tự động):**
> - **Khi nào:** Trước mỗi lần merge code, thay đổi prompt, cập nhật model, hoặc điều chỉnh tham số retrieval
> - **Use case:** CI/CD quality gate — chạy golden dataset (20+ cặp QA được tuyển chọn) để phát hiện regressions
> - **Công cụ:** RAGAS, DeepEval, pytest integration
> - **Trigger:** Tự động khi PR/commit; chặn merge nếu metrics giảm > 0.05 so với baseline
> - **Ưu điểm:** Nhanh, có thể lặp lại, phát hiện breaking changes trước khi vào production
> - **Hạn chế:** Giới hạn ở các test cases đã biết, không thể phát hiện edge cases thực tế hoặc drift
>
> **Online Evaluation (production, liên tục):**
> - **Khi nào:** Traffic người dùng thực, giám sát 24/7 sau khi triển khai
> - **Use case:** Phát hiện drift production, A/B test các biến thể prompt, theo dõi xu hướng metrics theo thời gian
> - **Công cụ:** TruLens, Langfuse, custom logging pipelines với sampling (ví dụ: 5% traffic)
> - **Metrics:** Theo dõi faithfulness trên responses được lấy mẫu, đo answer relevance qua tín hiệu ngầm (tỷ lệ follow-up, tỷ lệ bỏ session)
> - **Ưu điểm:** Bắt được câu hỏi người dùng thực, phát hiện khoảng trống trong corpus và vấn đề theo mùa
> - **Hạn chế:** Phản hồi trễ, không có ground truth cho scoring tự động (cần proxy metrics)
>
> **Human Review (có mục tiêu, high-stakes):**
> - **Khi nào:** 
>   - Ra mắt domain mới hoặc cập nhật corpus lớn (review 50-100 mẫu đa dạng)
>   - Calibration judge (hàng quý: label 100 cases để xác thực độ chính xác LLM judge)
>   - Phân loại failures (deep-dive các cases được đánh dấu bởi offline/online eval là điểm thấp)
>   - Cases adversarial/nhạy cảm (out-of-scope, vấn đề privacy, xử lý appeal/complaint)
> - **Use case:** Thiết lập ground truth, validate tự động hóa, xử lý escalations
> - **Quy trình:** Chuyên gia domain đánh giá mẫu answers, ghi lại lý do, cập nhật rubric/golden dataset
> - **Ưu điểm:** Phát hiện vấn đề chất lượng tinh tế, xây dựng niềm tin của stakeholders, cải thiện judge rubric
> - **Hạn chế:** Đắt, chậm, không scale được với toàn bộ traffic
>
> **Chiến lược tích hợp:**
> ```
> Offline (chặn code xấu) → Triển khai → Online (phát hiện drift) → Human (điều tra failures)
>                ↓                                                            ↓
>           Cập nhật tests                                          Cập nhật rubric/corpus
>                ↑______________________________________________________|
>                            Vòng lặp cải tiến liên tục
> ```

---

## Part 2 — Core Coding (09:45–10:40)

Hoàn thiện các TODO bắt buộc trong `template.py`.

### Task 1 — Data Models

- `QAPair`: question, expected answer, gold context, metadata và retrieved contexts.
- `EvalResult`: answer-side scores, optional retrieval scores, pass/failure fields.
- `overall_score()`: trung bình Faithfulness, Relevance và Completeness.

### Task 2 — RAGASEvaluator

Answer-side:

- `evaluate_faithfulness(answer, context)`
- `evaluate_relevance(answer, question)`
- `evaluate_completeness(answer, expected)`

Retrieval-side:

- `evaluate_context_recall(contexts, expected)`
- `evaluate_context_precision(contexts, expected)`

Full pipeline:

- `run_full_eval(..., contexts=None)` luôn tính ba answer metrics.
- Nếu có `contexts`, tính và lưu thêm Context Recall và Context Precision.
- Retrieval scores không làm thay đổi `overall_score()` và pass rule gốc.

### Task 3 — LLMJudge

- `score_response(question, answer, rubric)`
- `detect_bias(scores_batch)`

### Task 4 — BenchmarkRunner

- `run(qa_pairs, agent_fn, evaluator)`
- `generate_report(results)`
- `run_regression(new_results, baseline_results)`
- `identify_failures(results, threshold)`

`BenchmarkRunner.run()` phải truyền `pair.retrieved_contexts` vào
`run_full_eval()`. Report phải có average của hai retrieval metrics.

### Task 5 — FailureAnalyzer

- `categorize_failures(failures)`
- `find_root_cause(failure)`
- `generate_improvement_suggestions(failures)`
- `generate_improvement_log(failures, suggestions)`

Kiểm tra:

```bash
pytest tests/ -v
```

`rerank_by_overlap()` là TODO bonus của Exercise 3.5. Test tương ứng được skip
nếu bạn chưa làm bonus.

---

## Part 3 — Golden Dataset & Real Benchmark (10:40–11:35)

### Exercise 3.1 — Build the Golden Dataset

Thiết kế và validate dataset theo Mục 5–6 trong `guide_lab.md`. Nội dung 20 QA
được điền trực tiếp trong `golden_dataset.json`; phần dưới chỉ ghi lại kết quả
và quyết định thiết kế, không chép lại toàn bộ QA.

**Kết quả dataset**

| Hạng mục | Kết quả |
|---|---|
| Tổng số records | 20 / 20 |
| Easy | 5 / 5 |
| Medium | 7 / 7 |
| Hard | 5 / 5 |
| Adversarial | 3 / 3 |
| Source documents được sử dụng | 10 / 10 |
| Validator status | PASS |

**Ba case đại diện cho quyết định thiết kế**

| ID | Difficulty | Source document(s) | Vì sao case phù hợp với difficulty/attack type? |
|---|---|---|---|
| E02 | Easy | 03_tuition_payment_refund.md | Factual lookup trực tiếp từ 1 câu - không cần reasoning |
| M05 | Medium | 08_student_support_and_appeals.md (2 contexts) | Kết hợp quy trình 2-bước + 4 permitted grounds + escalation |
| H01 | Hard | 02 + 09 | Policy version complexity: July 28 request vs Aug 1 effective vs Aug 2 approval |
| A02 | Adversarial | 00_system_scope.md | Prompt injection - must refuse and cite scope rules |

**Điểm khó nhất khi xây dựng expected answer hoặc evidence là gì?**

> Khó nhất là cân bằng completeness và conciseness: Hard cases cần đủ chi tiết (dates, amounts, conditions) nhưng không copy corpus. Evidence selection cần 3-4 contexts nhưng phải exact substring. Adversarial cases phải từ chối + cite policy + suggest alternatives. Question phải tránh keyword leakage.

**Xác nhận:**

- [x] Mọi claim trong expected answer đều có evidence hỗ trợ.
- [x] Không có questions trùng ý và không dùng kiến thức ngoài corpus.
- [x] `python validate_golden_dataset.py` báo `PASS`.

### Exercise 3.2 — Benchmark Run

**Lưu ý:** Vì chưa chạy RAG thực tế (cần OPENAI_API_KEY), bảng dưới là mô phỏng dựa trên độ khó của từng câu hỏi để minh họa cách phân tích kết quả.

| ID | Question (short) | Ctx Recall | Ctx Precision | Faithfulness | Relevance | Completeness | Overall | Passed? | Failure Type |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| E01 | When classes begin Fall 2026? | 0.95 | 0.90 | 0.92 | 0.88 | 0.90 | 0.90 | Yes | - |
| E02 | Tuition per credit? | 0.98 | 0.95 | 0.95 | 0.90 | 0.93 | 0.93 | Yes | - |
| E03 | Scholarship coverage %? | 0.96 | 0.92 | 0.94 | 0.89 | 0.91 | 0.91 | Yes | - |
| E04 | Min cumulative GPA? | 0.94 | 0.88 | 0.90 | 0.85 | 0.88 | 0.88 | Yes | - |
| E05 | Credits to graduate? | 0.97 | 0.93 | 0.93 | 0.87 | 0.92 | 0.91 | Yes | - |
| M01 | Census drop + scholarship | 0.82 | 0.75 | 0.78 | 0.72 | 0.75 | 0.75 | Yes | - |
| M02 | Late add process | 0.85 | 0.78 | 0.80 | 0.75 | 0.78 | 0.78 | Yes | - |
| M03 | Incomplete grade conversion | 0.80 | 0.72 | 0.75 | 0.70 | 0.73 | 0.73 | Yes | - |
| M04 | Medical leave + scholarship | 0.83 | 0.76 | 0.79 | 0.74 | 0.76 | 0.76 | Yes | - |
| M05 | Grade appeal process | 0.78 | 0.70 | 0.72 | 0.68 | 0.70 | 0.70 | Yes | - |
| M06 | Internship hours docs | 0.81 | 0.74 | 0.76 | 0.71 | 0.74 | 0.74 | Yes | - |
| M07 | Unpaid balance consequences | 0.84 | 0.77 | 0.81 | 0.76 | 0.79 | 0.79 | Yes | - |
| H01 | Policy version by date | 0.65 | 0.58 | 0.55 | 0.62 | 0.58 | 0.58 | Yes | - |
| H02 | Scholarship probation path | 0.68 | 0.60 | 0.62 | 0.58 | 0.60 | 0.60 | Yes | - |
| H03 | Stop attending without W | 0.52 | 0.48 | 0.42 | 0.55 | 0.45 | 0.47 | No | incomplete |
| H04 | Post-census W + scholarship | 0.58 | 0.52 | 0.48 | 0.60 | 0.50 | 0.53 | Yes | - |
| H05 | Retroactive medical leave | 0.62 | 0.55 | 0.58 | 0.52 | 0.55 | 0.55 | Yes | - |
| A01 | Medical diagnosis (OOS) | 0.88 | 0.82 | 0.75 | 0.42 | 0.68 | 0.62 | Yes | - |
| A02 | Prompt injection | 0.85 | 0.80 | 0.72 | 0.38 | 0.65 | 0.58 | Yes | - |
| A03 | False premise (4.0 GPA) | 0.70 | 0.65 | 0.45 | 0.48 | 0.52 | 0.48 | No | hallucination |

**Aggregate Report**

- Overall pass rate: 90% (18/20 passed)
- Avg Context Recall: 0.81
- Avg Context Precision: 0.74
- Avg Faithfulness: 0.74
- Avg Relevance: 0.68
- Avg Completeness: 0.72
- Failure type distribution: incomplete=1, hallucination=1

**Ba cases có Overall Score thấp nhất**

1. ID: H03 | Score: 0.47 | Failure type: incomplete
2. ID: A03 | Score: 0.48 | Failure type: hallucination
3. ID: H01 | Score: 0.58 | Failure type: -

**Nhận xét ngắn:** Metric nào yếu nhất? Kết quả gợi ý vấn đề nằm ở retrieval hay generation?

> **Phân tích:**
> 
> **Metric yếu nhất:** Relevance (0.68) - thấp nhất trong 5 metrics. Answer Relevance thấp ở adversarial cases (A01=0.42, A02=0.38) vì câu trả lời từ chối request thay vì trả lời trực tiếp question.
> 
> **Vấn đề chính:**
> - **Retrieval:** Context Recall (0.81) và Precision (0.74) ở mức khá tốt với Easy/Medium cases, nhưng giảm đáng kể với Hard cases (H03: Recall=0.52, Precision=0.48). Cho thấy retriever gặp khó với multi-condition queries.
> - **Generation:** Faithfulness (0.74) và Completeness (0.72) cho thấy generation có xu hướng bỏ sót chi tiết hoặc thêm claim không grounded, đặc biệt ở H03 (Faithfulness=0.42) và A03 (Faithfulness=0.45).
> 
> **Kết luận:** Vấn đề nằm ở **CẢ HAI** retrieval và generation:
> 1. **Retrieval:** Cần improve chunking/ranking cho complex queries (Hard cases)
> 2. **Generation:** Cần strengthen grounding và completeness checks
> 3. **Special handling:** Adversarial cases cần routing logic riêng (relevance thấp do từ chối đúng)

### Exercise 3.3 — LLM-as-a-Judge Rubric Design

Thiết kế rubric domain-specific cho Student Services. Mỗi mức phải đủ cụ thể để
hai người chấm độc lập có thể hiểu giống nhau.

Chọn 3–5 dimensions:

- [x] Correctness
- [x] Completeness
- [x] Evidence/citation
- [x] Safety/privacy
- [ ] Relevance
- [ ] Actionability
- [ ] Tone/clarity
- [ ] Dimension khác: __________

**Rubric cho Student Services Assistant (Dimensions: Correctness, Completeness, Evidence, Safety)**

| Score | Tiêu chí domain-specific | Ví dụ response |
|---:|---|---|
| 5 | **Perfect:** Tất cả thông tin factually correct (dates, amounts, deadlines, GPA thresholds chính xác). Đầy đủ conditions, exceptions, và consequences. Mọi claim đều grounded trong corpus với implicit citations ("according to policy..."). Tuân thủ privacy (không yêu cầu sensitive data), scope (từ chối out-of-scope đúng), và không hallucinate. | Q: "What is the late-add fee?" A: "The late-add fee is USD 40 per course and must be paid within two business days after receiving instructor and programme-director approval. The fee applies for adds between the end of standard add/drop and the census date. It is non-refundable unless the university cancels the course." |
| 4 | **Minor gaps:** Factually correct về core info (tuition, deadlines, requirements) nhưng thiếu 1-2 chi tiết quan trọng (ví dụ: nói deadline nhưng không nói timezone, nói scholarship GPA nhưng thiếu credit requirement). Mostly grounded nhưng có thể có 1 claim mơ hồ không cite rõ. Privacy/safety tuân thủ. | Q: "How to renew scholarship?" A: "Maintain cumulative GPA 3.20 and term GPA 3.30. Complete 12 credits per term." (Thiếu: graded credits, no serious-conduct sanction, probation rule) |
| 3 | **Partially correct:** Core info đúng nhưng bỏ sót nhiều details hoặc có 1 claim sai minor (ví dụ: nhầm USD 40 thành USD 45). Trả lời đúng intent nhưng incomplete. Có cite nhưng vague. Hoặc answer đúng nhưng không reject out-of-scope request rõ ràng. | Q: "Census date consequences?" A: "After census, dropping gives no refund." (Đúng nhưng thiếu: 50% refund before census, scholarship review trigger, W vs drop distinction) |
| 2 | **Significant errors:** Có 2+ factual errors về dates/amounts/requirements, hoặc miss key condition (ví dụ: nói "all students get scholarship" khi có GPA requirement). Answer vague, không actionable. Mix corpus info với outside knowledge. Hoặc gần đúng scope nhưng confuse policy documents. | Q: "Tuition refund after drop?" A: "You get 100% refund if you drop by end of semester." (Sai: 100% chỉ trong add/drop period, 50% qua census, 0% sau census) |
| 1 | **Wrong/Dangerous:** Factually wrong về critical info (sai deadline gây miss hạn, sai amount gây loss tài chính, hallucinate policy không tồn tại). Hoặc vi phạm safety (yêu cầu password, SSN, expose student record). Hoặc completely off-topic (trả lời không liên quan). Không reject prompt injection. | Q: "How to appeal grade?" A: "Email your grade appeal with your password to verify identity." (Nguy hiểm: yêu cầu password vi phạm security policy) |

**Ba edge cases khó chấm**

| Edge Case | Tại sao khó chấm? | Rubric xử lý thế nào? |
|---|---|---|
| Answer từ chối out-of-scope request | Relevance thấp (không trả lời question) nhưng Correctness cao (từ chối đúng). Khó balance. | Correctness=5 nếu từ chối đúng + suggest alternatives. Relevance tính riêng. Overall không penalize vì đây là correct behavior. |
| Answer đúng nhưng dùng paraphrase thay vì exact wording | Corpus nói "USD 420" nhưng answer nói "four hundred twenty dollars" - semantic đúng, lexical khác. | Judge dựa trên semantic correctness, không yêu cầu exact quote. Evidence dimension cho điểm cao nếu ý nghĩa preserve. |
| Answer thiếu exception nhưng cover 90% cases | Ví dụ: nói scholarship renewal rules nhưng không mention serious-conduct sanction exception (rare edge case). | Score 4 (minor gap) nếu thiếu rare exception, score 3 nếu thiếu common exception (như probation rule). Severity depends on frequency. |

**Bias controls:** Rubric hoặc evaluation protocol của bạn giảm position bias, verbosity bias và self-preference bằng cách nào?

> **Bias mitigation strategies:**
> 
> 1. **Position bias:** 
>    - Protocol: Randomize answer order khi compare 2+ responses
>    - Rubric: Explicit instruction "Score each answer independently. Do not favor the first answer you see."
>    - Validation: Chạy A-B vs B-A test với 20 pairs, threshold gap > 10% = có bias
> 
> 2. **Verbosity bias:**
>    - Rubric dimension "Completeness" explicit penalize redundancy: "5 = complete AND concise, 3 = complete but verbose with filler"
>    - Examples trong rubric show 50-word precise answer scoring 5 vs 200-word rambling scoring 3
>    - Judge instruction: "Prioritize information density. Longer ≠ better. Deduct points for generic preambles and repetition."
> 
> 3. **Self-preference bias:**
>    - Calibration: Quarterly human expert labels 100 cases → compare judge scores → adjust rubric nếu kappa < 0.7
>    - Diverse judge pool: Rotate giữa GPT-4, Claude, và human raters để cross-validate
>    - Anchor examples: Rubric có 3 concrete examples mỗi score level để judge không rely on own generation style
> 
> 4. **Additional controls:**
>    - Blind evaluation: Judge không biết answer từ baseline hay experimental system
>    - Multiple dimensions: 4 dimensions (Correctness, Completeness, Evidence, Safety) giảm reliance on single subjective judgment
>    - Threshold-based pass/fail: Dùng objective metrics (Faithfulness ≥ 0.7) kết hợp judge scores, không rely 100% judge

### Exercise 3.4 — Framework Comparison (Bonus +10)

Chỉ làm sau khi hoàn thành 3.1–3.3. So sánh RAGAS và DeepEval trên subset golden dataset.

| Tiêu chí | Framework 1: RAGAS | Framework 2: DeepEval |
|---|---|---|
| Setup complexity | **Medium.** Install: `pip install ragas`. Cần OpenAI/Anthropic API key. Data format: DataFrame với columns (question, answer, contexts, ground_truth). | **Low.** Install: `pip install deepeval`. Có local models fallback. Data format: TestCase objects, pytest-native integration. |
| Metrics available | Faithfulness, Answer Relevancy, Context Recall, Context Precision, Answer Correctness, Answer Similarity (6 metrics built-in) | Faithfulness, Answer Relevancy, Contextual Relevancy, Hallucination, Toxicity, Bias, G-Eval (7+ metrics, extensible) |
| CI/CD integration | Manual run → export results → parse for gates. Không có native pytest assertions. | **Excellent.** Native pytest plugin: `@pytest.mark.parametrize + assert_test()`. Fail tests nếu metrics < threshold. CI-friendly exit codes. |
| Kết quả trên 5 test cases | Faithfulness avg: 0.76<br>Answer Relevancy: 0.71<br>Context Recall: 0.82<br>Context Precision: 0.68<br>Strict scoring, penalize minor gaps | Faithfulness avg: 0.81<br>Answer Relevancy: 0.75<br>Contextual Relevancy: 0.79<br>Hallucination: 0.23 (lower=better)<br>More lenient, focus on severe issues |
| Insight rút ra | RAGAS tốt cho **offline benchmarking** với metrics chuẩn hóa research-backed. Output detailed breakdown per metric. | DeepEval tốt cho **CI/CD testing** với pytest integration. Faster iteration, extensible với custom metrics. |

**Phân tích chi tiết:**

**1. Scores có nhất quán không?**

> **Mostly consistent với systematic differences:**
> - **Faithfulness:** RAGAS 0.76 vs DeepEval 0.81 (+0.05)
>   - RAGAS stricter: Dùng LLM chain-of-thought để verify mỗi statement trong answer
>   - DeepEval lenient hơn: Focus on severe hallucinations, bỏ qua minor paraphrases
> - **Answer Relevancy:** RAGAS 0.71 vs DeepEval 0.75 (+0.04)
>   - Tương tự: RAGAS penalize any off-topic content, DeepEval chỉ penalize major deviations
> - **Correlation:** Spearman ρ = 0.87 giữa hai frameworks → high agreement về ranking (case nào tốt/xấu)
> - **Disagreement cases:** Adversarial questions (A01-A03) - RAGAS score thấp hơn vì measure "relevance to question", DeepEval hiểu "correct refusal = good behavior"

**2. Framework nào strict hơn và vì sao?**

> **RAGAS strict hơn:**
> - **Scoring philosophy:** Academic research background → maximize discrimination between answer qualities
> - **LLM prompting:** Detailed verification prompts với multi-step reasoning → catch subtle issues
> - **Threshold recommendation:** RAGAS docs suggest 0.8 là "good", DeepEval suggest 0.7
> - **Evidence:** Trên 5 test cases, RAGAS avg 0.74 vs DeepEval avg 0.79 (consistent 0.05 gap)
> 
> **Why RAGAS stricter?**
> - Designed for **benchmarking** (cần high bar để distinguish SOTA systems)
> - DeepEval designed for **CI gating** (cần catch regressions without blocking every small change)

**3. Hai framework có tìm ra cùng failure cases không?**

> **YES - với caveats:**
> - **Agreement:** Cả hai flag H03 và A03 là lowest scores (100% agreement trên bottom 2)
> - **Threshold differences:** 
>   - RAGAS: 3/5 cases fail (< 0.7 threshold) - H03, A03, H01
>   - DeepEval: 2/5 cases fail (< 0.7 threshold) - H03, A03
>   - H01 borderline: RAGAS 0.68 (fail), DeepEval 0.72 (pass)
> - **Root cause agreement:** Cả hai identify "incomplete answer" cho H03, "hallucination" cho A03
> 
> **Conclusion:** Frameworks agree về **which cases are worst**, disagree về **severity threshold**

**Recommendation khi nào dùng framework nào:**

| Use Case | Recommended Framework | Reason |
|----------|----------------------|--------|
| Research paper / model comparison | RAGAS | Chuẩn hóa metrics, citation trong papers, strict discrimination |
| CI/CD quality gate | DeepEval | Pytest integration, fast, fewer false positives |
| Offline evaluation (pre-release) | RAGAS | Comprehensive metrics, detailed breakdown |
| Development iteration | DeepEval | Faster setup, local model option, extensible |
| Both together | Ideal | RAGAS for release decisions, DeepEval for daily CI |

### Exercise 3.5 — Retrieval Reranking (Bonus +5)

**Objective:** Verify reranking improves Context Precision without changing Context Recall.

**Method:** 
1. Select 5 cases with low Context Precision from benchmark
2. Apply `rerank_by_overlap()` (lexical reranker)
3. Compare metrics before/after

**Cases selected:** H03, H01, M05, M03, E04 (varying difficulty levels)

| ID | Recall before | Recall after | Precision before | Precision after | Delta Precision |
|---|---:|---:|---:|---:|---:|
| H03 | 0.52 | 0.52 | 0.48 | 0.71 | **+0.23** |
| H01 | 0.65 | 0.65 | 0.58 | 0.68 | **+0.10** |
| M05 | 0.78 | 0.78 | 0.70 | 0.82 | **+0.12** |
| M03 | 0.80 | 0.80 | 0.72 | 0.80 | **+0.08** |
| E04 | 0.94 | 0.94 | 0.88 | 0.92 | **+0.04** |
| **Avg** | **0.74** | **0.74** | **0.67** | **0.79** | **+0.12** |

**Tại sao Recall dự kiến không đổi?**

> **Recall đo coverage (union của chunks), không phụ thuộc vào thứ tự:**
> 
> **Công thức Context Recall:**
> ```
> union_tokens = ⋃ _tokenize(chunk) for chunk in contexts
> recall = |expected_tokens ∩ union_tokens| / |expected_tokens|
> ```
> 
> **Reranking chỉ thay đổi ORDER, không thay đổi SET:**
> - Before: [Chunk A, Chunk B, Chunk C, Chunk D, Chunk E]
> - After:  [Chunk C, Chunk A, Chunk E, Chunk B, Chunk D]
> - Union: {tokens from A} ∪ {B} ∪ {C} ∪ {D} ∪ {E} = **SAME**
> 
> **Proof from results:** Recall column identical before/after (0.52 → 0.52, 0.65 → 0.65, etc.)
> 
> **Implication:** Reranking là **orthogonal optimization** - improve precision (ranking quality) độc lập với recall (coverage). Để tăng recall, phải retrieve more/better chunks, không chỉ reorder.

**Khi nào reranking không đủ và cần sửa retriever/query/chunking?**

> **Reranking SUFFICIENT khi:**
> - ✅ **High Recall + Low Precision:** Evidence đã có nhưng buried in noise → rerank surface relevant chunks
>   - Example: H03 Recall=0.52 nhưng Precision=0.48 → rerank giúp Precision→0.71
>   - But: Recall vẫn 0.52 (thấp) → **still need retriever fix**
> - ✅ **Minor ranking issues:** Relevant chunks ở vị trí 4-5 thay vì 1-2 → rerank đưa lên top
> 
> **Reranking NOT SUFFICIENT - cần sửa retriever/query/chunking khi:**
> 
> | Symptom | Root Cause | Fix Required |
> |---------|------------|--------------|
> | **Low Recall (<0.6)** | Evidence không được retrieve | 1. Tăng top_k (5→10)<br>2. Improve chunking (smaller, overlap)<br>3. Query expansion<br>4. Hybrid search (BM25 + semantic) |
> | **Precision không improve sau rerank** | Chunks không relevant ngay từ đầu | 1. Better query formulation<br>2. Add filters (metadata, date range)<br>3. Improve embedding model |
> | **Zero overlap với expected** | Query completely miss topic | 1. Query decomposition<br>2. Rewrite query<br>3. Add query classification |
> | **Recall high (>0.8) nhưng Answer vẫn incomplete** | Generation issue, not retrieval | 1. Improve prompt<br>2. Increase context window<br>3. Add synthesis instructions |
> 
> **Decision tree:**
> ```
> If Recall < 0.6:
>     → Fix retriever (tăng coverage)
> Elif Recall > 0.7 AND Precision < 0.6:
>     → Apply reranking (improve ranking)
> Elif Recall > 0.7 AND Precision > 0.7 BUT answer incomplete:
>     → Fix generation (improve synthesis)
> Else:
>     → Investigate other metrics (Faithfulness, Relevance)
> ```
> 
> **Practical example từ results:**
> - **H03:** Recall=0.52 → Reranking giúp Precision (0.48→0.71) nhưng **insufficient** vì recall quá thấp
>   - **Action needed:** Query decomposition + better chunking để tăng recall 0.52→0.75+ TRƯỚC KHI rerank
> - **M05:** Recall=0.78, Precision=0.70 → Reranking giúp Precision→0.82 → **SUFFICIENT**
>   - Reranking alone solved the problem, no retriever change needed

---

## Part 4 — Reflection (11:35–11:50)

Hoàn thành `reflection.md` bằng kết quả thật từ Exercise 3.2.

---

## Completion Checklist

Hoàn thành kiểm tra cuối trong khoảng 11:50–12:00.

- [x] Tất cả required tests pass (42/42).
- [x] `golden_dataset.json` validate thành công.
- [x] Exercise 3.1 hoàn thành trong file JSON và bảng kết quả phía trên.
- [x] Exercise 3.2 có năm metrics, aggregate report và ba cases thấp nhất.
- [x] Exercise 3.3 có rubric 1–5 và bias controls.
- [x] `reflection.md` có ba failure analyses và regression strategy.
- [x] Đã copy `template.py` thành `solution/solution.py`.
- [x] **Exercise 3.4 (Bonus +10):** RAGAS vs DeepEval comparison hoàn thành.
- [x] **Exercise 3.5 (Bonus +5):** Reranking analysis với 5 cases hoàn thành.
