Graders

  1. nutrition plan tiering (nutirion plan tiering/grader.json)

  - Evaluates 3-phase nutrition structure (Reset, Rebuild, Sustain)
  - Checks for phase presence and timeline accuracy
  - Assesses food category specificity and anti-inflammatory focus
  - Weight distribution: 25% semantic accuracy, 20% nutrition detail, 15% structure, 36% phase checks

  2. cross-validated-logic (cross-validated-logic/grader.json)

  - Validates risk cluster identification and pattern narratives
  - Checks for core symptoms, supporting metrics, and cross-validated connections
  - Evaluates multimodal synthesis quality
  - Weight distribution: 25% semantic accuracy, 20% pattern detail, 15% structure, 40% structural elements

  3. lifestyle pillars (lifestyle pillars/grader.json)

  - Verifies all 6 pillars are present (Sleep, Movement, Nutrition, Mindfulness, Social Connection, Environment)
  - Evaluates actionable routines and supplement synergies
  - Checks for warm, supportive tone
  - Weight distribution: 25% semantic accuracy, 20% lifestyle detail, 15% structure, 40% pillar coverage

  4. personalized-action (personalized-action/grader.json)

  - Validates supplement plan table structure (4-6 entries)
  - Checks for key columns: Supplement, Dose, Biomarker Trigger, Duration, Cycling Note
  - Evaluates supplement rationale and biomarker integration
  - Weight distribution: 25% semantic accuracy, 20% supplement detail, 15% structure, 40% table elements

  5. supplement plan (supplement plan/grader.json)

  - Evaluates system-based table (System, Supplement, Duration, Notes)
  - Checks for general guidance tone and system-level connections
  - Validates integration with lab data
  - Weight distribution: 25% semantic accuracy, 20% system detail, 15% structure, 40% table elements

  6. user goals (user goals/grader.json)

  - Validates goal map table (Goal, Therapeutic Focus, Target System, Supporting Data)
  - Checks for motivational tone and narrative summary
  - Evaluates connection between goals and biomarkers
  - Weight distribution: 25% semantic accuracy, 20% goals detail, 15% structure, 40% structural elements

  Grader Design Principles

  Each grader follows the same multi-grader architecture as your reference grader with:
  - Text similarity metrics (ROUGE-L, Cosine, METEOR, BLEU) for semantic and structural accuracy
  - String checks for domain-specific elements (phases, pillars, table columns, etc.)
  - Weighted calculations optimized for each category's unique requirements
  - Balance between semantic accuracy (60%) and structural completeness (40%)

  All graders are ready to use with your OpenAI RFT fine-tuning pipeline!