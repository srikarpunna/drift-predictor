"""
35 pilot prompts — 5 per task family.

Prompts are split into two paths:
  - Instructor path (schema_name set): run_structured() via Instructor.
    Prompt text is natural language only — no "output JSON" instruction.
    Tests H2: schema adherence drift.
  - Text path (schema_name=None): run_text(), oracle validates raw output.
    Tests H1 (verbosity), format drift, instruction following.

Task family → path:
  classification      → Instructor
  extraction          → Instructor
  structured_output   → Instructor
  transformation      → Instructor
  tool_calling        → text  (tests instruction following for tool dispatch)
  summarization       → text  (tests length/format constraints)
  multi_constraint    → text  (tests competing constraint handling)
"""

from src.benchmark.prompt_item import OracleConfig, PromptItem

PILOT_PROMPTS: list[PromptItem] = [

    # -------------------------------------------------------------------------
    # CLASSIFICATION — Instructor path → SentimentOutput / IntentOutput / UrgencyOutput
    # Prompt: natural language task only. Instructor enforces schema.
    # -------------------------------------------------------------------------
    PromptItem(
        id="cls-001",
        task_family="classification",
        prompt_text=(
            'Classify the sentiment of this customer review.\n\n'
            'Review: "The product arrived on time and works exactly as described. Very happy!"'
        ),
        schema_name="SentimentOutput",
        oracle=OracleConfig(
            type="structured",
            assertions=[{"field": "label", "op": "in", "value": ["positive", "negative", "neutral"]}],
        ),
    ),
    PromptItem(
        id="cls-002",
        task_family="classification",
        prompt_text=(
            'Classify the sentiment of this customer review.\n\n'
            'Review: "Terrible quality. Broke after two days. Complete waste of money."'
        ),
        schema_name="SentimentOutput",
        oracle=OracleConfig(
            type="structured",
            assertions=[{"field": "label", "op": "in", "value": ["positive", "negative", "neutral"]}],
        ),
    ),
    PromptItem(
        id="cls-003",
        task_family="classification",
        prompt_text=(
            'Classify the intent of this user message into one category.\n\n'
            'Message: "Can you tell me what your return policy is?"'
        ),
        schema_name="IntentOutput",
        oracle=OracleConfig(
            type="structured",
            assertions=[{"field": "intent", "op": "in", "value": ["question", "complaint", "compliment", "request", "other"]}],
        ),
    ),
    PromptItem(
        id="cls-004",
        task_family="classification",
        prompt_text=(
            'Classify this support ticket by urgency level and provide a one-sentence reason.\n\n'
            'Ticket: "Our production database is down and we cannot process any orders."'
        ),
        schema_name="UrgencyOutput",
        oracle=OracleConfig(
            type="structured",
            assertions=[
                {"field": "urgency", "op": "in", "value": ["low", "medium", "high", "critical"]},
                {"field": "reason", "op": "not_null"},
            ],
        ),
    ),
    PromptItem(
        id="cls-005",
        task_family="classification",
        prompt_text=(
            'Classify the sentiment of this product review. Include a confidence score.\n\n'
            'Review: "Not bad for the price. Some features are missing but it does the job."'
        ),
        schema_name="SentimentOutput",
        oracle=OracleConfig(
            type="structured",
            assertions=[{"field": "label", "op": "in", "value": ["positive", "negative", "neutral"]}],
        ),
    ),

    # -------------------------------------------------------------------------
    # EXTRACTION — Instructor path
    # -------------------------------------------------------------------------
    PromptItem(
        id="ext-001",
        task_family="extraction",
        prompt_text=(
            'Extract the named entities (persons, organizations, locations) from this text.\n\n'
            'Text: "Elon Musk announced that Tesla will open a new factory in Austin, Texas next year."'
        ),
        schema_name="NamedEntities",
        oracle=OracleConfig(
            type="structured",
            assertions=[
                {"field": "persons", "op": "isinstance", "value": "list"},
                {"field": "organizations", "op": "isinstance", "value": "list"},
                {"field": "locations", "op": "isinstance", "value": "list"},
            ],
        ),
    ),
    PromptItem(
        id="ext-002",
        task_family="extraction",
        prompt_text=(
            'Extract key information from this job posting.\n\n'
            'Posting: "Senior Software Engineer at Stripe. Remote-friendly. Competitive salary not disclosed."'
        ),
        schema_name="JobPosting",
        oracle=OracleConfig(
            type="structured",
            assertions=[
                {"field": "job_title", "op": "not_null"},
                {"field": "company", "op": "not_null"},
                {"field": "salary_mentioned", "op": "isinstance", "value": "bool"},
            ],
        ),
    ),
    PromptItem(
        id="ext-003",
        task_family="extraction",
        prompt_text=(
            'Extract the contact information from this business card.\n\n'
            'Text: "Dr. Sarah Chen | CTO at Databricks | sarah.chen@databricks.com | +1-415-555-0192"'
        ),
        schema_name="ContactInfo",
        oracle=OracleConfig(
            type="structured",
            assertions=[
                {"field": "name", "op": "not_null"},
                {"field": "email", "op": "contains", "value": "@"},
            ],
        ),
    ),
    PromptItem(
        id="ext-004",
        task_family="extraction",
        prompt_text=(
            'Extract the financial figures from this earnings report.\n\n'
            'Text: "Q3 revenue came in at $2.4 billion, up 18% year-over-year. Net profit was $340 million."'
        ),
        schema_name="FinancialFigures",
        oracle=OracleConfig(
            type="structured",
            assertions=[
                {"field": "revenue_usd_millions", "op": "not_null"},
                {"field": "yoy_growth_percent", "op": "not_null"},
            ],
        ),
    ),
    PromptItem(
        id="ext-005",
        task_family="extraction",
        prompt_text=(
            'Extract named entities from this text. Classify each as a person, organization, or location.\n\n'
            'Text: "Apple CEO Tim Cook met with EU Commissioner Margrethe Vestager in Brussels to discuss the Digital Markets Act."'
        ),
        schema_name="NamedEntities",
        oracle=OracleConfig(
            type="structured",
            assertions=[
                {"field": "persons", "op": "isinstance", "value": "list"},
                {"field": "organizations", "op": "isinstance", "value": "list"},
                {"field": "locations", "op": "isinstance", "value": "list"},
            ],
        ),
    ),

    # -------------------------------------------------------------------------
    # STRUCTURED OUTPUT — Instructor path (strictest schema adherence test)
    # -------------------------------------------------------------------------
    PromptItem(
        id="so-001",
        task_family="structured_output",
        prompt_text=(
            'Extract product details from this description.\n\n'
            'Description: "The Acme Pro Blender (model X200) is currently available for $89.99. '
            'Great for smoothies and soups. Currently in stock."'
        ),
        schema_name="ProductListing",
        oracle=OracleConfig(
            type="structured",
            assertions=[
                {"field": "product_name", "op": "isinstance", "value": "str"},
                {"field": "price_usd", "op": "isinstance", "value": "float"},
                {"field": "in_stock", "op": "isinstance", "value": "bool"},
                {"field": "tags", "op": "isinstance", "value": "list"},
            ],
        ),
    ),
    PromptItem(
        id="so-002",
        task_family="structured_output",
        prompt_text=(
            'Parse this user record and their permissions.\n\n'
            'Data: "User ID 4821, name John Smith, email john@acme.com, admin role with permissions: read, write, delete, admin."'
        ),
        schema_name="UserPermissions",
        oracle=OracleConfig(
            type="structured",
            assertions=[
                {"field": "permissions", "op": "isinstance", "value": "list"},
            ],
        ),
    ),
    PromptItem(
        id="so-003",
        task_family="structured_output",
        prompt_text=(
            'Convert this meeting note into structured format.\n\n'
            'Note: "Met on Tuesday Dec 10 with Alice, Bob, and Carol. '
            'Decided to launch v2 in Q1 and freeze the feature backlog. '
            'Alice will write the spec by Friday. Bob handles infrastructure review."'
        ),
        schema_name="MeetingNotes",
        oracle=OracleConfig(
            type="structured",
            assertions=[
                {"field": "attendees", "op": "isinstance", "value": "list"},
                {"field": "decisions", "op": "isinstance", "value": "list"},
                {"field": "action_items", "op": "isinstance", "value": "list"},
            ],
        ),
    ),
    PromptItem(
        id="so-004",
        task_family="structured_output",
        prompt_text=(
            'Parse this address into its components.\n\n'
            'Address: "1600 Amphitheatre Parkway, Mountain View, CA 94043, United States"'
        ),
        schema_name="ParsedAddress",
        oracle=OracleConfig(
            type="structured",
            assertions=[
                {"field": "city", "op": "not_null"},
                {"field": "state", "op": "not_null"},
            ],
        ),
    ),
    PromptItem(
        id="so-005",
        task_family="structured_output",
        prompt_text=(
            'Analyze this code snippet for bugs, complexity, and issues.\n\n'
            'Code:\n```python\ndef divide(a, b):\n    return a / b\n```'
        ),
        schema_name="CodeAnalysis",
        oracle=OracleConfig(
            type="structured",
            assertions=[
                {"field": "language", "op": "not_null"},
                {"field": "has_bugs", "op": "isinstance", "value": "bool"},
                {"field": "complexity", "op": "in", "value": ["low", "medium", "high"]},
                {"field": "issues", "op": "isinstance", "value": "list"},
            ],
        ),
    ),

    # -------------------------------------------------------------------------
    # TOOL CALLING — text path (instruction following, not schema adherence)
    # Oracle: raw JSON output matches expected tool call structure
    # -------------------------------------------------------------------------
    PromptItem(
        id="tc-001",
        task_family="tool_calling",
        prompt_text=(
            'You have access to a weather tool. When asked about weather, you MUST respond with ONLY '
            'a JSON tool call in this exact format (no prose):\n'
            '{"tool": "get_weather", "arguments": {"location": string, "unit": "celsius"|"fahrenheit"}}\n\n'
            'User: What is the weather like in Tokyo right now?'
        ),
        oracle=OracleConfig(
            type="unit_test",
            required_keys=["tool", "arguments"],
            assertions=[{"field": "tool", "op": "eq", "value": "get_weather"}],
        ),
    ),
    PromptItem(
        id="tc-002",
        task_family="tool_calling",
        prompt_text=(
            'You have access to a database query tool. Respond ONLY with JSON in this format:\n'
            '{"tool": "query_db", "arguments": {"table": string, "filters": object, "limit": integer}}\n\n'
            'Request: Find the top 5 users who signed up in the last 30 days from the users table.'
        ),
        oracle=OracleConfig(
            type="unit_test",
            required_keys=["tool", "arguments"],
            assertions=[{"field": "tool", "op": "eq", "value": "query_db"}],
        ),
    ),
    PromptItem(
        id="tc-003",
        task_family="tool_calling",
        prompt_text=(
            'You have access to a send_email tool. Respond ONLY with JSON:\n'
            '{"tool": "send_email", "arguments": {"to": string, "subject": string, "body": string, "priority": "low"|"normal"|"high"}}\n\n'
            'Task: Send a high-priority email to alice@company.com notifying her that the server deployment is complete.'
        ),
        oracle=OracleConfig(
            type="unit_test",
            required_keys=["tool", "arguments"],
            assertions=[{"field": "tool", "op": "eq", "value": "send_email"}],
        ),
    ),
    PromptItem(
        id="tc-004",
        task_family="tool_calling",
        prompt_text=(
            'You have two tools: search_web and summarize_text. '
            'Choose the correct tool and respond ONLY with JSON:\n'
            '{"tool": "search_web"|"summarize_text", "arguments": object}\n\n'
            'Request: Find recent news articles about AI regulation in the EU.'
        ),
        oracle=OracleConfig(
            type="unit_test",
            required_keys=["tool", "arguments"],
            assertions=[{"field": "tool", "op": "in", "value": ["search_web", "summarize_text"]}],
        ),
    ),
    PromptItem(
        id="tc-005",
        task_family="tool_calling",
        prompt_text=(
            'You have a calculator tool. Respond ONLY with JSON:\n'
            '{"tool": "calculator", "arguments": {"expression": string, "precision": integer}}\n'
            'The precision field MUST be between 0 and 10 (inclusive).\n\n'
            'Request: Calculate 15% of 847.50, rounded to 2 decimal places.'
        ),
        oracle=OracleConfig(
            type="unit_test",
            required_keys=["tool", "arguments"],
            assertions=[{"field": "tool", "op": "eq", "value": "calculator"}],
        ),
    ),

    # -------------------------------------------------------------------------
    # SUMMARIZATION — text path (length/format constraint testing)
    # -------------------------------------------------------------------------
    PromptItem(
        id="sum-001",
        task_family="summarization",
        prompt_text=(
            'Summarize the following text in EXACTLY one sentence of no more than 30 words. '
            'Output ONLY the summary sentence, nothing else.\n\n'
            'Text: "The global electric vehicle market experienced unprecedented growth in 2023, '
            'with sales surpassing 14 million units worldwide. This represents a 35% increase compared '
            'to 2022, driven primarily by aggressive government subsidies in China, Europe, and North America, '
            'as well as significant price reductions by major manufacturers. Battery technology improvements '
            'have extended average range beyond 400 kilometers per charge."'
        ),
        oracle=OracleConfig(type="parser", max_output_chars=300),
    ),
    PromptItem(
        id="sum-002",
        task_family="summarization",
        prompt_text=(
            'Write an executive summary of this meeting transcript in 3 bullet points. '
            'Output ONLY the 3 bullet points, each starting with "•". No intro, no outro.\n\n'
            'Transcript: "Q3 results were mixed. Revenue hit $4.2M, up 12% YoY, but below the $4.8M target. '
            'Marketing spend was $800K over budget due to the new brand campaign. '
            'The board approved a $2M investment in the APAC expansion for Q4. '
            'Headcount will increase by 15 engineers over the next two quarters. '
            'Product roadmap prioritization was deferred to the next meeting."'
        ),
        oracle=OracleConfig(type="parser", max_output_chars=600),
    ),
    PromptItem(
        id="sum-003",
        task_family="summarization",
        prompt_text=(
            'Summarize this research abstract in plain English for a non-technical audience. '
            'Keep it under 50 words. Output ONLY the plain-English summary.\n\n'
            'Abstract: "We present a novel transformer-based architecture that achieves state-of-the-art '
            'performance on the GLUE benchmark with 23% fewer parameters than comparable models. '
            'Our approach leverages sparse attention mechanisms and dynamic layer dropping during inference, '
            'reducing latency by 41% while maintaining 98.2% of baseline accuracy across all evaluated tasks."'
        ),
        oracle=OracleConfig(type="parser", max_output_chars=400),
    ),
    PromptItem(
        id="sum-004",
        task_family="summarization",
        prompt_text=(
            'Summarize the risk factors in this paragraph as a JSON list. '
            'Output ONLY: {"risks": ["risk1", "risk2", ...]} — include all distinct risks.\n\n'
            'Text: "The project faces several challenges: budget overruns due to vendor delays, '
            'key personnel attrition risk as two senior engineers are interviewing elsewhere, '
            'regulatory uncertainty in three target markets, and technical debt in the legacy codebase '
            'that may slow feature delivery."'
        ),
        oracle=OracleConfig(
            type="unit_test",
            required_keys=["risks"],
            assertions=[{"field": "risks", "op": "isinstance", "value": "list"}],
        ),
    ),
    PromptItem(
        id="sum-005",
        task_family="summarization",
        prompt_text=(
            'Generate a tweet-length summary (max 280 characters) of this article excerpt. '
            'Output ONLY the tweet text. No hashtags unless naturally relevant.\n\n'
            'Article: "Scientists at MIT have developed a new material that can absorb carbon dioxide '
            'directly from the atmosphere at room temperature, with 90% efficiency. The material, '
            'a modified metal-organic framework, can be regenerated using low-grade heat, making it '
            'potentially viable for large-scale deployment. The team estimates costs could reach $50 per ton CO2 '
            'at scale, competitive with current carbon capture methods."'
        ),
        oracle=OracleConfig(type="parser", max_output_chars=350),
    ),

    # -------------------------------------------------------------------------
    # TRANSFORMATION — Instructor path
    # -------------------------------------------------------------------------
    PromptItem(
        id="tr-001",
        task_family="transformation",
        prompt_text=(
            'Convert this CSV row to a structured record.\n\n'
            'CSV header: name,age,city,occupation\n'
            'CSV row: "Maria Garcia",28,Barcelona,architect'
        ),
        schema_name="CsvToJson",
        oracle=OracleConfig(
            type="structured",
            assertions=[
                {"field": "name", "op": "not_null"},
                {"field": "age", "op": "not_null"},
            ],
        ),
    ),
    PromptItem(
        id="tr-002",
        task_family="transformation",
        prompt_text=(
            'Normalize this US phone number to E.164 international format (+1XXXXXXXXXX).\n\n'
            'Input: (415) 555-0123'
        ),
        schema_name="NormalizedPhone",
        oracle=OracleConfig(
            type="structured",
            assertions=[{"field": "normalized", "op": "contains", "value": "+1"}],
        ),
    ),
    PromptItem(
        id="tr-003",
        task_family="transformation",
        prompt_text=(
            'Redact all personally identifiable information (PII) from this text. '
            'Replace each PII item with [REDACTED]. Count the number of PII items redacted.\n\n'
            'Text: "Please contact John Smith at jsmith@email.com or call 555-867-5309. '
            'His SSN is 123-45-6789."'
        ),
        schema_name="RedactedText",
        oracle=OracleConfig(
            type="structured",
            assertions=[
                {"field": "redacted_text", "op": "contains", "value": "[REDACTED]"},
                {"field": "pii_count", "op": "isinstance", "value": "int"},
            ],
        ),
    ),
    PromptItem(
        id="tr-004",
        task_family="transformation",
        prompt_text=(
            'Convert this informal Slack message into a formal business email. '
            'Generate both a subject line and email body.\n\n'
            'Slack message: "hey team!! just fyi the deploy is done, everything looks good 🎉 '
            'staging is up if anyone wants to test. ping me if anything looks weird"'
        ),
        schema_name="FormalEmail",
        oracle=OracleConfig(
            type="structured",
            assertions=[
                {"field": "subject", "op": "not_null"},
                {"field": "body", "op": "min_length", "value": 50},
            ],
        ),
    ),
    PromptItem(
        id="tr-005",
        task_family="transformation",
        prompt_text=(
            'Convert these variable names from snake_case to camelCase.\n\n'
            'Items: ["user_name", "first_name", "last_login_date", "account_balance", "is_active"]'
        ),
        schema_name="CamelCaseList",
        oracle=OracleConfig(
            type="structured",
            assertions=[{"field": "converted", "op": "isinstance", "value": "list"}],
        ),
    ),

    # -------------------------------------------------------------------------
    # MULTI-CONSTRAINT — text path (hardest; tests competing constraints)
    # -------------------------------------------------------------------------
    PromptItem(
        id="mc-001",
        task_family="multi_constraint",
        prompt_text=(
            'You must simultaneously satisfy ALL of these constraints:\n'
            '1. Output ONLY valid JSON\n'
            '2. The JSON must have exactly 3 keys: "summary", "sentiment", "word_count"\n'
            '3. "sentiment" must be exactly one of: positive, negative, neutral\n'
            '4. "word_count" must be an integer equal to the number of words in the original text\n'
            '5. "summary" must be 10 words or fewer\n\n'
            'Text: "The quarterly results exceeded all analyst expectations with revenue '
            'growing 42% year over year and margins expanding for the third consecutive quarter."'
        ),
        oracle=OracleConfig(
            type="unit_test",
            required_keys=["summary", "sentiment", "word_count"],
            assertions=[
                {"field": "sentiment", "op": "in", "value": ["positive", "negative", "neutral"]},
                {"field": "word_count", "op": "isinstance", "value": "int"},
                {"field": "summary", "op": "max_length", "value": 80},
            ],
        ),
    ),
    PromptItem(
        id="mc-002",
        task_family="multi_constraint",
        prompt_text=(
            'Constraints you MUST follow:\n'
            '1. Respond in ONLY valid JSON with these exact top-level keys: "primary_topic", "secondary_topic"\n'
            '2. Primary and secondary topics must be DIFFERENT from each other\n'
            '3. Both must come from this exact list: technology, finance, health, politics, environment, sports\n'
            '4. Format: {"primary_topic": "<value>", "secondary_topic": "<value>"}\n\n'
            'Text: "The WHO announced new guidelines on carbon emission limits for hospitals, '
            'citing both public health concerns and climate impact."'
        ),
        oracle=OracleConfig(
            type="unit_test",
            required_keys=["primary_topic", "secondary_topic"],
            assertions=[
                {"field": "primary_topic", "op": "in", "value": ["technology", "finance", "health", "politics", "environment", "sports"]},
                {"field": "secondary_topic", "op": "in", "value": ["technology", "finance", "health", "politics", "environment", "sports"]},
            ],
        ),
    ),
    PromptItem(
        id="mc-003",
        task_family="multi_constraint",
        prompt_text=(
            'All of these constraints are mandatory:\n'
            '1. Output ONLY valid JSON, no markdown\n'
            '2. Generate exactly 3 interview questions for this job description\n'
            '3. Each question must be stored in "questions" as an array of objects\n'
            '4. Each object must have "question" (string) and "type" ("behavioral"|"technical"|"situational")\n'
            '5. You must include at least one of each type\n\n'
            'Job: "Senior Data Engineer. Must have 5+ years Python, SQL, Spark. '
            'Will lead a team of 4 junior engineers."'
        ),
        oracle=OracleConfig(
            type="unit_test",
            required_keys=["questions"],
            assertions=[{"field": "questions", "op": "isinstance", "value": "list"}],
        ),
    ),
    PromptItem(
        id="mc-004",
        task_family="multi_constraint",
        prompt_text=(
            'Follow ALL constraints:\n'
            '1. Output ONLY valid JSON with top-level keys: "spanish", "french", "original_language"\n'
            '2. Translate the text to Spanish AND French\n'
            '3. Detect and include the original language\n'
            '4. Translations must preserve the formal register of the original\n\n'
            'Text: "We regret to inform you that your application has not been successful at this time."'
        ),
        oracle=OracleConfig(
            type="unit_test",
            required_keys=["spanish", "french", "original_language"],
            assertions=[
                {"field": "spanish", "op": "not_null"},
                {"field": "french", "op": "not_null"},
            ],
        ),
    ),
    PromptItem(
        id="mc-005",
        task_family="multi_constraint",
        prompt_text=(
            'Strict output requirements:\n'
            '1. Valid JSON only, no other text\n'
            '2. Keys must be exactly: "timestamp", "error_level", "message", "affected_service"\n'
            '3. error_level must be normalized to one of: DEBUG, INFO, WARN, ERROR, FATAL\n'
            '4. Timestamp must be in ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ)\n'
            '5. If a field cannot be extracted, use null\n\n'
            'Log: "[2024-03-15 14:23:07] CRITICAL: Connection pool exhausted in auth-service. '
            'All 50 connections in use. New requests failing."'
        ),
        oracle=OracleConfig(
            type="unit_test",
            required_keys=["timestamp", "error_level", "message", "affected_service"],
            assertions=[
                {"field": "error_level", "op": "in", "value": ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]},
                {"field": "message", "op": "not_null"},
            ],
        ),
    ),
]
