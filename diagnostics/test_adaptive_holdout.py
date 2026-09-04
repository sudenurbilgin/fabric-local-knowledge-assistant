import hashlib
import re
import sys
import time
from statistics import mean

from foundry_local_sdk import FoundryLocalManager

from fabric_rag.rag import RAGSession
from fabric_rag.retrieval import load_stored_chunks
from .test_adaptive_context_expansion import (
    INSUFFICIENT_CONTEXT_PATTERN,
    adaptive_answer,
    normalize,
    retrieve_once,
)
from .retrieval_performance_baseline import (
    BASELINE_CHUNK_COUNT,
    BASELINE_REQUIREMENT_MESSAGE,
    require_baseline_corpus,
)


ADAPTIVE_THRESHOLD = 0.528094
EXPECTED_MODEL = "Phi-4-mini-instruct-generic-cpu:5"
EXPECTED_PROVIDER = "CPUExecutionProvider"
CITATION_PATTERN = re.compile(r"\[([0-9]+)\]")
MARKDOWN_URL_PATTERN = re.compile(r"\[[^\]]+\]\([^\)]+\)|https?://|www\.")
PROTECTED_FILES = [
    "src/fabric_rag/rag.py",
    "src/fabric_rag/retrieval.py",
    "streamlit_app.py",
    "rag.db",
    "src/fabric_rag/documents.py",
    "src/fabric_rag/knowledge_base.py",
]


SUPPORTED_HOLDOUT = [
    {
        "topic": "Shortcut propagation",
        "question": (
            "How do OneLake shortcuts expose external or cross-workspace data "
            "without duplicating it, and what happens when the source changes?"
        ),
        "reason": (
            "OneLake chunks 5-6 explain shortcut references, no-copy access, "
            "and immediate visibility of source updates across a chunk boundary."
        ),
        "evidence": [
            {
                "source": "onelake-overview.md",
                "chunk": 5,
                "terms": ["A shortcut is a reference", "external to OneLake"],
            },
            {
                "source": "onelake-overview.md",
                "chunk": 6,
                "terms": ["without copying it", "changes are immediately visible"],
            },
        ],
        "answer_groups": [
            ["shortcut"],
            ["without copying", "without duplicating", "no copy"],
            [
                "source data updates",
                "source data changes",
                "source changes",
                "source updates",
            ],
            [
                "immediately visible",
                "immediately reflected",
                "automatically visible",
                "reflected",
            ],
        ],
    },
    {
        "topic": "Cross-engine OneLake security",
        "question": (
            "How are OneLake security roles enforced when the same data is "
            "accessed through SQL, a Spark notebook, and a Power BI report?"
        ),
        "reason": (
            "OneLake chunk 4 states that granular roles are stored once and "
            "automatically enforced across SQL, Spark, and Power BI."
        ),
        "evidence": [
            {
                "source": "onelake-overview.md",
                "chunk": 4,
                "terms": [
                    "OneLake security roles",
                    "query via SQL",
                    "run a Spark notebook",
                    "view a Power BI report",
                    "automatically enforces them",
                ],
            }
        ],
        "answer_groups": [
            ["security roles", "permissions"],
            ["SQL"],
            ["Spark"],
            ["Power BI"],
            [
                "automatically",
                "same rule",
                "consistent",
                "uniformly",
                "across all analytics experiences",
                "unified approach",
            ],
        ],
    },
    {
        "topic": "Open table interoperability",
        "question": (
            "How does OneLake let Delta Lake and Apache Iceberg readers work "
            "with the same tables without manual format conversion?"
        ),
        "reason": (
            "OneLake chunk 9 describes metadata virtualization in both Delta-to-"
            "Iceberg and Iceberg-to-Delta directions."
        ),
        "evidence": [
            {
                "source": "onelake-overview.md",
                "chunk": 9,
                "terms": [
                    "metadata virtualization",
                    "Iceberg tables can be read as Delta Lake tables",
                    "Delta Lake tables can be read by external Iceberg readers",
                    "without manual conversion",
                ],
            }
        ],
        "answer_groups": [
            ["metadata virtualization", "virtual metadata"],
            ["Iceberg"],
            ["Delta Lake", "Delta"],
            [
                "without manual conversion",
                "without manual format conversion",
                "without converting",
                "no manual conversion",
            ],
        ],
    },
    {
        "topic": "Lakehouse table visibility",
        "question": (
            "Why might a file placed in a lakehouse not appear in the SQL "
            "analytics endpoint, and what automatic steps occur for a supported "
            "table in the Tables folder?"
        ),
        "reason": (
            "Lakehouse chunks 6-7 explain the Delta-only visibility rule and the "
            "validation, metadata extraction, and metastore registration sequence."
        ),
        "evidence": [
            {
                "source": "lakehouse-overview.md",
                "chunk": 6,
                "terms": ["Only Delta tables appear", "Parquet, CSV"],
            },
            {
                "source": "lakehouse-overview.md",
                "chunk": 7,
                "terms": [
                    "Validates the file",
                    "Extracts metadata",
                    "Registers the table in the metastore",
                ],
            },
        ],
        "answer_groups": [
            ["only Delta", "Delta tables"],
            [
                "Parquet",
                "CSV",
                "other formats",
                "non-Delta",
                "not in the Delta format",
                "not converted to Delta",
            ],
            ["validates", "validation"],
            ["extracts metadata", "metadata"],
            ["registers", "registration", "metastore"],
        ],
    },
    {
        "topic": "Lakehouse multitasking",
        "question": (
            "How does the Fabric lakehouse interface preserve ongoing work and "
            "user context when someone switches between browser tabs?"
        ),
        "reason": (
            "Lakehouse chunk 7 documents continued operations, retained selections, "
            "background reloads, and scoped notifications."
        ),
        "evidence": [
            {
                "source": "lakehouse-overview.md",
                "chunk": 7,
                "terms": [
                    "Preserve running operations",
                    "Retain your context",
                    "Non-blocking list reload",
                    "Scoped notifications",
                ],
            }
        ],
        "answer_groups": [
            ["continue running", "continue", "preserve running"],
            ["context", "selected tables", "stay open"],
            ["background", "non-blocking"],
            ["notifications"],
        ],
    },
    {
        "topic": "ETL and ELT timing",
        "question": (
            "How do ETL and ELT differ in when transformation happens, and how "
            "can Fabric Data Factory use both approaches?"
        ),
        "reason": (
            "Data Factory chunk 3 contrasts transform-before-load with load-before-"
            "transform and explicitly says both can be combined."
        ),
        "evidence": [
            {
                "source": "data-factory-overview.md",
                "chunk": 3,
                "terms": [
                    "Transform your data before loading",
                    "Load raw data first",
                    "Fabric Data Factory supports both",
                    "Combine both approaches",
                ],
            }
        ],
        "answer_groups": [
            ["before loading", "before it is loaded"],
            [
                "load raw data first",
                "load first",
                "after the data is loaded",
                "transformation occurs after",
            ],
            ["supports both", "combine both", "both approaches"],
        ],
    },
    {
        "topic": "Mirroring versus shortcuts",
        "question": (
            "How do Mirroring and Shortcuts differ as Fabric ingestion methods "
            "when avoiding traditional ETL pipelines or extra data copies?"
        ),
        "reason": (
            "Lifecycle chunk 3 contrasts continuous replication without an ETL "
            "pipeline with no-copy external-storage virtualization."
        ),
        "evidence": [
            {
                "source": "data-lifecycle.md",
                "chunk": 3,
                "terms": [
                    "Mirroring",
                    "without building ETL pipelines",
                    "Shortcuts",
                    "no-copy data virtualization",
                ],
            }
        ],
        "answer_groups": [
            ["Mirroring"],
            ["replication", "replicates"],
            ["Shortcuts"],
            ["no-copy", "without copying", "reference"],
        ],
    },
    {
        "topic": "Data Factory Copilot",
        "question": (
            "How can Copilot help a Data Factory developer create, understand, "
            "and troubleshoot pipelines or dataflows?"
        ),
        "reason": (
            "Data Factory chunk 7 covers natural-language creation/editing, "
            "summaries of existing work, and error explanations with suggested fixes."
        ),
        "evidence": [
            {
                "source": "data-factory-overview.md",
                "chunk": 7,
                "terms": [
                    "design, edit, and manage pipelines and dataflows",
                    "plain-English prompts",
                    "summarizes your existing dataflow queries and pipelines",
                    "explains what went wrong and suggests ways to fix it",
                ],
            }
        ],
        "answer_groups": [
            ["natural language", "plain-English", "prompts"],
            ["design", "create", "edit"],
            ["summarizes", "summarize"],
            ["errors", "went wrong", "troubleshoot"],
            ["suggests", "fix"],
        ],
    },
]


UNSUPPORTED_HOLDOUT = [
    {
        "topic": "F64 monthly price",
        "question": (
            "What is the current monthly price in US dollars for an F64 "
            "Microsoft Fabric capacity?"
        ),
        "reason": (
            "The corpus contains F64 resource guardrails but no currency price or "
            "monthly licensing amount."
        ),
        "related_terms": ["F64", "pricing", "subscription"],
        "missing_patterns": [
            r"monthly price",
            r"price per month",
            r"(?:USD|dollars) per month",
        ],
        "unsafe_patterns": [r"[$€£]\s*\d", r"\d[\d,.]*\s*(?:USD|dollars)"],
    },
    {
        "topic": "Data Factory regions",
        "question": "Which Azure regions currently support Microsoft Fabric Data Factory?",
        "reason": (
            "The corpus mentions regions generally but contains no Data Factory "
            "regional-availability list."
        ),
        "related_terms": ["region", "Data Factory"],
        "missing_patterns": [
            r"Data Factory (?:is )?available in (?:the following )?regions",
            r"regions (?:that )?support .*Data Factory",
        ],
        "unsafe_patterns": [r"available in (?:East|West|North|South|Central)"],
    },
    {
        "topic": "F32 capacity units",
        "question": "How many Capacity Units are assigned to the Fabric F32 SKU?",
        "reason": (
            "An F32 row appears in a Direct Lake guardrail table, but the table "
            "does not state Capacity Unit allocation."
        ),
        "related_terms": ["F32", "Fabric SKU", "capacity"],
        "missing_patterns": [r"F32.{0,80}(?:Capacity Units|\bCUs\b)"],
        "unsafe_patterns": [r"F32.{0,80}\d+\s*(?:Capacity Units|\bCUs\b)"],
    },
    {
        "topic": "OneLake administrator role",
        "question": (
            "Which Microsoft Entra directory role is required to configure "
            "OneLake tenant settings?"
        ),
        "reason": (
            "The corpus discusses tenant governance and Entra ID but does not name "
            "a required directory role for configuring OneLake tenant settings."
        ),
        "related_terms": ["Microsoft Entra", "tenant", "administrator"],
        "missing_patterns": [
            r"(?:directory role|Global Administrator|Fabric Administrator).{0,100}"
            r"OneLake tenant settings"
        ],
        "unsafe_patterns": [
            r"(?:Global|Fabric|Power Platform) Administrator",
            r"required role is",
        ],
    },
    {
        "topic": "OneLake RPO and RTO",
        "question": (
            "What recovery point objective and recovery time objective does "
            "OneLake document for a regional outage?"
        ),
        "reason": (
            "The corpus explains redundancy and optional geo-replication but gives "
            "no RPO or RTO values."
        ),
        "related_terms": ["disaster recovery", "regional", "geo-replicate"],
        "missing_patterns": [
            r"recovery point objective",
            r"recovery time objective",
            r"\bRPO\b",
            r"\bRTO\b",
        ],
        "unsafe_patterns": [
            r"(?:RPO|recovery point objective).{0,80}\d+",
            r"(?:RTO|recovery time objective).{0,80}\d+",
        ],
    },
    {
        "topic": "Pipeline activity limit",
        "question": (
            "What is the maximum number of activities allowed in one Fabric "
            "Data Factory pipeline?"
        ),
        "reason": (
            "The corpus describes pipeline activities and orchestration but states "
            "no per-pipeline activity-count limit."
        ),
        "related_terms": ["pipeline activities", "pipelines can include", "activities"],
        "missing_patterns": [
            r"maximum (?:number of )?activities.{0,80}pipeline",
            r"pipeline.{0,80}(?:activity limit|activities maximum)",
        ],
        "unsafe_patterns": [r"maximum.{0,80}\d+.{0,40}activities"],
    },
    {
        "topic": "Direct Lake GA date",
        "question": "On what date did Direct Lake become generally available?",
        "reason": (
            "The indexed article content describes Direct Lake behavior but contains "
            "no general-availability announcement or date."
        ),
        "related_terms": ["Direct Lake", "date"],
        "missing_patterns": [
            r"Direct Lake.{0,100}general(?:ly)? availab",
            r"general availability.{0,100}Direct Lake",
        ],
        "unsafe_patterns": [
            r"(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December)\s+\d{1,2},?\s+20\d{2}"
        ],
    },
    {
        "topic": "OneLake firewall ports",
        "question": (
            "Which outbound firewall ports must be opened for an application to "
            "use the OneLake ADLS Gen2 APIs?"
        ),
        "reason": (
            "The corpus confirms ADLS Gen2 API compatibility but provides no "
            "firewall-port or network-endpoint requirements."
        ),
        "related_terms": ["ADLS Gen2 APIs", "application", "connect"],
        "missing_patterns": [
            r"firewall.{0,100}port",
            r"outbound port",
            r"port\s+\d+",
        ],
        "unsafe_patterns": [r"port\s+\d+", r"\bTCP\s*\d+"],
    },
]


def file_hashes():
    return {
        path: hashlib.sha256(open(path, "rb").read()).hexdigest()
        for path in PROTECTED_FILES
    }


def validate_supported_evidence(chunks):
    lookup = {(chunk["source"], chunk["chunk_number"]): chunk for chunk in chunks}
    print("\n" + "=" * 100)
    print("SUPPORTED HOLDOUT EVIDENCE VALIDATION")
    print("=" * 100)
    for case in SUPPORTED_HOLDOUT:
        print(f"\n{case['topic']}: {case['question']}")
        print(f"Why supported: {case['reason']}")
        for evidence in case["evidence"]:
            key = (evidence["source"], evidence["chunk"])
            chunk = lookup.get(key)
            if chunk is None:
                raise RuntimeError(f"Required evidence chunk is missing: {key}")
            missing = [
                term
                for term in evidence["terms"]
                if normalize(term) not in normalize(chunk["text"])
            ]
            if missing:
                raise RuntimeError(f"Evidence terms missing from {key}: {missing}")
            print(
                f"Evidence: {evidence['source']} chunk {evidence['chunk']} | "
                f"verified terms: {', '.join(evidence['terms'])}"
            )


def validate_unsupported_absence(chunks):
    print("\n" + "=" * 100)
    print("NEAR-DOMAIN UNSUPPORTED CORPUS SEARCH")
    print("=" * 100)
    for case in UNSUPPORTED_HOLDOUT:
        joined = "\n".join(chunk["text"] for chunk in chunks)
        evidence_found = [
            pattern
            for pattern in case["missing_patterns"]
            if re.search(pattern, joined, re.IGNORECASE | re.DOTALL)
        ]
        if evidence_found:
            raise RuntimeError(
                f"Candidate is not safely unsupported: {case['topic']} matched "
                f"{evidence_found}"
            )
        related = []
        for chunk in chunks:
            hits = [
                term
                for term in case["related_terms"]
                if normalize(term) in normalize(chunk["text"])
            ]
            if hits:
                related.append(
                    f"{chunk['source']} chunk {chunk['chunk_number']} ({', '.join(hits)})"
                )
        print(f"\n{case['topic']}: {case['question']}")
        print(f"Why unsupported: {case['reason']}")
        print(
            "Answer evidence found across all "
            f"{BASELINE_CHUNK_COUNT} baseline chunks: NO"
        )
        print(
            "Related but insufficient chunks: "
            + ("; ".join(related[:8]) if related else "none")
        )


def answer_groups_present(answer, groups):
    normalized_answer = normalize(answer)

    def concept_present(option):
        normalized_option = normalize(option)
        if normalized_option in normalized_answer:
            return True

        def stems(text):
            tokens = re.findall(r"[a-z0-9]+", text)
            normalized_tokens = set()
            for token in tokens:
                if len(token) <= 2:
                    continue
                if token.endswith("ing") and len(token) > 5:
                    token = token[:-3]
                elif token.endswith("ed") and len(token) > 4:
                    token = token[:-2]
                elif token.endswith("s") and len(token) > 3:
                    token = token[:-1]
                normalized_tokens.add(token)
            return normalized_tokens

        option_stems = stems(normalized_option)
        return bool(option_stems) and option_stems.issubset(stems(normalized_answer))

    return all(
        any(concept_present(option) for option in group)
        for group in groups
    )


def context_has_expected_evidence(case, chunks):
    context = normalize("\n".join(chunk["text"] for chunk in chunks))
    return all(
        normalize(term) in context
        for evidence in case["evidence"]
        for term in evidence["terms"]
    )


def cited_context_has_expected_evidence(case, chunks, citations):
    cited_text = normalize(
        "\n".join(
            chunks[int(label) - 1]["text"]
            for label in sorted(set(citations))
            if 1 <= int(label) <= len(chunks)
        )
    )
    return answer_groups_present(cited_text, case["answer_groups"])


def evaluate_supported(case, answer, chunks):
    citations = CITATION_PATTERN.findall(answer)
    valid_citations = bool(citations) and set(citations).issubset({"1", "2", "3"})
    evidence_retrieved = context_has_expected_evidence(case, chunks)
    checks = {
        "evidence_retrieved": evidence_retrieved,
        "answer_addresses_question": answer_groups_present(answer, case["answer_groups"]),
        "citations_present_and_valid": valid_citations,
        "cited_context_contains_required_evidence": (
            valid_citations
            and cited_context_has_expected_evidence(case, chunks, citations)
        ),
        "no_insufficient_context_signal": (
            INSUFFICIENT_CONTEXT_PATTERN.search(answer) is None
        ),
        "no_markdown_urls": MARKDOWN_URL_PATTERN.search(answer) is None,
    }
    passed = all(checks.values())
    if passed:
        failure_type = "none"
    elif not evidence_retrieved:
        failure_type = "retrieval-related"
    else:
        failure_type = "generation-related"
    return checks, passed, failure_type


def evaluate_unsupported(case, answer):
    citations = CITATION_PATTERN.findall(answer)
    unsafe_assertions = [
        pattern
        for pattern in case["unsafe_patterns"]
        if re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
    ]
    checks = {
        "clear_insufficient_context_refusal": (
            INSUFFICIENT_CONTEXT_PATTERN.search(answer) is not None
        ),
        "no_source_label_citations": not citations,
        "no_detected_unsupported_assertion": not unsafe_assertions,
        "no_markdown_urls": MARKDOWN_URL_PATTERN.search(answer) is None,
    }
    return checks, all(checks.values()), unsafe_assertions


def print_retrieval(result):
    print("Top-3 retrieval:")
    for rank, chunk in enumerate(result["retrieval"]["baseline_chunks"], start=1):
        print(
            f"  {rank}. {chunk['source']} chunk {chunk['chunk_number']} | "
            f"{chunk['similarity']:.6f}"
        )
    if result["retrieval"]["neighbor"]:
        neighbor = result["retrieval"]["neighbor"][0]
        print(
            f"Neighbor: {neighbor['source']} chunk {neighbor['chunk_number']} | "
            f"{neighbor['similarity']:.6f}"
        )
    else:
        print("Neighbor: none")


def print_case(case, result):
    adaptive = result["adaptive"]
    print("\n" + "=" * 100)
    print(f"{'SUPPORTED' if result['supported'] else 'UNSUPPORTED'} HOLDOUT: {case['topic']}")
    print("=" * 100)
    print(f"Question: {case['question']}")
    print(f"Corpus rationale: {case['reason']}")
    print_retrieval(result)
    print(
        f"Threshold position: {'ABOVE' if adaptive['high_confidence'] else 'BELOW'} "
        f"{ADAPTIVE_THRESHOLD:.6f}"
    )
    print(f"Baseline insufficient signal: {'YES' if adaptive['baseline_insufficient'] else 'NO'}")
    print(f"Retry: {'YES' if adaptive['retry_triggered'] else 'NO'}")
    print(f"Final context: {adaptive['final_condition']}")
    print(f"Baseline latency: {adaptive['baseline_time']:.3f} s")
    print(f"Retry latency: {adaptive['retry_time']:.3f} s")
    print(f"Total generation time: {adaptive['total_time']:.3f} s")
    print("\nBASELINE ANSWER")
    print(adaptive["baseline_answer"])
    if adaptive["retry_triggered"]:
        print("\nFINAL ANSWER AFTER RETRY")
        print(adaptive["final_answer"])
    print("\nACCEPTANCE CHECKS")
    for name, value in result["checks"].items():
        print(f"- {name}: {'PASS' if value else 'FAIL'}")
    if result["supported"]:
        print(f"Failure classification: {result['failure_type']}")
    else:
        print(f"Unsupported retry classification: {result['retry_classification']}")
        if result["unsafe_assertions"]:
            print(f"Detected assertion patterns: {result['unsafe_assertions']}")
    print(f"RESULT: {'PASS' if result['passed'] else 'FAIL'}")


def run_case(session, case, supported, retrieval_cache):
    retrieval = retrieve_once(case["question"], retrieval_cache)
    adaptive = adaptive_answer(
        session,
        case["question"],
        retrieval,
        ADAPTIVE_THRESHOLD,
    )
    if supported:
        checks, passed, failure_type = evaluate_supported(
            case,
            adaptive["final_answer"],
            adaptive["final_chunks"],
        )
        baseline_checks, baseline_passed, _ = evaluate_supported(
            case,
            adaptive["baseline_answer"],
            retrieval["baseline_chunks"],
        )
        useful_retry = adaptive["retry_triggered"] and not baseline_passed and passed
        result = {
            "supported": True,
            "retrieval": retrieval,
            "adaptive": adaptive,
            "checks": checks,
            "baseline_checks": baseline_checks,
            "baseline_passed": baseline_passed,
            "passed": passed,
            "failure_type": failure_type,
            "useful_retry": useful_retry,
            "retry_classification": (
                "USEFUL RETRY"
                if useful_retry
                else "SAFE BUT UNNECESSARY RETRY"
                if adaptive["retry_triggered"] and passed
                else "UNSAFE RETRY"
                if adaptive["retry_triggered"] and not passed
                else "NO RETRY"
            ),
        }
    else:
        checks, passed, unsafe_assertions = evaluate_unsupported(
            case, adaptive["final_answer"]
        )
        if adaptive["retry_triggered"] and passed:
            retry_classification = "SAFE BUT UNNECESSARY RETRY"
        elif adaptive["retry_triggered"] and not passed:
            retry_classification = "UNSAFE FAILURE"
        else:
            retry_classification = "NO RETRY"
        result = {
            "supported": False,
            "retrieval": retrieval,
            "adaptive": adaptive,
            "checks": checks,
            "passed": passed,
            "unsafe_assertions": unsafe_assertions,
            "retry_classification": retry_classification,
            "useful_retry": False,
        }
    print_case(case, result)
    return result


def loaded_models():
    manager = FoundryLocalManager.instance
    return (
        [model.id for model in manager.catalog.get_loaded_models()]
        if manager is not None
        else []
    )


def classify(results):
    supported = [result for result in results if result["supported"]]
    unsupported = [result for result in results if not result["supported"]]
    supported_passes = sum(result["passed"] for result in supported)
    unsupported_passes = sum(result["passed"] for result in unsupported)
    unsafe_retries = sum(
        result["adaptive"]["retry_triggered"] and not result["passed"]
        for result in results
    )
    safe_unnecessary = sum(
        result["retry_classification"] == "SAFE BUT UNNECESSARY RETRY"
        for result in results
    )
    supported_below = sum(
        result["adaptive"]["top1"] < ADAPTIVE_THRESHOLD for result in supported
    )
    unsupported_above = sum(
        result["adaptive"]["top1"] >= ADAPTIVE_THRESHOLD for result in unsupported
    )
    if (
        supported_passes >= 7
        and unsupported_passes == 8
        and unsafe_retries == 0
        and safe_unnecessary == 0
        and supported_below == 0
        and unsupported_above == 0
    ):
        return (
            "A. HOLDOUT VALIDATION PASSED",
            "Adaptive retry is suitable for a minimal production integration, "
            "with the threshold explicitly treated as corpus-specific.",
        )
    if unsupported_passes == 8 and unsafe_retries == 0:
        return (
            "B. HOLDOUT VALIDATION PARTIALLY PASSED",
            "Keep the adaptive design, but refine the gate before production.",
        )
    return (
        "C. HOLDOUT VALIDATION FAILED",
        "Do not integrate the adaptive strategy.",
    )


def print_footer(results, final_hashes_match, model_ids):
    supported = [result for result in results if result["supported"]]
    unsupported = [result for result in results if not result["supported"]]
    supported_top1 = [result["adaptive"]["top1"] for result in supported]
    unsupported_top1 = [result["adaptive"]["top1"] for result in unsupported]
    retries = [result for result in results if result["adaptive"]["retry_triggered"]]
    useful = sum(result["useful_retry"] for result in results)
    supported_retries = sum(
        result["adaptive"]["retry_triggered"] for result in supported
    )
    unsupported_retries = sum(
        result["adaptive"]["retry_triggered"] for result in unsupported
    )
    safe_unnecessary = sum(
        result["retry_classification"] == "SAFE BUT UNNECESSARY RETRY"
        for result in results
    )
    unsafe = sum(
        result["adaptive"]["retry_triggered"] and not result["passed"]
        for result in results
    )
    classification, recommendation = classify(results)

    print("\n" + "=" * 100)
    print("ADAPTIVE HOLDOUT VALIDATION")
    print("=" * 100)
    print(f"\nFrozen threshold: {ADAPTIVE_THRESHOLD:.6f}")
    print("\nSUPPORTED HOLDOUT")
    print("Topic | Top-1 | Retry | Context | Result")
    for case, result in zip(SUPPORTED_HOLDOUT, supported):
        adaptive = result["adaptive"]
        print(
            f"{case['topic']} | {adaptive['top1']:.6f} | "
            f"{'YES' if adaptive['retry_triggered'] else 'NO'} | "
            f"{adaptive['final_condition']} | {'PASS' if result['passed'] else 'FAIL'}"
        )
    print(f"Passed: {sum(result['passed'] for result in supported)} / 8")

    print("\nNEAR-DOMAIN UNSUPPORTED HOLDOUT")
    print("Topic | Top-1 | Above threshold | Retry | Result")
    for case, result in zip(UNSUPPORTED_HOLDOUT, unsupported):
        adaptive = result["adaptive"]
        print(
            f"{case['topic']} | {adaptive['top1']:.6f} | "
            f"{'YES' if adaptive['top1'] >= ADAPTIVE_THRESHOLD else 'NO'} | "
            f"{'YES' if adaptive['retry_triggered'] else 'NO'} | "
            f"{'PASS' if result['passed'] else 'FAIL'}"
        )
    print(f"Passed: {sum(result['passed'] for result in unsupported)} / 8")

    print("\nTHRESHOLD GENERALIZATION")
    print(
        f"Supported Top-1 range: {min(supported_top1):.6f} - "
        f"{max(supported_top1):.6f}; average {mean(supported_top1):.6f}"
    )
    print(
        f"Unsupported Top-1 range: {min(unsupported_top1):.6f} - "
        f"{max(unsupported_top1):.6f}; average {mean(unsupported_top1):.6f}"
    )
    print(
        "Supported below threshold: "
        f"{sum(score < ADAPTIVE_THRESHOLD for score in supported_top1)}"
    )
    print(
        "Unsupported above threshold: "
        f"{sum(score >= ADAPTIVE_THRESHOLD for score in unsupported_top1)}"
    )

    print("\nRETRY SUMMARY")
    print(f"Retries: {len(retries)} / 16")
    print(f"Supported retries: {supported_retries}")
    print(f"Unsupported retries: {unsupported_retries}")
    print(f"Useful retries: {useful}")
    print(f"Safe unnecessary retries: {safe_unnecessary}")
    print(f"Unsafe retries: {unsafe}")
    print(
        "Average baseline latency: "
        f"{mean(result['adaptive']['baseline_time'] for result in results):.3f} s"
    )
    print(
        "Average retry latency: "
        f"{mean(result['adaptive']['retry_time'] for result in retries):.3f} s"
        if retries
        else "Average retry latency: n/a"
    )
    print(
        "Total additional retry latency: "
        f"{sum(result['adaptive']['retry_time'] for result in retries):.3f} s"
    )

    print("\nFINAL CLASSIFICATION")
    print(classification)
    print(recommendation)
    print(f"\nProtected hashes unchanged: {'YES' if final_hashes_match else 'NO'}")
    print(f"Loaded models after cleanup: {model_ids}")


def main():
    require_baseline_corpus()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    hashes_before = file_hashes()
    print("PROTECTED SHA-256 HASHES BEFORE")
    for path, digest in hashes_before.items():
        print(f"{path}: {digest}")

    chunks = load_stored_chunks()
    if len(chunks) != BASELINE_CHUNK_COUNT:
        raise RuntimeError(BASELINE_REQUIREMENT_MESSAGE)
    validate_supported_evidence(chunks)
    validate_unsupported_absence(chunks)

    retrieval_cache = {}
    session = None
    results = []
    try:
        session = RAGSession()
        if session.model != EXPECTED_MODEL:
            raise RuntimeError(
                f"Expected model {EXPECTED_MODEL}, but selected {session.model}."
            )
        if session.execution_provider != EXPECTED_PROVIDER:
            raise RuntimeError(
                f"Expected provider {EXPECTED_PROVIDER}, but selected "
                f"{session.execution_provider}."
            )
        print(f"\nMODEL: {session.model}")
        print(f"PROVIDER: {session.execution_provider}")

        for case in SUPPORTED_HOLDOUT:
            results.append(run_case(session, case, True, retrieval_cache))
        for case in UNSUPPORTED_HOLDOUT:
            results.append(run_case(session, case, False, retrieval_cache))
    finally:
        if session is not None:
            session.close()

    hashes_after = file_hashes()
    hashes_match = hashes_before == hashes_after
    print("\nPROTECTED SHA-256 HASHES AFTER")
    for path, digest in hashes_after.items():
        print(f"{path}: {digest} | {'UNCHANGED' if digest == hashes_before[path] else 'CHANGED'}")
    print_footer(results, hashes_match, loaded_models())


if __name__ == "__main__":
    main()
