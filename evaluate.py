"""
AMO PAY AI — Confusion Matrix Evaluation Runner
Sends test questions to the AI and evaluates responses
"""

import httpx
import json
import asyncio
import time
from datetime import datetime

# ── Config ─────────────────────────────────────
AI_SERVER_URL = "http://localhost:8000/chat"
TEST_CASES_FILE = "test_cases.json"
RESULTS_FILE = "eval_results.json"

# ── Load test cases ────────────────────────────
with open(TEST_CASES_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)
    test_cases = test_data["test_cases"]

print(f"📋 Loaded {len(test_cases)} test cases\n")

# ── Language detection helper ──────────────────
def detect_response_language(text: str) -> str:
    """Detect if AI response is in English or Kinyarwanda"""
    import re
    text_lower = text.lower()
    words = re.findall(r'\w+', text_lower)

    kinyarwanda_markers = [
        'muraho', 'bite', 'mwaramutse', 'ndashaka', 'mfasha',
        'ndabaza', 'uburyo', 'kohereza', 'amafaranga', 'nshaka',
        'nagira', 'ubwoko', 'urakoze', 'murakaza', 'mbwira',
        'nkurikije', 'gukora', 'hamwe', 'gute', 'nifuza',
        'yego', 'oya', 'muri', 'kandi', 'ariko', 'ese',
        'nta', 'iki', 'aho', 'ijambo', 'ibanga', 'ingano', 
        'umupaka', 'ryari', 'he', 'ninde', 'ibihe', 'ayahe'
    ]

    english_markers = [
        'the', 'you', 'your', 'can', 'will', 'is', 'how', 'what',
        'send', 'money', 'transfer', 'fee', 'limit', 'merchant',
        'verify', 'account', 'help', 'please', 'thanks'
    ]

    kw_hits = 0
    for m in kinyarwanda_markers:
        if m in words or any(m in w for w in words if "'" in w):
            kw_hits += 1

    en_hits = 0
    for m in english_markers:
        if m in words:
            en_hits += 1

    if kw_hits > en_hits:
        return "Kinyarwanda"
    elif en_hits > kw_hits:
        return "English"
    else:
        return "Unknown"

# ── Topic coverage checker ─────────────────────
def check_topic_coverage(response: str, expected_topics: list) -> dict:
    """Check how many expected topics are covered in the response"""
    response_lower = response.lower()
    covered = []
    missing = []

    for topic in expected_topics:
        if topic.lower() in response_lower:
            covered.append(topic)
        else:
            missing.append(topic)

    coverage = len(covered) / len(expected_topics) if expected_topics else 0
    return {
        "covered": covered,
        "missing": missing,
        "coverage_score": round(coverage * 100, 1)
    }

# ── Classify result ────────────────────────────
def classify_result(language_correct: bool, coverage_score: float, category: str) -> str:
    """
    Enhanced classification logic:
    - TP: Correct results for in-scope domains.
    - TN: Correct blocking for out-of-scope domains.
    - FP: Hallucinations or wrong language.
    - FN: Missed information or system failures.
    """
    is_oos = category == "Out of Scope"
    
    if is_oos:
        # For Out of Scope, we WANT low coverage (blocking)
        if language_correct and coverage_score < 30:
            return "TN" # Correctly handled out-of-scope
        elif not language_correct:
            return "FP" # Hallucination/Wrong language
        else:
            return "FN" # Failed to block out-of-scope query
    else:
        # Standard in-scope classification
        if language_correct and coverage_score >= 50:
            return "TP"
        elif not language_correct and coverage_score >= 50:
            return "FP"
        elif language_correct and coverage_score < 50:
            return "FN" # Missed coverage
        else:
            return "FN"

# ── Main evaluation ────────────────────────────
async def run_evaluation():
    results = []
    summary = {
        "total": 0,
        "TP": 0, "FP": 0, "TN": 0, "FN": 0,
        "language_correct": 0,
        "language_wrong": 0,
        "high_coverage": 0,
        "low_coverage": 0,
        "by_category": {},
        "by_language": {
            "English": {"total": 0, "language_correct": 0, "avg_coverage": 0},
            "Kinyarwanda": {"total": 0, "language_correct": 0, "avg_coverage": 0}
        }
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, test in enumerate(test_cases):
            print(f"[{i+1}/{len(test_cases)}] Testing: {test['id']} — {test['question'][:50]}...")

            try:
                start_time = time.time()

                # Send request to AI server
                response = await client.post(
                    AI_SERVER_URL,
                    json={
                        "message": test["question"],
                        "history": []
                    }
                )

                elapsed = round(time.time() - start_time, 2)

                if response.status_code != 200:
                    print(f"  ❌ Server error: {response.status_code}")
                    summary["total"] += 1
                    summary["FN"] += 1  # Count error as failure (FN)
                    results.append({
                        **test,
                        "status": "ERROR",
                        "error": f"HTTP {response.status_code}",
                        "response": None,
                        "response_time": elapsed
                    })
                    continue

                data = response.json()
                ai_reply = data.get("reply", "")
                model_used = data.get("model", "unknown")

                # Update total count
                summary["total"] += 1

                # Evaluate
                detected_lang = detect_response_language(ai_reply)
                language_correct = detected_lang == test["expected_language"]
                topic_eval = check_topic_coverage(ai_reply, test["expected_topics"])
                result_class = classify_result(language_correct, topic_eval["coverage_score"], test["category"])

                # Build result
                result = {
                    "id": test["id"],
                    "category": test["category"],
                    "language": test["language"],
                    "question": test["question"],
                    "expected_language": test["expected_language"],
                    "detected_language": detected_lang,
                    "language_correct": language_correct,
                    "coverage_score": topic_eval["coverage_score"],
                    "topics_covered": topic_eval["covered"],
                    "topics_missing": topic_eval["missing"],
                    "result_class": result_class,
                    "response_time": elapsed,
                    "model_used": model_used,
                    "ai_reply": ai_reply,
                    "status": "SUCCESS"
                }

                results.append(result)

                # Update summary
                summary[result_class] += 1

                if language_correct:
                    summary["language_correct"] += 1
                else:
                    summary["language_wrong"] += 1

                if topic_eval["coverage_score"] >= 50:
                    summary["high_coverage"] += 1
                else:
                    summary["low_coverage"] += 1

                # By category
                cat = test["category"]
                if cat not in summary["by_category"]:
                    summary["by_category"][cat] = {"total": 0, "TP": 0, "FP": 0, "TN": 0, "FN": 0, "avg_coverage": 0}
                summary["by_category"][cat]["total"] += 1
                summary["by_category"][cat][result_class] += 1
                summary["by_category"][cat]["avg_coverage"] += topic_eval["coverage_score"]

                # By language
                lang = test["language"]
                summary["by_language"][lang]["total"] += 1
                if language_correct:
                    summary["by_language"][lang]["language_correct"] += 1
                summary["by_language"][lang]["avg_coverage"] += topic_eval["coverage_score"]

                # Print result
                icon = "✅" if result_class == "TP" else "⚠️" if result_class in ["FP", "TN"] else "❌"
                print(f"  {icon} [{result_class}] Lang: {detected_lang} ({'' if language_correct else 'WRONG - expected ' + test['expected_language']}) | Coverage: {topic_eval['coverage_score']}% | Time: {elapsed}s")

                # Maximized delay to avoid rate limiting
                await asyncio.sleep(7)

            except Exception as e:
                print(f"  ❌ Exception: {str(e)}")
                summary["total"] += 1
                summary["FN"] += 1  # Count exception as failure (FN)
                results.append({
                    **test,
                    "status": "ERROR",
                    "error": str(e),
                    "response": None
                })

    # Finalize averages
    for cat in summary["by_category"]:
        total = summary["by_category"][cat]["total"]
        if total > 0:
            summary["by_category"][cat]["avg_coverage"] = round(
                summary["by_category"][cat]["avg_coverage"] / total, 1
            )

    for lang in summary["by_language"]:
        total = summary["by_language"][lang]["total"]
        if total > 0:
            summary["by_language"][lang]["avg_coverage"] = round(
                summary["by_language"][lang]["avg_coverage"] / total, 1
            )
            summary["by_language"][lang]["language_accuracy"] = round(
                summary["by_language"][lang]["language_correct"] / total * 100, 1
            )

    # Calculate metrics
    TP = summary["TP"]
    FP = summary["FP"]
    FN = summary["FN"]
    TN = summary["TN"]

    precision = round(TP / (TP + FP) * 100, 1) if (TP + FP) > 0 else 0
    recall = round(TP / (TP + FN) * 100, 1) if (TP + FN) > 0 else 0
    f1 = round(2 * precision * recall / (precision + recall), 1) if (precision + recall) > 0 else 0
    accuracy = round((TP + TN) / summary["total"] * 100, 1) if summary["total"] > 0 else 0
    language_accuracy = round(summary["language_correct"] / summary["total"] * 100, 1) if summary["total"] > 0 else 0

    summary["metrics"] = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "language_accuracy": language_accuracy
    }

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "results": results
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print final summary
    print("\n" + "="*60)
    print("📊 EVALUATION COMPLETE")
    print("="*60)
    print(f"Total Tests:        {summary['total']}")
    print(f"✅ True Positive:   {TP} (correct lang + good coverage)")
    print(f"⚠️  False Positive:  {FP} (wrong lang + good coverage)")
    print(f"🔵 True Negative:   {TN} (correct lang + low coverage)")
    print(f"❌ False Negative:  {FN} (wrong lang + low coverage)")
    print(f"\n📈 Metrics:")
    print(f"   Accuracy:         {accuracy}%")
    print(f"   Language Accuracy:{language_accuracy}%")
    print(f"   Precision:        {precision}%")
    print(f"   Recall:           {recall}%")
    print(f"   F1 Score:         {f1}%")
    print(f"\n💾 Results saved to: {RESULTS_FILE}")

    return output

if __name__ == "__main__":
    asyncio.run(run_evaluation())