"""
Claim Extraction & Fact-Check Reliability Evaluator
=====================================================
Evaluates a YouTube transcript fact-checking pipeline.
Uses real output from the team's API to measure accuracy,
confidence calibration, and robustness to edge cases.

Usage:
    python claim_evaluator.py              # Run full evaluation
    python claim_evaluator.py --verbose    # Detailed per-claim results
"""

import json
import sys
from datetime import datetime

# -------------------------------------------------------------------
# GROUND TRUTH: Hand-labeled verdicts for the flat earth video
# Each claim from the API output is labeled with the correct verdict
# -------------------------------------------------------------------

API_OUTPUT = {
    "video_id": "lOusCD1bfks",
    "word_count": 572,
    "factuality_score": 32.0,
    "summary": {"supported": 3, "refuted": 4, "unverifiable": 3, "total": 10},
    "claims": [
        {
            "claim": "The Earth is a spinning ball hurling through space at 67,000 mph",
            "category": "science",
            "verdict": "supported",
            "confidence": 0.95,
            "evidence_summary": "The Earth is indeed a spinning sphere, rotating on its axis and orbiting the Sun at an average speed of approximately 67,000 miles per hour.",
            "sources": ["NASA - Earth Facts", "National Geographic - Earth Overview"],
        },
        {
            "claim": "NASA claims curvature should be visible at 35,000 ft",
            "category": "science",
            "verdict": "unverifiable",
            "confidence": 0.5,
            "evidence_summary": "There is no available evidence to confirm or deny the specific claim that NASA states curvature should be visible at 35,000 feet.",
            "sources": [],
        },
        {
            "claim": "Water always seeks level",
            "category": "science",
            "verdict": "supported",
            "confidence": 0.9,
            "evidence_summary": "The principle that water seeks its own level is a fundamental concept in fluid mechanics.",
            "sources": ["Fluid Mechanics Basics", "Principles of Hydrostatics"],
        },
        {
            "claim": "NASA claims to have thousands of photos of Earth from space",
            "category": "science",
            "verdict": "unverifiable",
            "confidence": 0.5,
            "evidence_summary": "The search did not provide specific evidence regarding the number of photos NASA has of Earth from space.",
            "sources": ["NASA", "NASA - Wikipedia"],
        },
        {
            "claim": "The sun is 93 million miles away",
            "category": "science",
            "verdict": "supported",
            "confidence": 0.95,
            "evidence_summary": "The distance from the Earth to the Sun is commonly accepted to be approximately 93 million miles.",
            "sources": ["NASA Solar System Exploration", "National Geographic"],
        },
        {
            "claim": "Flights from Sydney to Santiago happen daily",
            "category": "statistics",
            "verdict": "unverifiable",
            "confidence": 0.0,
            "evidence_summary": "No search results were found to confirm or deny the frequency of flights from Sydney to Santiago.",
            "sources": [],
        },
        {
            "claim": "Antarctica is a massive ice wall guarding the edge of the Flat Earth",
            "category": "other",
            "verdict": "refuted",
            "confidence": 1.0,
            "evidence_summary": "Antarctica is not a massive ice wall; it is a continent located at the southernmost part of the Earth.",
            "sources": ["National Geographic - Antarctica", "NASA - Earth Overview"],
        },
        {
            "claim": "Gravity is a theory not a law",
            "category": "science",
            "verdict": "refuted",
            "confidence": 0.9,
            "evidence_summary": "Gravity is described as a law in the context of Newton's law of gravity.",
            "sources": ["What is gravity? - NASA", "Gravity/Gravitational Field - NASA Earthdata"],
        },
        {
            "claim": "The moon landing was faked",
            "category": "history",
            "verdict": "refuted",
            "confidence": 0.95,
            "evidence_summary": "The moon landing has been extensively documented and verified through multiple sources.",
            "sources": ["NASA Apollo Missions", "Smithsonian National Air and Space Museum"],
        },
        {
            "claim": "Stars should be a blur if Earth were spinning at 1,000 mph",
            "category": "science",
            "verdict": "refuted",
            "confidence": 0.9,
            "evidence_summary": "The claim is inaccurate. The vast distances to stars mean their apparent motion is negligible.",
            "sources": ["NASA - How Fast Does the Earth Rotate?", "Astronomy Magazine"],
        },
    ],
}

# Hand-labeled ground truth for each claim
GROUND_TRUTH = {
    "The Earth is a spinning ball hurling through space at 67,000 mph": {
        "correct_verdict": "supported",
        "notes": "Accurate. Earth orbits at ~67,000 mph.",
    },
    "NASA claims curvature should be visible at 35,000 ft": {
        "correct_verdict": "unverifiable",
        "notes": "Specific NASA statement hard to pin down. Curvature is subtly visible at that altitude but the attribution to NASA is unclear.",
    },
    "Water always seeks level": {
        "correct_verdict": "misleading",
        "notes": "Technically true in local contexts but used here as flat-earth argument. System should flag misleading framing.",
    },
    "NASA claims to have thousands of photos of Earth from space": {
        "correct_verdict": "supported",
        "notes": "NASA absolutely has thousands of Earth photos. This should be supported, not unverifiable. Evaluation failure.",
    },
    "The sun is 93 million miles away": {
        "correct_verdict": "supported",
        "notes": "Accurate. Average Earth-Sun distance is ~93 million miles.",
    },
    "Flights from Sydney to Santiago happen daily": {
        "correct_verdict": "supported",
        "notes": "Multiple airlines operate this route regularly. Should be verifiable with a simple search.",
    },
    "Antarctica is a massive ice wall guarding the edge of the Flat Earth": {
        "correct_verdict": "refuted",
        "notes": "Conspiracy claim. Correctly refuted.",
    },
    "Gravity is a theory not a law": {
        "correct_verdict": "refuted",
        "notes": "Misleading framing. Gravity is both a law (Newton) and a theory (Einstein). Correctly refuted.",
    },
    "The moon landing was faked": {
        "correct_verdict": "refuted",
        "notes": "Conspiracy claim. Correctly refuted.",
    },
    "Stars should be a blur if Earth were spinning at 1,000 mph": {
        "correct_verdict": "refuted",
        "notes": "Incorrect reasoning. Stars are too far away for rotation to cause blur. Correctly refuted.",
    },
}


# -------------------------------------------------------------------
# METRIC 1: Verdict Accuracy (Precision / Recall / F1)
# -------------------------------------------------------------------

def evaluate_verdict_accuracy(claims, ground_truth, verbose=False):
    correct = 0
    incorrect = 0
    errors = []

    for claim_data in claims:
        claim_text = claim_data["claim"]
        predicted = claim_data["verdict"]
        gt = ground_truth.get(claim_text, {})
        expected = gt.get("correct_verdict", predicted)
        notes = gt.get("notes", "")

        # Count "misleading" as incorrect if system said "supported"
        if expected == "misleading" and predicted == "supported":
            is_correct = False
        elif expected == predicted:
            is_correct = True
        else:
            is_correct = False

        if is_correct:
            correct += 1
        else:
            incorrect += 1
            errors.append({
                "claim": claim_text,
                "predicted": predicted,
                "expected": expected,
                "notes": notes,
            })

        if verbose:
            icon = "OK" if is_correct else "WRONG"
            print(f"  [{icon}] {claim_text[:60]}...")
            print(f"         Predicted: {predicted}  |  Expected: {expected}")
            if not is_correct:
                print(f"         Note: {notes}")
            print()

    total = correct + incorrect
    accuracy = correct / total if total > 0 else 0

    return {
        "correct": correct,
        "incorrect": incorrect,
        "total": total,
        "accuracy": round(accuracy, 3),
        "errors": errors,
    }


# -------------------------------------------------------------------
# METRIC 2: Confidence Calibration
# -------------------------------------------------------------------

def evaluate_confidence_calibration(claims, ground_truth):
    """
    Check if confidence scores match actual correctness.
    High confidence on wrong answers = poorly calibrated.
    """
    results = []

    for claim_data in claims:
        claim_text = claim_data["claim"]
        predicted = claim_data["verdict"]
        confidence = claim_data["confidence"]
        gt = ground_truth.get(claim_text, {})
        expected = gt.get("correct_verdict", predicted)

        if expected == "misleading" and predicted == "supported":
            is_correct = False
        elif expected == predicted:
            is_correct = True
        else:
            is_correct = False

        results.append({
            "claim": claim_text[:55] + "...",
            "confidence": confidence,
            "correct": is_correct,
        })

    # Flag overconfident wrong answers
    overconfident = [r for r in results if not r["correct"] and r["confidence"] >= 0.8]
    underconfident = [r for r in results if r["correct"] and r["confidence"] <= 0.3]

    # Average confidence for correct vs incorrect
    correct_confs = [r["confidence"] for r in results if r["correct"]]
    incorrect_confs = [r["confidence"] for r in results if not r["correct"]]

    avg_correct = sum(correct_confs) / len(correct_confs) if correct_confs else 0
    avg_incorrect = sum(incorrect_confs) / len(incorrect_confs) if incorrect_confs else 0

    return {
        "avg_confidence_correct": round(avg_correct, 3),
        "avg_confidence_incorrect": round(avg_incorrect, 3),
        "overconfident_errors": overconfident,
        "underconfident_correct": underconfident,
        "calibration_gap": round(avg_correct - avg_incorrect, 3),
    }


# -------------------------------------------------------------------
# STRESS TESTS
# -------------------------------------------------------------------

STRESS_TESTS = [
    {
        "name": "Conspiracy claims correctly refuted",
        "description": "System should refute obvious conspiracy theories",
        "check_claims": [
            "Antarctica is a massive ice wall guarding the edge of the Flat Earth",
            "The moon landing was faked",
        ],
        "expected_verdict": "refuted",
    },
    {
        "name": "Well-established facts correctly supported",
        "description": "System should support widely accepted scientific facts",
        "check_claims": [
            "The sun is 93 million miles away",
            "The Earth is a spinning ball hurling through space at 67,000 mph",
        ],
        "expected_verdict": "supported",
    },
    {
        "name": "Easily verifiable claims not marked unverifiable",
        "description": "System should not punt on claims that are easy to verify",
        "check_claims": [
            "NASA claims to have thousands of photos of Earth from space",
            "Flights from Sydney to Santiago happen daily",
        ],
        "expected_verdict_not": "unverifiable",
    },
    {
        "name": "Source quality check",
        "description": "Refuted claims should cite authoritative sources",
        "check_type": "sources_present",
        "for_verdict": "refuted",
    },
    {
        "name": "Misleading framing detection",
        "description": "System should flag claims that are technically true but misleadingly framed",
        "check_claims": ["Water always seeks level"],
        "note": "True in isolation but used as flat-earth evidence. System marked it as supported without noting the misleading context.",
    },
]


def run_stress_tests(claims):
    results = []
    claim_lookup = {c["claim"]: c for c in claims}

    for test in STRESS_TESTS:
        passed = True
        failures = []

        if "check_claims" in test and "expected_verdict" in test:
            for claim_text in test["check_claims"]:
                c = claim_lookup.get(claim_text)
                if c and c["verdict"] != test["expected_verdict"]:
                    passed = False
                    failures.append(f"'{claim_text[:50]}...' got '{c['verdict']}', expected '{test['expected_verdict']}'")

        elif "check_claims" in test and "expected_verdict_not" in test:
            for claim_text in test["check_claims"]:
                c = claim_lookup.get(claim_text)
                if c and c["verdict"] == test["expected_verdict_not"]:
                    passed = False
                    failures.append(f"'{claim_text[:50]}...' was '{c['verdict']}' but should not be")

        elif test.get("check_type") == "sources_present":
            for c in claims:
                if c["verdict"] == test["for_verdict"] and len(c.get("sources", [])) == 0:
                    passed = False
                    failures.append(f"'{c['claim'][:50]}...' refuted but has no sources")

        elif "note" in test:
            # Manual check — always flag as failed for discussion
            passed = False
            failures.append(test["note"])

        results.append({
            "name": test["name"],
            "description": test["description"],
            "passed": passed,
            "failures": failures,
        })

    return results


# -------------------------------------------------------------------
# ERROR TAXONOMY
# -------------------------------------------------------------------

def build_error_taxonomy(verdict_result, calibration_result):
    taxonomy = {}

    for err in verdict_result["errors"]:
        if err["predicted"] == "unverifiable" and err["expected"] in ("supported", "refuted"):
            etype = "false_unverifiable"
        elif err["predicted"] == "supported" and err["expected"] == "misleading":
            etype = "missed_misleading_framing"
        elif err["predicted"] == "supported" and err["expected"] == "refuted":
            etype = "false_support"
        elif err["predicted"] == "refuted" and err["expected"] == "supported":
            etype = "false_refutation"
        else:
            etype = "other_misclassification"
        taxonomy[etype] = taxonomy.get(etype, 0) + 1

    if calibration_result["overconfident_errors"]:
        taxonomy["overconfident_on_wrong_answer"] = len(calibration_result["overconfident_errors"])

    return taxonomy


# -------------------------------------------------------------------
# MAIN EVALUATION
# -------------------------------------------------------------------

def run_evaluation(verbose=False):
    claims = API_OUTPUT["claims"]

    print()
    print("=" * 62)
    print("   FACT-CHECK PIPELINE EVALUATOR")
    print("   Video: Flat Earth Claims (lOusCD1bfks)")
    print("   Pipeline factuality score: {}/100".format(API_OUTPUT["factuality_score"]))
    print("=" * 62)
    print()

    # --- Metric 1: Verdict Accuracy ---
    print("-" * 62)
    print("  METRIC 1: Verdict Accuracy")
    print("-" * 62)
    print()

    verdict_result = evaluate_verdict_accuracy(claims, GROUND_TRUTH, verbose=verbose)

    print(f"  Correct:   {verdict_result['correct']}/{verdict_result['total']}")
    print(f"  Incorrect: {verdict_result['incorrect']}/{verdict_result['total']}")
    print(f"  Accuracy:  {verdict_result['accuracy']}")

    if verdict_result["errors"] and not verbose:
        print(f"\n  Misclassified claims:")
        for err in verdict_result["errors"]:
            print(f"    x '{err['claim'][:55]}...'")
            print(f"      Got: {err['predicted']}  Expected: {err['expected']}")

    # --- Metric 2: Confidence Calibration ---
    print()
    print("-" * 62)
    print("  METRIC 2: Confidence Calibration")
    print("-" * 62)
    print()

    cal_result = evaluate_confidence_calibration(claims, GROUND_TRUTH)

    print(f"  Avg confidence on CORRECT verdicts:   {cal_result['avg_confidence_correct']}")
    print(f"  Avg confidence on INCORRECT verdicts: {cal_result['avg_confidence_incorrect']}")
    print(f"  Calibration gap:                      {cal_result['calibration_gap']}")

    if cal_result["overconfident_errors"]:
        print(f"\n  OVERCONFIDENT wrong answers (confidence >= 0.8):")
        for oc in cal_result["overconfident_errors"]:
            print(f"    ! {oc['claim']}  (confidence: {oc['confidence']})")

    if cal_result["underconfident_correct"]:
        print(f"\n  UNDERCONFIDENT correct answers (confidence <= 0.3):")
        for uc in cal_result["underconfident_correct"]:
            print(f"    ? {uc['claim']}  (confidence: {uc['confidence']})")

    # --- Stress Tests ---
    print()
    print("-" * 62)
    print("  STRESS TESTS")
    print("-" * 62)

    stress_results = run_stress_tests(claims)
    passed_count = sum(1 for r in stress_results if r["passed"])

    for r in stress_results:
        icon = "PASS" if r["passed"] else "FAIL"
        print(f"\n  [{icon}] {r['name']}")
        print(f"    {r['description']}")
        if r["failures"]:
            for f in r["failures"]:
                print(f"    -> {f}")

    print(f"\n  Result: {passed_count}/{len(stress_results)} passed")

    # --- Error Taxonomy ---
    print()
    print("-" * 62)
    print("  ERROR TAXONOMY")
    print("-" * 62)
    print()

    taxonomy = build_error_taxonomy(verdict_result, cal_result)

    if taxonomy:
        for etype, count in sorted(taxonomy.items(), key=lambda x: -x[1]):
            label = etype.replace("_", " ").title()
            bar = "#" * (count * 5)
            print(f"    {label:<35} {count}  {bar}")
    else:
        print("    No errors found.")

    # --- Summary ---
    print()
    print("=" * 62)
    print("  EVALUATION SUMMARY")
    print("=" * 62)
    print(f"    Verdict accuracy:     {verdict_result['accuracy']} ({verdict_result['correct']}/{verdict_result['total']})")
    print(f"    Calibration gap:      {cal_result['calibration_gap']}")
    print(f"    Stress tests passed:  {passed_count}/{len(stress_results)}")
    print(f"    Total error types:    {len(taxonomy)}")
    print()
    print("  Key findings:")
    print("    1. System correctly refutes conspiracy claims (moon landing,")
    print("       flat earth) with high confidence and good sources.")
    print("    2. System is too quick to mark claims 'unverifiable' when")
    print("       simple web searches would confirm them (NASA photos,")
    print("       Sydney-Santiago flights).")
    print("    3. System misses misleading framing — 'water seeks level'")
    print("       is technically true but used as flat-earth evidence.")
    print("    4. Confidence calibration needs work — system is sometimes")
    print("       highly confident on incorrect verdicts.")
    print()

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "video_id": API_OUTPUT["video_id"],
        "pipeline_factuality_score": API_OUTPUT["factuality_score"],
        "verdict_accuracy": verdict_result,
        "confidence_calibration": cal_result,
        "stress_tests": {"passed": passed_count, "total": len(stress_results), "details": stress_results},
        "error_taxonomy": taxonomy,
    }

    filename = f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Report saved to: {filename}")
    print()


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    run_evaluation(verbose=verbose)
