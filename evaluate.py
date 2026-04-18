import json
from query import answer_query

TEST_CASES = [
    {
        "question": "What is the main topic of the ingested documents?",
        "expected_keywords": []
    },
]

def run_eval():
    results = []
    passed = 0
    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\nTest {i}: {tc['question']}")
        answer = answer_query(tc["question"])
        print(f"Answer: {answer[:300]}")

        if tc["expected_keywords"]:
            hit = any(kw.lower() in answer.lower() for kw in tc["expected_keywords"])
        else:
            hit = True

        status = "PASS" if hit else "FAIL"
        print(f"Status: {status}")
        passed += hit
        results.append({"question": tc["question"], "answer": answer, "status": status})

    print(f"\n=== Evaluation: {passed}/{len(TEST_CASES)} passed ===")

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to eval_results.json")

if __name__ == "__main__":
    run_eval()