from retriever import retrieve_documents

test_cases = [
    {
        "query": "Give me the overview of NimbusHome Device Protection Plan",
        "expected_keywords": [
            "smart-home devices",
            "basic",
            "plus",
            "total"
        ]
    },
    {
        "query": "NimbusHome Device Protection ELIGIBILITY AND ENROLLMENT",
        "expected_keywords": [
            "60 days",
            "proof of purchase",
            "serial number",
            "enrollment fee"
        ]
    }
]


def evaluate_retrieval(k=3):

    hits = 0

    for test in test_cases:

        results = retrieve_documents(test["query"], k=k)

        retrieved_text = " ".join(
            doc.page_content.lower()
            for doc, score in results
        )

        found = any(
            keyword.lower() in retrieved_text
            for keyword in test["expected_keywords"]
        )

        if found:
            hits += 1

    hit_rate = hits / len(test_cases)

    print(f"Total Queries : {test_cases}")
    print(f"Hits          : {hits}")
    print(f"Hit@{k}        : {hit_rate:.2f}")

    print(f"Hit@{k}: {hit_rate:.2f}")


if __name__ == "__main__":
    evaluate_retrieval()