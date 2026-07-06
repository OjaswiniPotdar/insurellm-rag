from datetime import datetime
import os

from dotenv import load_dotenv
from openai import OpenAI
from timescale_vector import client

from app.database.vector_store import VectorStore
from app.services.synthesizer import Synthesizer



load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Check your .env file."
    )

print(f"Loaded API Key: {api_key[:20]}...")


try:
    test_client = OpenAI(api_key=api_key)

    test_response = test_client.embeddings.create(
        model="text-embedding-3-small",
        input=["test"]
    )

    print("OpenAI API Key is working.\n")

except Exception as e:
    print("\nOpenAI Authentication Failed")
    print(e)
    exit()



vec = VectorStore()



def run_query(question, **search_kwargs):

    print("\n================================================")
    print(f"Question: {question}")
    print("------------------------------------------------")

    try:
        results = vec.search(
            question,
            limit=3,
            **search_kwargs
        )

        if results.empty:
            print("No relevant documents found.")
            print("================================================\n")
            return

        response = Synthesizer.generate_response(
            question=question,
            context=results
        )

        print("\nAnswer:")
        print(response.answer)

        print("\nThought process:")
        for step in response.thought_process:
            print(f"- {step}")

        print(f"\nEnough Context: {response.enough_context}")

    except Exception as e:
        print("\nERROR OCCURRED:")
        print(type(e).__name__)
        print(e)

    print("================================================\n")


run_query(
    "What types of insurance policies does the company offer?"
)

run_query(
    "What is Bizllm?"
)



metadata_filter = {
    "category": "Policy"
}

run_query(
    "What insurance policies are available?",
    metadata_filter=metadata_filter
)


pred = client.Predicates(
    "category",
    "==",
    "Claims"
)

run_query(
    "How do I submit an insurance claim?",
    predicates=pred
)


pred = (
    client.Predicates("category", "==", "Claims")
    | client.Predicates("category", "==", "Policy")
)

run_query(
    "Explain the insurance claim process.",
    predicates=pred
)


pred = (
    client.Predicates("category", "==", "Claims")
    & client.Predicates("created_at", ">", "2024-09-01")
)

run_query(
    "What documents are required to submit a claim?",
    predicates=pred
)



time_range = (
    datetime(2024, 8, 1),
    datetime(2024, 8, 30)
)

run_query(
    "Why Join Insurellm?",
    time_range=time_range
)