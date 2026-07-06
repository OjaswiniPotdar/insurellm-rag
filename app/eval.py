import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from app.config.settings import get_settings
from app.hybrid import HybridRetriever

load_dotenv(override=True)

settings = get_settings()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
retriever = HybridRetriever()


# ------------------------------------------------
# Evaluation Dataset
# ------------------------------------------------

evaluation_dataset = [
    {
        "question": "What is the monthly payment amount in the contract with Advantage Medical Coverage?",
        "answer": "$15,000 per month"
    },
    {
        "question": "How many covered members does the Healthllm Professional Tier support for Advantage Medical Coverage?",
        "answer": "50,000 covered members"
    },
    {
        "question": "What is the total contract value for the DriveSmart Insurance Carllm agreement?",
        "answer": "$702,000"
    },
    {
        "question": "Which company received exclusive use of Bizllm in the specialty construction insurance vertical?",
        "answer": "Continental Commercial Group"
    },
    {
        "question": "What is the response time guarantee for critical support issues in the GlobalRe Partners contract?",
        "answer": "10 minutes for P1 critical issues"
    },
    {
        "question": "How many API calls per month are included in the Fortress Business Underwriters Bizllm contract?",
        "answer": "100,000 API calls per month"
    },
    {
        "question": "Which contract includes telematics integration features such as hard braking and speeding analysis?",
        "answer": "DriveSmart Insurance for Carllm"
    },
    {
        "question": "What is the duration of the FastTrack Insurance Services Claimllm contract?",
        "answer": "18 months"
    },
    {
        "question": "Which client uses the Lifellm Starter Tier package, and what is its monthly fee?",
        "answer": "Evergreen Life Insurance, $3,500 per month"
    },
    {
        "question": "What are the renewal notice requirements for the Atlantic Risk Solutions Bizllm agreement?",
        "answer": "45 days before expiration"
    },
    {
        "question": "Which contract provides support for up to 50,000 covered members?",
        "answer": "Advantage Medical Coverage contract"
    },
    {
        "question": "What catastrophe modeling vendors are integrated into the GlobalRe Partners Rellm platform?",
        "answer": "RMS, AIR, CoreLogic, and KCC"
    },
    {
        "question": "Which agreement includes a 99.95% uptime SLA?",
        "answer": "GlobalRe Partners Rellm agreement"
    },
    {
        "question": "What is the lead pricing model used in the ConnectInsure Agency Markellm contract?",
        "answer": "$25 per qualified lead generated"
    },
    {
        "question": "Which contract includes white-label mobile apps for iOS and Android?",
        "answer": "DriveSmart Insurance for Carllm"
    },
    {
        "question": "How many states does DriveSmart Insurance currently operate in according to the contract?",
        "answer": "8 states"
    },
    {
        "question": "What are the supported languages in the GlobalRe Partners white-label platform?",
        "answer": "English, Spanish, French, German, Mandarin, Japanese, Portuguese, Dutch, Italian, Korean, Arabic, and Russian"
    },
    {
        "question": "Which client receives 500 hours annually of custom development included in their agreement?",
        "answer": "Continental Commercial Group"
    },
    {
        "question": "What payment penalty applies for late payments in the Atlantic Risk Solutions contract?",
        "answer": "1.5% per month"
    },
    {
        "question": "Which Homellm contract includes a dedicated support team available 24/7?",
        "answer": "GreenValley Insurance for Homellm"
    },
    {
        "question": "What is the total contract value of the United Healthcare Alliance Healthllm agreement?",
        "answer": "$2,784,000"
    },
    {
        "question": "Which contract includes wearable integrations such as Fitbit, Apple Watch, and Garmin?",
        "answer": "Metropolitan Life Group Lifellm contract"
    },
    {
        "question": "What is the monthly subscription fee for the Summit Commercial Insurance Bizllm Business Tier package?",
        "answer": "$6,000 per month"
    },
    {
        "question": "Which contract guarantees 99.95% uptime with financial credits for downtime?",
        "answer": "National Claims Network and United Healthcare Alliance agreements"
    },
    {
        "question": "How many active members does WellCare Insurance Co. currently support under the Healthllm Essential Tier?",
        "answer": "11,000 members"
    },
    {
        "question": "What overage fee applies if Rapid Claims Associates exceeds 5,000 claims annually?",
        "answer": "$0.90 per claim"
    },
    {
        "question": "Which contract includes blockchain-based policy verification features?",
        "answer": "Metropolitan Life Group Lifellm contract"
    },
    {
        "question": "What is the fraud detection accuracy guarantee in the Premier Adjusters Inc. Claimllm contract?",
        "answer": "85%+ accuracy"
    },
    {
        "question": "Which Homellm enterprise contract supports IoT integrations with smart home devices?",
        "answer": "SafeHaven Property Insurance contract"
    },
    {
        "question": "How many provider records and member records are migrated during the United Healthcare Alliance implementation?",
        "answer": "15,000+ provider records and 250,000+ member records"
    },
    {
        "question": "Which contract includes catastrophe claims management with satellite imagery analysis?",
        "answer": "National Claims Network Claimllm contract"
    },
    {
        "question": "What is the payment schedule for the TechDrive Insurance Carllm contract?",
        "answer": "Payments due on the 5th of each month"
    },
    {
        "question": "Which client receives 2,000 hours annually of custom development included in their agreement?",
        "answer": "Metropolitan Life Group"
    },
    {
        "question": "What is the maximum number of active policies supported under Guardian Life Partners' Growth Tier?",
        "answer": "10,000 active policies"
    },
    {
        "question": "Which agreement includes AI-powered litigation prediction for claims handling?",
        "answer": "National Claims Network Claimllm agreement"
    },
    {
        "question": "What are the response times for P1, P2, and P3 issues in the National Claims Network contract?",
        "answer": "10 minutes for P1, 30 minutes for P2, and 2 hours for P3"
    },
    {
        "question": "Which contract includes premium discounts based on IoT adoption of up to 20%?",
        "answer": "SafeHaven Property Insurance Homellm contract"
    },
    {
        "question": "What is the total contract value of the Metropolitan Life Group Lifellm Enterprise agreement?",
        "answer": "$1,098,000"
    },
    {
        "question": "Which contract provides support for 15,000+ healthcare providers and credentialing workflows?",
        "answer": "United Healthcare Alliance Healthllm agreement"
    },
    {
        "question": "What integrations are supported in the SafeHaven Property Insurance Homellm Enterprise contract?",
        "answer": "Duck Creek, Applied Epic, Guidewire ClaimCenter, Nationwide Canopy, Ting, RMS RiskLink, Salesforce CRM, and payment processors"
    },
    {
        "question": "Who founded Insurellm?",
        "answer": "Avery Lancaster"
    },
    {
        "question": "When was Insurellm founded?",
        "answer": "2015"
    },
    {
        "question": "How many employees does Insurellm currently have?",
        "answer": "32 employees"
    },
    {
        "question": "How many active contracts does Insurellm manage?",
        "answer": "32 active contracts"
    },
    {
        "question": "How many product lines does Insurellm offer?",
        "answer": "eight product lines"
    },
    {
        "question": "What was Insurellm’s first product?",
        "answer": "Markellm"
    },
    {
        "question": "Which Insurellm product focuses on claims processing?",
        "answer": "Claimllm"
    },
    {
        "question": "Which Insurellm platform is designed for reinsurance companies?",
        "answer": "Rellm"
    },
    {
        "question": "Which product is Insurellm’s commercial insurance platform?",
        "answer": "Bizllm"
    },
    {
        "question": "Which Insurellm product provides a marketplace connecting consumers with insurance providers?",
        "answer": "Markellm"
    },
    {
        "question": "Which Insurellm platform specializes in life insurance?",
        "answer": "Lifellm"
    },
    {
        "question": "Which Insurellm platform focuses on health insurance?",
        "answer": "Healthllm"
    },
    {
        "question": "Which Insurellm platform focuses on home insurance?",
        "answer": "Homellm"
    },
    {
        "question": "Which Insurellm platform focuses on auto insurance?",
        "answer": "Carllm"
    },
    {
        "question": "How many contracts does Bizllm currently have?",
        "answer": "7 contracts"
    },
    
]


# ------------------------------------------------
# Fuzzy Matching
# ------------------------------------------------

def is_correct(expected: str, answer: str) -> bool:
    """
    Check correctness using fuzzy keyword matching.
    Returns True if 60%+ of meaningful keywords from
    the expected answer appear in the generated answer.
    """
    expected_lower = expected.lower()
    answer_lower = answer.lower()

    if expected_lower in answer_lower:
        return True

    # Extract meaningful keywords (length > 2, skip stopwords)
    stopwords = {"the", "and", "for", "per", "with", "that", "this", "are", "was", "has", "have"}
    keywords = [
        w.strip(".,;:$%+")
        for w in expected_lower.split()
        if len(w.strip(".,;:$%+")) > 2 and w not in stopwords
    ]

    if not keywords:
        return False

    matches = sum(1 for kw in keywords if kw in answer_lower)
    score = matches / len(keywords)

    return score >= 0.6


# ------------------------------------------------
# Generate Answer from LLM
# ------------------------------------------------

def generate_answer(question: str, context: str) -> str:

    prompt = f"""
Use ONLY the provided context to answer the question.
Be concise and specific. Include exact numbers, names, and values when available.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=settings.openai.default_model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=300
    )

    return response.choices[0].message.content.strip()


# ------------------------------------------------
# Faithfulness Check
# ------------------------------------------------

def check_faithfulness(context: str, answer: str) -> bool:

    prompt = f"""
Determine if the answer is fully supported by the context.

Context:
{context}

Answer:
{answer}

Respond with only YES or NO.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return "YES" in response.choices[0].message.content.upper()


# ------------------------------------------------
# Retrieval Recall@k
# ------------------------------------------------

def evaluate_retrieval() -> float:

    hits = 0

    for item in evaluation_dataset:

        question = item["question"]
        expected_answer = item["answer"].lower()

        results = retriever.retrieve(question)

        if results.empty:
            continue

        context = " ".join(results["contents"].tolist()).lower()

        # Use fuzzy matching for retrieval check too
        if is_correct(expected_answer, context):
            hits += 1

    recall = hits / len(evaluation_dataset)

    print("\nRetrieval Evaluation")
    print("---------------------")
    print(f"Recall@k: {recall:.2f}")

    return recall


# ------------------------------------------------
# Generation Evaluation
# ------------------------------------------------

def evaluate_generation():

    correct = 0
    faithful = 0

    for item in evaluation_dataset:

        question = item["question"]
        expected = item["answer"]

        results = retriever.retrieve(question)

        if results.empty:
            continue

        context = "\n\n".join(results["contents"].tolist())

        answer = generate_answer(question, context)

        if is_correct(expected, answer):
            correct += 1

        if check_faithfulness(context, answer):
            faithful += 1

        print(f"\nQuestion: {question}")
        print(f"Expected: {expected}")
        print(f"Answer:   {answer}")
        print(f"Correct:  {is_correct(expected, answer)}")

    accuracy = correct / len(evaluation_dataset)
    faithfulness = faithful / len(evaluation_dataset)

    print("\nGeneration Evaluation")
    print("---------------------")
    print(f"Answer Accuracy: {accuracy:.2f}")
    print(f"Faithfulness: {faithfulness:.2f}")

    return accuracy, faithfulness


# ------------------------------------------------
# Full Evaluation
# ------------------------------------------------

def run_full_evaluation():

    print("\n==============================")
    print("Running RAG Evaluation")
    print("==============================")

    recall = evaluate_retrieval()

    accuracy, faithfulness = evaluate_generation()

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")
    print(f"Retrieval Recall@k: {recall:.2f}")
    print(f"Answer Accuracy:    {accuracy:.2f}")
    print(f"Faithfulness:       {faithfulness:.2f}")


# ------------------------------------------------
# Run
# ------------------------------------------------

if __name__ == "__main__":
    run_full_evaluation()