
import csv
import random
from pathlib import Path

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Templates for each ticket category and priority.
ticket_templates = {
    "Billing": {
        "High": [
            "I was charged twice for my subscription",
            "My payment was deducted but my order failed",
            "I received an unexpected duplicate invoice",
            "My account was charged the wrong amount",
            "I need urgent help with a failed refund",
        ],
        "Medium": [
            "Please explain the extra charge on my bill",
            "I need a copy of my latest invoice",
            "My monthly subscription price looks incorrect",
            "Can you check my recent payment history",
            "I have a question about my billing statement",
        ],
        "Low": [
            "Where can I download an old invoice",
            "How do I update my billing address",
            "I want to understand the billing cycle",
            "Can you explain the taxes on my invoice",
            "Where can I view my previous receipts",
        ],
    },
    "Technical Support": {
        "High": [
            "The application crashes every time I open it",
            "I cannot access the service and my work is blocked",
            "The website is completely unavailable",
            "The system keeps failing during checkout",
            "I urgently need help because the app is down",
        ],
        "Medium": [
            "The application loads very slowly",
            "I am getting an error when uploading a file",
            "The dashboard is not displaying correctly",
            "Some features stopped working today",
            "The application freezes occasionally",
        ],
        "Low": [
            "How do I change my display settings",
            "Where can I find the application tutorial",
            "How do I enable desktop notifications",
            "Can I customize the dashboard layout",
            "I need help understanding the application settings",
        ],
    },
    "Account Access": {
        "High": [
            "My account has been locked and I cannot log in",
            "Someone changed my password without permission",
            "I suspect unauthorized access to my account",
            "I cannot access my account before an important deadline",
            "My account recovery process keeps failing",
        ],
        "Medium": [
            "I am not receiving the password reset email",
            "The login page rejects my correct password",
            "My verification code is not arriving",
            "I cannot update my account email address",
            "Two factor authentication is not working",
        ],
        "Low": [
            "How do I change my profile picture",
            "I want to update my account preferences",
            "Where can I edit my contact information",
            "How do I change my username",
            "Can I update my notification preferences",
        ],
    },
    "Delivery": {
        "High": [
            "My urgent order has not arrived and is overdue",
            "The tracking page says delivered but I received nothing",
            "My package is missing and I need it immediately",
            "My delivery is delayed and the item is essential",
            "The courier lost my package",
        ],
        "Medium": [
            "My order tracking has not updated",
            "Can you check the expected delivery date",
            "My package has been delayed by two days",
            "I received an update that my shipment is delayed",
            "Please help me locate my current shipment",
        ],
        "Low": [
            "How do I change my delivery instructions",
            "Can I track an order after it ships",
            "Where can I find the shipping policy",
            "Can I update my preferred delivery time",
            "How long does standard delivery usually take",
        ],
    },
    "Product Inquiry": {
        "High": [
            "I need product information before an urgent purchase",
            "Please confirm whether this product is safe to use",
            "I need immediate clarification about product compatibility",
            "A product specification appears to be incorrect",
            "Please clarify an important product limitation",
        ],
        "Medium": [
            "Does this product support my device",
            "Can you explain the available product features",
            "What is the difference between these two plans",
            "Is this item compatible with my current setup",
            "Please provide more details about this product",
        ],
        "Low": [
            "Where can I read the product brochure",
            "Do you have a product comparison guide",
            "What colors are available for this item",
            "Where can I find product dimensions",
            "Can I view the general product specifications",
        ],
    },
    "Cancellation": {
        "High": [
            "Please stop the renewal that is charging me today",
            "I need an urgent cancellation before another payment",
            "Cancel my order immediately because it is incorrect",
            "I was charged after requesting cancellation",
            "My cancellation request failed and another charge is pending",
        ],
        "Medium": [
            "I would like to cancel my subscription",
            "Please help me cancel an order before dispatch",
            "How can I end my current membership",
            "I submitted a cancellation request but got no update",
            "Can you confirm my cancellation status",
        ],
        "Low": [
            "Where can I read the cancellation policy",
            "How long does cancellation normally take",
            "Can I cancel a future renewal",
            "I want to understand the subscription cancellation terms",
            "Where can I find cancellation instructions",
        ],
    },
}

# Build 300 records: 6 categories x 3 priorities x 16-17 tickets.
rows = []
ticket_id = 1

for category, priority_groups in ticket_templates.items():
    for priority, templates in priority_groups.items():
        for i in range(16):
            text = templates[i % len(templates)]

            # Vary wording slightly to create additional examples.
            variations = [
                text,
                "Hello, " + text.lower(),
                text + ". Please assist me.",
                "I need help: " + text.lower(),
                "Customer issue: " + text.lower(),
            ]

            rows.append({
                "ticket_id": f"T{ticket_id:03d}",
                "text": variations[i % len(variations)],
                "category": category,
                "priority": priority,
            })
            ticket_id += 1

# 6 categories x 3 priorities x 16 tickets = 288.
# Add 12 more examples to reach exactly 300.
extra_examples = [
    ("Billing", "High", "I have been charged twice and need a refund"),
    ("Billing", "Medium", "Please check the amount on my latest bill"),
    ("Technical Support", "High", "The website is down and I cannot work"),
    ("Technical Support", "Low", "How can I change my application theme"),
    ("Account Access", "High", "My account is locked after suspicious activity"),
    ("Account Access", "Medium", "I am unable to receive my login code"),
    ("Delivery", "High", "My package is missing and the deadline is today"),
    ("Delivery", "Low", "Where can I read the delivery policy"),
    ("Product Inquiry", "Medium", "Can this product work with my device"),
    ("Product Inquiry", "Low", "Where can I find the product guide"),
    ("Cancellation", "High", "Cancel my subscription before the next charge"),
    ("Cancellation", "Medium", "Can you check my cancellation request"),
]

for category, priority, text in extra_examples:
    rows.append({
        "ticket_id": f"T{ticket_id:03d}",
        "text": text,
        "category": category,
        "priority": priority,
    })
    ticket_id += 1

random.shuffle(rows)

# Stratified splitting by category AND priority.
# This ensures every category-priority combination is represented.
train, validation, test = [], [], []
groups = {}

for row in rows:
    key = (row["category"], row["priority"])
    groups.setdefault(key, []).append(row)

for group in groups.values():
    random.shuffle(group)
    n = len(group)
    n_train = round(n * 0.60)
    n_validation = round(n * 0.20)

    train.extend(group[:n_train])
    validation.extend(group[n_train:n_train + n_validation])
    test.extend(group[n_train + n_validation:])

for name, records in [
    ("train.csv", train),
    ("validation.csv", validation),
    ("test.csv", test),
]:
    path = DATA_DIR / name

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["ticket_id", "text", "category", "priority"],
        )
        writer.writeheader()
        writer.writerows(records)

print("Dataset created successfully!")
print("Total tickets:", len(rows))
print("Training tickets:", len(train))
print("Validation tickets:", len(validation))
print("Test tickets:", len(test))
print("\nCategory counts:")

for category in sorted(ticket_templates):
    count = sum(r["category"] == category for r in rows)
    print(f"{category}: {count}")

print("\nFiles saved in:", DATA_DIR)
