#!/usr/bin/env python3
"""
Pipeline to generate Claude completions for all rows in data/items.csv
Writes results to data/items_with_completions.csv
"""

import csv
import os
from pathlib import Path
import anthropic

def generate_completions(input_csv: str, output_csv: str, model: str = "claude-opus-4-8"):
    """Generate completions for each row in the input CSV."""

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    results = []

    # Read input CSV
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    print(f"Processing {total} items...")

    for idx, row in enumerate(rows, 1):
        item_id = row['id']
        text = row['text']
        category = row['category']

        # Generate completion
        try:
            message = client.messages.create(
                model=model,
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": text
                    }
                ]
            )

            completion = message.content[0].text
            results.append({
                'id': item_id,
                'text': text,
                'category': category,
                'completion': completion
            })

            print(f"[{idx}/{total}] {item_id}: ✓")

        except Exception as e:
            print(f"[{idx}/{total}] {item_id}: ✗ ({str(e)})")
            results.append({
                'id': item_id,
                'text': text,
                'category': category,
                'completion': f"ERROR: {str(e)}"
            })

    # Write results
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'text', 'category', 'completion'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCompleted. Results written to {output_csv}")

if __name__ == "__main__":
    input_file = Path("data/items.csv")
    output_file = Path("data/items_with_completions.csv")

    if not input_file.exists():
        print(f"Error: {input_file} not found")
        exit(1)

    generate_completions(str(input_file), str(output_file))
