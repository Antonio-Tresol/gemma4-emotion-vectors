#!/usr/bin/env python3
"""Pipeline to get Claude completions for every row in a CSV file."""

import csv
import os
import sys
from pathlib import Path
from typing import Optional
import json

import anthropic

def load_csv(filepath: str) -> list[dict]:
    """Load CSV file and return list of dictionaries."""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def get_completion(client: anthropic.Anthropic, prompt: str) -> str:
    """Get a single completion from Claude."""
    message = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return message.content[0].text

def process_items(input_csv: str, output_csv: str, api_key: Optional[str] = None) -> None:
    """Process all items in CSV and write completions to output CSV."""
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set and api_key not provided")

    client = anthropic.Anthropic(api_key=api_key)

    # Load input data
    print(f"Loading items from {input_csv}...")
    items = load_csv(input_csv)
    print(f"Found {len(items)} items to process")

    # Process each item
    results = []
    for i, item in enumerate(items, 1):
        item_id = item.get('id', f'item-{i}')
        text = item.get('text', '')
        category = item.get('category', '')

        print(f"[{i}/{len(items)}] Processing {item_id}...", end=" ", flush=True)

        try:
            completion = get_completion(client, text)
            result = {
                'id': item_id,
                'text': text,
                'category': category,
                'completion': completion
            }
            results.append(result)
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
            result = {
                'id': item_id,
                'text': text,
                'category': category,
                'completion': f"ERROR: {str(e)}"
            }
            results.append(result)

    # Write output CSV
    print(f"\nWriting results to {output_csv}...")
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['id', 'text', 'category', 'completion']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✓ Complete! Processed {len(results)} items")

if __name__ == "__main__":
    input_file = "data/items.csv"
    output_file = "data/items_completed.csv"

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    process_items(input_file, output_file)
