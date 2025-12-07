#!/usr/bin/env python3
import json
import sys

def convert_to_jsonl(input_file, output_file):
    """
    Convert multi-line JSON objects to single-line JSONL format
    """
    with open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.read()

    # Split by lines and reconstruct JSON objects
    json_objects = []
    current_object = ""
    brace_count = 0

    for line in content.split('\n'):
        current_object += line + '\n'

        # Count braces to determine when we have a complete JSON object
        for char in line:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1

        # When brace count reaches 0, we have a complete JSON object
        if brace_count == 0 and current_object.strip():
            try:
                # Parse and minify the JSON object
                json_obj = json.loads(current_object.strip())
                json_objects.append(json_obj)
                current_object = ""
            except json.JSONDecodeError:
                # If parsing fails, continue accumulating lines
                continue

    # Write as JSONL (one JSON object per line)
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for obj in json_objects:
            outfile.write(json.dumps(obj, separators=(',', ':'), ensure_ascii=False) + '\n')

    print(f"Converted {len(json_objects)} JSON objects to JSONL format")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    input_file = "/Users/mishra/personal/evra/evra-server/fine-tuning/sft-dataset/all_evra_sft_dataset.jsonl"
    output_file = "/Users/mishra/personal/evra/evra-server/fine-tuning/sft-dataset/all_evra_sft_dataset.jsonl"

    # Create backup first
    backup_file = input_file + ".backup"
    with open(input_file, 'r', encoding='utf-8') as infile, open(backup_file, 'w', encoding='utf-8') as backupfile:
        backupfile.write(infile.read())
    print(f"Backup created: {backup_file}")

    convert_to_jsonl(input_file, output_file)