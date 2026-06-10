import json
import sys

def main(report_path):
    with open(report_path, 'r') as f:
        data = json.load(f)

    images = data.get("images", [])
    pass_count = 0
    non_pass = []

    for img in images:
        status = img["overall_status"]
        if status == "pass":
            pass_count += 1
        else:
            non_pass.append((img["file_name"], status))

    print(f"pass: {pass_count}")
    print(f"non-pass: {len(non_pass)}")
    for name, status in non_pass:
        print(f"  [{status}] {name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_report.py <judge_report.json>")
        sys.exit(1)
    main(sys.argv[1])
