import os
import glob
import difflib
import itertools

def extract_model_code(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            if "# ======== MODEL CODE ========" in content:
                return content.split("# ======== MODEL CODE ========")[1].strip()
            else:
                return ""
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""

def calculate_distance(code1, code2):
    # Use SequenceMatcher ratio as a proxy for similarity, then invert for distance
    # ratio() returns a float in [0, 1] measuring the similarity of the sequences.
    # 1.0 means identical, 0.0 means completely different.
    # We want "varies the most", so we want the lowest ratio.
    return difflib.SequenceMatcher(None, code1, code2).ratio()

def main():
    base_dir = "examples/output"
    # Find all llm_analysis_*.py files
    files = glob.glob(os.path.join(base_dir, "**", "llm_analysis_*.py"), recursive=True)
    
    if not files:
        print("No files found.")
        return

    model_codes = {}
    for f in files:
        code = extract_model_code(f)
        if code:
            model_codes[f] = code
        else:
            # print(f"No MODEL CODE found in {f}")
            pass

    file_list = list(model_codes.keys())
    if len(file_list) < 2:
        print("Not enough files with MODEL CODE to compare.")
        return

    pairs = []
    for f1, f2 in itertools.combinations(file_list, 2):
        sim = calculate_distance(model_codes[f1], model_codes[f2])
        pairs.append((sim, f1, f2))

    # Sort by similarity (ascending)
    pairs.sort(key=lambda x: x[0])

    print("Top 5 most varying pairs:")
    for i, (sim, f1, f2) in enumerate(pairs[:5]):
        print(f"Rank {i+1} (Similarity {sim:.4f}):")
        print(f"  File 1: {f1}")
        print(f"  File 2: {f2}")

if __name__ == "__main__":
    main()

