def extract_final_answer(model_output):
    """
    Extract key statistics about how reliance on the majority (Choice_code=1)
    changes with age across cultures from the provided model_output.

    Returns:
      {
        "object": { ... numeric results ... },
        "description": "Plain-language interpretation of the results"
      }
    The function prefers to parse the precomputed marginal-effects text
    stored in model_output['margeff']. If that is unavailable or parsing
    fails, it returns an explanatory message.
    """
    import re

    out = {"object": None, "description": None}

    # Helper to convert string tokens to floats safely
    def to_float(tok):
        try:
            return float(tok)
        except Exception:
            return None

    mfx_text = model_output.get("margeff", None)
    if not mfx_text or not isinstance(mfx_text, str):
        out["description"] = "No marginal-effects text found in model_output['margeff']."
        return out

    # Find the Choice_code=1 block in the marginal effects text
    try:
        # Split on the header for the Choice_code=1 block
        parts = mfx_text.split("\n  Choice_code=1")
        if len(parts) < 2:
            raise ValueError("Choice_code=1 block not found")

        block = parts[1]
        # Trim everything after the next dashed separator (which precedes Choice_code=2)
        block = block.split("\n-----------------------------------------------------------------------------------")[0]

        # Each line with data typically looks like:
        # varname (padded) dy/dx std err z P>|z| [0.025 0.975]
        # We'll parse by taking the first 18 chars as varname area (matches formatting)
        lines = block.strip().splitlines()

        # Remove the header line if present (the column names)
        # Usually the first non-empty line is the header, so detect and skip it
        parsed = {}
        for line in lines:
            line_stripped = line.rstrip()
            if not line_stripped:
                continue
            # skip header line that contains "dy/dx" or "std err"
            if re.search(r"\bdy/dx\b", line_stripped) and re.search(r"\bstd err\b", line_stripped):
                continue
            # Defensive: if the line starts with dashes skip
            if set(line_stripped.strip()) <= set("-"):
                continue

            # Extract varname from fixed-width region if possible
            if len(line_stripped) >= 18:
                varname = line_stripped[:18].strip()
                rest = line_stripped[18:].strip()
            else:
                pieces = line_stripped.split()
                varname = pieces[0]
                rest = " ".join(pieces[1:])

            nums = re.split(r"\s+", rest)
            # Expect at least 6 numeric tokens: dy/dx, std err, z, P>|z|, [0.025, 0.975]
            if len(nums) >= 6:
                dy_dx = to_float(nums[0])
                std_err = to_float(nums[1])
                z = to_float(nums[2])
                p = to_float(nums[3])
                ci_lower = to_float(nums[4])
                ci_upper = to_float(nums[5])
                parsed[varname] = {
                    "dy/dx": dy_dx,
                    "std_err": std_err,
                    "z": z,
                    "p": p,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            else:
                # Skip lines we cannot parse reliably
                continue

        if not parsed:
            raise ValueError("No variables parsed from Choice_code=1 block")

        # Pull out focal statistics
        focal = {}
        # main age terms
        if "age_c" in parsed:
            focal["age_c"] = parsed["age_c"]
        if "age_c2" in parsed:
            focal["age_c2"] = parsed["age_c2"]

        # interactions: any var that starts with 'age_c:culture_'
        interactions = {}
        for name, stats in parsed.items():
            if name.startswith("age_c:culture_"):
                interactions[name] = stats
        focal["age_by_culture_interactions"] = interactions

        # Also include strong controls relevant to interpretation
        for ctrl in ("majority_first", "gender_b"):
            if ctrl in parsed:
                focal[ctrl] = parsed[ctrl]

        # Build a concise interpretation string using extracted numbers
        # Determine significance flags
        def sig_label(p):
            if p is None:
                return "n/a"
            if p < 0.01:
                return "p<0.01"
            if p < 0.05:
                return "p<0.05"
            if p < 0.10:
                return "p<0.10"
            return "ns"

        # Interpret main age terms
        age_interp_parts = []
        if "age_c" in focal:
            a = focal["age_c"]
            age_interp_parts.append(
                f"Linear age marginal effect on choosing the majority: dy/dx={a['dy/dx']:.4f}, "
                f"SE={a['std_err']:.4f}, p={a['p']:.3f} ({sig_label(a['p'])})."
            )
        if "age_c2" in focal:
            a2 = focal["age_c2"]
            age_interp_parts.append(
                f"Quadratic age (age^2) marginal effect: dy/dx={a2['dy/dx']:.4f}, "
                f"SE={a2['std_err']:.4f}, p={a2['p']:.3f} ({sig_label(a2['p'])})."
            )

        # Interpret interactions: list significant or marginal ones first
        inter_lines = []
        for name, stats in sorted(interactions.items()):
            inter_lines.append(
                f"{name}: dy/dx={stats['dy/dx']:.4f}, SE={stats['std_err']:.4f}, "
                f"p={stats['p']:.3f} ({sig_label(stats['p'])})"
            )

        # Controls
        ctrl_lines = []
        for ctrl in ("majority_first", "gender_b"):
            if ctrl in focal:
                s = focal[ctrl]
                ctrl_lines.append(
                    f"{ctrl}: dy/dx={s['dy/dx']:.4f}, SE={s['std_err']:.4f}, p={s['p']:.3f} ({sig_label(s['p'])})"
                )

        description_lines = []
        description_lines.append(
            "Key marginal effects for the probability of selecting the majority (Choice_code=1):"
        )
        description_lines += age_interp_parts
        if inter_lines:
            description_lines.append("Age-by-culture interaction marginal effects (per culture dummy):")
            description_lines += inter_lines
        if ctrl_lines:
            description_lines.append("Important controls (order and gender):")
            description_lines += ctrl_lines

        # Short plain-language summary focusing on the substantive question
        # Use the numbers to make the interpretation:
        plain_summary_parts = []
        # If age quadratic significant -> say non-linear change with age
        if "age_c2" in focal and focal["age_c2"]["p"] is not None and focal["age_c2"]["p"] < 0.05:
            plain_summary_parts.append(
                "There is evidence of a non-linear (quadratic) developmental change in reliance on the majority: "
                "the quadratic term is positive and statistically significant, indicating an accelerating change in the "
                "probability of choosing the majority across ages (holding other variables constant)."
            )
        else:
            plain_summary_parts.append(
                "There is no strong evidence of a simple linear age effect on majority choice (linear age term not statistically significant at p<0.05)."
            )

        # Check culture moderators for significant differences
        sig_interactions = {k: v for k, v in interactions.items() if v.get("p") is not None and v["p"] < 0.05}
        marginal_interactions = {k: v for k, v in interactions.items() if v.get("p") is not None and 0.05 <= v["p"] < 0.10}

        if sig_interactions:
            for k, v in sig_interactions.items():
                plain_summary_parts.append(
                    f"In {k.replace('age_c:culture_','culture_')}, the age slope differs significantly "
                    f"(interaction dy/dx={v['dy/dx']:.4f}, p={v['p']:.3f}), meaning the developmental trajectory "
                    "of majority preference is moderated by that culture."
                )
        if marginal_interactions:
            for k, v in marginal_interactions.items():
                plain_summary_parts.append(
                    f"In {k.replace('age_c:culture_','culture_')}, there is a marginal (p<0.10) moderation of the age effect "
                    f"(interaction dy/dx={v['dy/dx']:.4f}, p={v['p']:.3f})."
                )

        # Note strong order effect if present
        if "majority_first" in focal and focal["majority_first"]["p"] is not None and focal["majority_first"]["p"] < 0.001:
            plain_summary_parts.append(
                "There is a strong demonstration order effect: showing the majority first substantially increases the probability of choosing the majority."
            )

        # Compose final description
        description = "\n".join(description_lines) + "\n\nSummary:\n" + " ".join(plain_summary_parts)

        out["object"] = {
            "focal_marginal_effects": focal,
            "parsed_marginal_effects_text_block": block.strip(),
        }
        out["description"] = description
        return out

    except Exception as e:
        out["description"] = f"Failed to parse marginal-effects text for Choice_code=1: {e}"
        return out