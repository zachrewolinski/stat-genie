import json
from pathlib import Path

results = json.loads(Path('analysis_results.json').read_text())

# Extract values
agg = {row['skin_group']: row for row in results['group_agg']}
poisson = results['poisson']

rate_dark = agg['dark']['rate_per_game']
rate_light = agg['light']['rate_per_game']
rr = results['rate_ratio']['rr']
ci_low = results['rate_ratio']['rr_ci_low']
ci_high = results['rate_ratio']['rr_ci_high']

explanation = (
    "Using the player–referee dyads with skin-tone ratings, I averaged the two rater scores "
    "(0–1 scale) and compared dark (>=0.75) vs light (<=0.25) players. "
    f"Dark-skin dyads had {agg['dark']['red_cards']} red cards over {agg['dark']['games']} games "
    f"(rate {rate_dark:.6f} per game), while light-skin dyads had {agg['light']['red_cards']} red cards "
    f"over {agg['light']['games']} games (rate {rate_light:.6f} per game). "
    "A Poisson regression with log(games) as an offset and robust SEs showed a higher red-card rate "
    f"for dark vs light players (rate ratio {rr:.3f}, 95% CI {ci_low:.3f}–{ci_high:.3f}, p={poisson['p']:.4f}). "
    "This indicates a statistically significant but modest increase in red-card likelihood for darker-skinned players."
)

conclusion = {
    "response": 68,
    "explanation": explanation
}

Path('conclusion.txt').write_text(json.dumps(conclusion))
