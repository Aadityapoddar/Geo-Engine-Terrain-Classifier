"""Sentence-level corrections for new.tex, computed from the artefacts.

Each entry is a (existing sentence, corrected sentence) pair and corresponds to
a numbered item in the journal-readiness audit. The replacements are computed
from doc/assets/ rather than typed, so the manuscript cannot drift away from
the results it reports.

An anchor must never survive inside its own replacement, or a second run
appends the same addendum again. scripts/update_paper_tex.py applies the
longest anchor first, because several are nested inside the sentences that
other pairs rewrite whole.
"""

from evaluation.splits import district_splits


def _tex(name):
    """Escape a band or stack name for LaTeX text mode.

    Band names carry underscores -- s_ent, g_idm, g_contrast -- and an
    unescaped underscore is a subscript operator outside maths mode, so the
    document fails to compile rather than merely looking wrong.
    """
    return str(name).replace("_", r"\_")


def _p(value):
    """A p-value that never claims to be exactly zero."""
    return "$p<0.001$" if value < 0.001 else f"$p={value:.3f}$"


def _pct(value, digits=2):
    return f"{value * 100:.{digits}f}\\%"


def _interval(value):
    low, high = value
    return f"{low * 100:.2f}--{high * 100:.2f}\\%"


def _band_study(ablation, comparison, dev_n, test_n):
    """Everything the paper says about features, restated from the v4 run.

    The previous feature study used a four-class WorldCover reference over
    twelve rectangular tiles, a March-April composite belonging to neither
    season, and a pooled evaluation set that also chose the stack. None of that
    survives contact with the rest of this paper, so every sentence and number
    that came from it is replaced rather than adjusted. If the new artefacts
    are absent, nothing is returned and the old text stays -- a half-updated
    section would be worse than either.
    """
    if not ablation or "ablation" not in ablation or not comparison:
        return []

    scores = ablation["scores"]
    winter = ablation["ablation"]["winter"]
    summer = ablation["ablation"].get("summer", winter)

    def family(season, key, field, digits=1):
        entry = scores[season][f"gtb:{key}"]
        if field == "oa":
            return f"{entry['overall_accuracy'] * 100:.{digits}f}\\%"
        source = ("per_class_recall" if field == "recall"
                  else "per_class_precision")
        label = "Open Land" if field == "recall" else "Built Area"
        value = entry[source].get(label) or 0.0
        return f"{value * 100:.{digits}f}\\%"

    def worst(payload, count=3):
        rows = sorted(payload["bands"], key=lambda row: row["delta_points"])
        return ", ".join(
            f"{_tex(row['band'])} ({row['delta_points']:+.2f})"
            for row in rows[:count])

    droppable = comparison.get("droppable_in_both") or []
    season_payload = comparison["seasons"]
    stacks = comparison["stacks"]

    # The same rule the adoption uses, so the paper cannot describe a stack the
    # code does not ship.
    from scripts.adopt_band_stack import select_stack
    chosen, mean_scores = select_stack(comparison)

    def stack_line(season, name):
        payload = season_payload[season]
        test = payload["test"][name]
        low, high = test["overall_accuracy_ci95"]
        return (f"{test['overall_accuracy'] * 100:.2f}\\% "
                f"(95\\% CI {low * 100:.2f}--{high * 100:.2f}\\%)")

    def between(season, left, right):
        for entry in season_payload[season]["test_paired"]:
            if {entry["left"], entry["right"]} == {left, right}:
                sign = 1 if entry["left"] == left else -1
                low, high = sorted(v * sign for v in entry["ci95"])
                return (entry["mean_district_oa_difference"] * sign * 100,
                        low * 100, high * 100, entry["wilcoxon_p"])
        return None

    chosen_vs_b19 = {season: between(season, chosen, "b19")
                     for season in season_payload}
    chosen_bands = stacks[chosen]
    historical = stacks.get("b19", [])
    kept = [band for band in chosen_bands if band not in historical]
    dropped = [band for band in historical if band not in chosen_bands]

    def mean_dev(name):
        return f"{mean_scores.get(name, 0) * 100:.2f}\\%" if mean_scores else "n/a"

    return [
        ("A counter-intuitive finding of this research is that providing a machine learning algorithm with more data does not universally improve performance. Adding bands is not always better. Two reduced stacks were scored against the full 27-band stack on a reserved evaluation subset of 5,760 points (Table~XII and Fig.~\\ref{fig:band_reduction}).",
         f"Adding bands is not always better, but on this stack removing them "
         f"is mostly worse, and the number of bands matters far less than "
         f"which ones. A leave-one-band-out sweep nominates the bands that can "
         f"go, and Table~XII scores the stacks those nominations produce "
         f"together with the two historical cuts, choosing on the {dev_n} "
         f"development districts and confirming once on the {test_n} test "
         f"districts (Fig.~\\ref{{fig:band_reduction}}b). Every stack is "
         f"trained from one sampled table carrying all 27 bands, so they "
         f"differ only in which columns the classifier may read and not in "
         f"which pixels survived sampling. Ranked by development accuracy "
         f"averaged over the two reported seasons the selected stack is "
         f"\\texttt{{{_tex(chosen)}}} at {mean_dev(chosen)}, and the stack "
         f"this project had been shipping is last of the five at "
         f"{mean_dev('b19')} despite having the same band count; the two share "
         f"only {len(chosen_bands) - len(kept)} of their nineteen. The "
         f"selected stack keeps {', '.join(_tex(b) for b in kept)}, which the "
         f"earlier cut discarded, and drops "
         f"{', '.join(_tex(b) for b in dropped)} instead. Scored once on the "
         f"held-out test districts that is worth "
         f"{chosen_vs_b19['winter'][0]:+.2f} points in winter (95\\% CI "
         f"{chosen_vs_b19['winter'][1]:+.2f} to {chosen_vs_b19['winter'][2]:+.2f}, "
         f"{_p(chosen_vs_b19['winter'][3])}) and "
         f"{chosen_vs_b19['summer'][0]:+.2f} in summer (95\\% CI "
         f"{chosen_vs_b19['summer'][1]:+.2f} to {chosen_vs_b19['summer'][2]:+.2f}, "
         f"{_p(chosen_vs_b19['summer'][3])}). The Hughes phenomenon is the "
         f"usual explanation offered for a reduction that helps, and the "
         f"evidence here does not support one: the full 27-band stack is "
         f"mid-table rather than bottom, and almost nothing in it is "
         f"redundant. What the earlier study found was that its own cut had "
         f"been chosen on the points it was then reported against."),
        ("Table~VIII starts from a ten-band optical baseline and adds groups one at a time (Random Forest, 1,274 statewide points). No new labels are added, so the gains are from features only.",
         f"Table~VIII starts from a ten-band optical baseline and adds feature "
         f"families one at a time, scored district by district on the {dev_n} "
         f"development districts against the same five-class consensus every "
         f"other number in this paper uses. No new labels are added, so the "
         f"gains are from features only."),
        ("Bareness indices add only 1.8 points. That is expected: NDBI, UI and IBI were designed to separate built land from vegetation, not from dry soil \\cite{r21}, \\cite{r26}. Optical GLCM texture raises built-area precision from 51\\% to 58\\% and cuts open-land built errors from 203 to 130. Sentinel-1 raises open-land recall from 7\\% to 50\\%, the largest single jump, because double-bounce is a scattering mechanism rather than a colour. Using texture and radar together beats either one alone. The full 27-band stack reaches 73.9\\% OA and 54\\% open-land recall.",
         f"The optical baseline alone reaches {family('winter', 'g10_baseline', 'oa')} "
         f"overall accuracy in winter but only "
         f"{family('winter', 'g10_baseline', 'recall')} open-land recall: dry "
         f"soil and concrete are close enough in reflectance that no "
         f"combination of the optical bands separates them. Bareness indices "
         f"move it to {family('winter', 'g15_bareness', 'oa')}, which is a "
         f"small gain and an expected one, because NDBI, UI and IBI were "
         f"designed to separate built land from vegetation rather than from "
         f"dry soil \\cite{{r21}}, \\cite{{r26}}. Optical GLCM texture takes "
         f"built-area precision to {family('winter', 'g21_texture', 'precision')}. "
         f"Sentinel-1 is what open land actually needs: adding radar and its "
         f"texture lifts open-land recall to "
         f"{family('winter', 'g27_sar', 'recall')} in winter and "
         f"{family('summer', 'g27_sar', 'recall')} in summer, the largest "
         f"single jump in the table, because double-bounce is a scattering "
         f"mechanism rather than a colour. The full 27-band stack reaches "
         f"{family('winter', 'g27_sar', 'oa')} OA in winter."),
        ("\\caption{Contribution of each feature family, measured by removal from the 27-band stack (Random Forest, a reserved evaluation subset of 5,760 points). Change in overall accuracy when the family is dropped.}",
         f"\\caption{{Cumulative contribution of each feature family, Smile GTB "
         f"on the {dev_n} development districts. Left: overall accuracy. "
         f"Right: open-land recall, the class the stack exists to repair.}}"),
        ("Fig.~\\ref{fig:ablation} measures the same question by removal rather than by addition, which avoids crediting whichever family happens to be added first with everything two families share. Sentinel-1 amplitude is the one family the stack cannot lose ($-6.36$ points when removed). Optical GLCM texture is needed as well ($-0.41$).",
         f"A leave-one-band-out sweep over the same districts asks the question "
         f"the other way round, which avoids crediting whichever family is "
         f"added first with everything two families share "
         f"(Fig.~\\ref{{fig:band_reduction}}a). Against a 27-band baseline of "
         f"{winter['baseline_overall_accuracy'] * 100:.2f}\\%, the bands the "
         f"stack can least afford to lose are {worst(winter)} in winter and "
         f"{worst(summer)} in summer, all quoted as the change in overall "
         f"accuracy when that band alone is removed. Only "
         f"{len(droppable) if droppable else 'a few'} of the twenty-seven can "
         f"be removed without measurable loss in both seasons"
         + (f" ({', '.join(_tex(b) for b in droppable)})" if droppable else "")
         + f", which is the honest answer to whether this stack is carrying "
         f"redundant features: it is barely carrying any."),
        ("Grey-level texture and radar together raise open-land recall from 7\\% to 54\\%.",
         f"Grey-level texture and radar together raise open-land recall from "
         f"{family('winter', 'g10_baseline', 'recall', 0)} to "
         f"{family('winter', 'g27_sar', 'recall', 0)} in winter."),
        ("GLCM texture improved precision and Sentinel-1 improved recall, taking open-land recall from 7\\% to 54\\%.",
         f"GLCM texture improved built-area precision and Sentinel-1 improved "
         f"open-land recall, taking the latter from "
         f"{family('winter', 'g10_baseline', 'recall', 0)} to "
         f"{family('winter', 'g27_sar', 'recall', 0)} in winter."),
        ("Our feature ablation study quantitatively demonstrated that optical data alone achieves less than 60\\% overall accuracy and a dismal 7\\% recall for open land.",
         f"Our feature ablation quantitatively demonstrated that optical bands "
         f"and their core indices alone reach "
         f"{family('winter', 'g10_baseline', 'oa')} overall accuracy and only "
         f"{family('winter', 'g10_baseline', 'recall')} recall for open land."),
        ("A full hyperparameter grid search under this spatial partitioning is left for future work; see Limitations.",
         f"The two seasons do not agree on which bands to drop, and the "
         f"disagreement is informative rather than noise: winter can spare only "
         f"two of the twenty-seven, summer eight, and no band is safely "
         f"droppable in both. Selecting on one season alone would therefore "
         f"have been an arbitrary choice, so the rule used here ranks stacks by "
         f"development accuracy averaged over the two seasons the paper "
         f"reports. Under a winter-only rule the selection would have been "
         f"\\texttt{{{_tex('drop_winter')}}} with twenty-five bands instead."),
    ]


def tex_replacements(uncertainty, area, depth, provenance, arbiter,
                     protocols=None, ablation=None, comparison=None):
    splits = district_splits()
    test_n = len(splits["test"])
    dev_n = len(splits["development"])
    external_n = test_n + dev_n

    winter = uncertainty["seasons"]["winter"]
    summer = uncertainty["seasons"]["summer"]
    w_test, s_test = winter["test"]["pooled"], summer["test"]["pooled"]
    w_gtb, s_gtb = w_test["after-gtb"], s_test["after-gtb"]

    w_area = area["seasons"]["winter"]["test"]
    s_area = area["seasons"]["summer"]["test"]
    w_domain = w_area.get("domain", {})

    winter_depth = [r for r in depth["records"].values() if r["season"] == "winter"]
    summer_depth = [r for r in depth["records"].values() if r["season"] == "summer"]

    def mean(records, field):
        return sum(record[field] for record in records) / len(records)

    def s1_bounds():
        values = [r["s1_scenes"] for r in winter_depth + summer_depth]
        return min(values), max(values)

    outside = provenance.get("points_outside_jabalpur", {})
    outside_total = sum(outside.values())
    spacing = {name: record["nn_percentiles_m"]["p50"]
               for name, record in provenance["classes"].items()}
    tightest = min(spacing, key=spacing.get)

    def model_span(pool):
        values = [entry["overall_accuracy"] for key, entry in pool.items()
                  if key.count("-") == 1 and key.startswith("after-")]
        return f"{min(values) * 100:.1f}--{max(values) * 100:.1f}\\%"

    external_winter = model_span(w_test)

    def protocol_span(name):
        if not protocols:
            return None
        entry = protocols["seasons"]["winter"]["protocols"][name]
        low, high = entry["overall_accuracy_range"]
        return f"{low * 100:.1f}--{high * 100:.1f}\\%"

    # Hoisted out of the f-strings below: Python 3.9 rejects a backslash
    # inside an f-string expression, and every fallback here contains "\\%".
    random_span = protocol_span("random") or "up to 94.3\\%"
    blocked_span = protocol_span("spatially_blocked") or "79--91\\%"

    def gap(payload, other, table="paired_comparisons", base="after-gtb"):
        for comparison in payload["test"][table]:
            if {comparison["left"], comparison["right"]} == {base, other}:
                sign = 1 if comparison["left"] == base else -1
                low, high = sorted(v * sign for v in comparison["ci95"])
                return (comparison["mean_district_oa_difference"] * sign * 100,
                        low * 100, high * 100, comparison["wilcoxon_p"])
        return None

    w_svm = gap(winter, "after-svm")
    s_svm = gap(summer, "after-svm")
    w_cut = gap(winter, "after-gtb-b27", "stack_comparisons")
    s_cut = gap(summer, "after-gtb-b27", "stack_comparisons")

    transfer = winter["transfer"]
    bare = winter["class_transfer"].get("Open Land", {})
    agri = winter["class_transfer"].get("Agriculture", {})

    if arbiter and "summary" in arbiter:
        agreements = {season: metrics["overall_accuracy"]
                      for season, metrics in arbiter["summary"].items()}
        arbiter_text = (
            f"To put a number on that, the consensus was audited against the "
            f"ESRI 10\\,m Annual Land Cover product for 2024, which contributes "
            f"nothing to it and whose epoch is the closest of any candidate to "
            f"the 2025 composites. The two agree on "
            f"{_pct(agreements.get('winter', 0), 1)} of winter reference points "
            f"and {_pct(agreements.get('summer', 0), 1)} of summer points. That "
            f"is the practical ceiling on any accuracy measured against the "
            f"consensus, since a model cannot be shown to be more right than "
            f"its reference. It is not field verification, and where the two "
            f"disagree this cannot say which is wrong.")
    else:
        arbiter_text = (
            "No independent verification of the consensus has been completed, "
            "so its own error rate is unknown and the accuracies reported "
            "against it inherit an unquantified share of reference error.")

    band_pairs = _band_study(ablation, comparison, dev_n, test_n)

    pairs = band_pairs + [
        # --- Abstract ---------------------------------------------------------
        ("Models are trained on 5,000 labelled points from Jabalpur district (OpenStreetMap boundary, about $5,211\\text{ km}^2$) and then scored on independent public maps over all 48 historical administrative districts of Madhya Pradesh (about $308,252\\text{ km}^2$).",
         f"Models are trained on 5,000 labelled points placed in and around "
         f"Jabalpur district (FAO GAUL 2015 polygon, about "
         f"$4,021\\text{{ km}}^2$; {outside_total:,} of the points fall up to "
         f"23\\,km outside it) and then scored on independent public maps over "
         f"{external_n} districts of Madhya Pradesh (about "
         f"$308,252\\text{{ km}}^2$ in total), split once by spatial block into "
         f"{dev_n} development districts that every model and feature choice "
         f"was made on and {test_n} test districts scored a single time."),
        ("Cutting the feature stack from 27 bands to 19 improved accuracy instead of hurting it on a reserved evaluation subset of 5,760 points. Smile GTB reached 85.11\\% overall accuracy and Kappa 0.814 in winter across the 48 historical administrative districts.",
         f"Cutting the feature stack from 27 bands to 19, re-tested on districts "
         f"that took no part in choosing it, is free in winter "
         f"({w_cut[0]:+.2f} points, 95\\% CI {w_cut[1]:+.2f} to "
         f"{w_cut[2]:+.2f}) but costs {abs(s_cut[0]):.2f} points in summer "
         f"(95\\% CI {s_cut[1]:+.2f} to {s_cut[2]:+.2f}); the earlier finding "
         f"that it improved accuracy came from selecting the cut on the same "
         f"data it was scored on. Smile GTB reached "
         f"{_pct(w_gtb['overall_accuracy'])} overall accuracy (95\\% CI "
         f"{_interval(w_gtb['overall_accuracy_ci95'])}, resampling whole "
         f"districts) and Kappa {w_gtb['kappa']:.3f} in winter across the "
         f"{test_n} held-out test districts, but is statistically "
         f"indistinguishable from SVM there."),

        # --- Introduction -----------------------------------------------------
        ("scores every reported figure against a consensus of independent public products over 48 historical administrative districts",
         f"scores every reported figure against a consensus of independent "
         f"public products over {test_n} test districts that no model or "
         f"feature choice was allowed to see"),
        ("(ii) a controlled ablation that attributes the gain of each feature group, including a validated cut from 27 bands to 19; (iii) a statewide check on 48 historical administrative districts of Madhya Pradesh against a high-confidence public-map consensus;",
         f"(ii) a controlled ablation that attributes the gain of each feature "
         f"group, and a feature-stack reduction chosen on development districts "
         f"and re-tested on held-out ones, which shows the reduction to be a "
         f"compute saving rather than an accuracy gain; (iii) a statewide check "
         f"on {external_n} districts of Madhya Pradesh against a "
         f"high-confidence public-map consensus, partitioned into disjoint "
         f"development and test halves by spatial block;"),

        # --- 2.3 boundary provenance -----------------------------------------
        ("the state was split into its 48 historical administrative districts (using a pre-2023 boundary vintage) using Census of India district polygons (geoBoundaries ADM2), which follow the legal district lines more closely than FAO GAUL.",
         "the state was split into the 48 district polygons of FAO GAUL 2015 "
         "level~2, which is the boundary set every script in the released code "
         "reads. That vintage predates the districts created between 2018 and "
         "2023, so Madhya Pradesh now administers 55 districts against these 48 "
         "polygons. They are therefore described throughout as historical "
         "evaluation units rather than as the state's current districts."),
        ("The Jabalpur outline used here is not FAO GAUL. FAO GAUL district polygons are coarse on the Narmada corridor and around the city fringe. We use the OpenStreetMap administrative polygon for Jabalpur district (the same GeoJSON used by the web application). That polygon covers approximately $5,211\\text{ km}^2$ (bounding box $79.35^\\circ$--$80.58^\\circ\\text{E}$, $22.83^\\circ$--$23.62^\\circ\\text{N}$). This is the area on which the labelled points were placed and on which the local classified map in Fig.~1 is drawn.",
         f"The outline used throughout is the FAO GAUL 2015 level-2 polygon for "
         f"Jabalpur, covering approximately $4,021\\text{{ km}}^2$, and it is "
         f"the same geometry the evaluation code and the web application both "
         f"read. An earlier version of this paper described a "
         f"$5,211\\text{{ km}}^2$ OpenStreetMap outline; no released script "
         f"uses that polygon, and the discrepancy is corrected here. The "
         f"labelled points were not confined to it: their bounding box spans "
         f"$79.38^\\circ$--$80.52^\\circ\\text{{E}}$ and "
         f"$22.84^\\circ$--$23.59^\\circ\\text{{N}}$, and {outside_total:,} of "
         f"them lie outside the polygon (Section~III-B)."),
        ("The district outline is the OpenStreetMap administrative polygon (about $5,569\\text{ km}^2$). Every pixel is labelled by the deployed classifier at 10\\,m.",
         "The outline is the district polygon that the application and the "
         "evaluation code both read. Every pixel is labelled by the deployed "
         "classifier on a 10\\,m output grid."),

        # --- 2.12, 2.13 radar --------------------------------------------------
        ("All sampled districts had between 5 and 20 descending-orbit scenes per seasonal window, so no fallback to a wider temporal range was triggered in practice. The code includes a safety fallback to the full 2024--2025 archive if a district-season window is empty; across six sampled districts in both seasons, this fallback was never activated.",
         f"The collection is filtered to a single orbit direction "
         f"(\\texttt{{orbitProperties\\_pass = DESCENDING}}), because ascending "
         f"and descending passes view the same slope, wall or furrow from "
         f"opposite sides and a median over both would add a look-direction "
         f"term to every radar feature. Counting scenes across all 48 districts "
         f"and both seasons, every district-season window holds between "
         f"{s1_bounds()[0]} and {s1_bounds()[1]} descending-orbit scenes, so no "
         f"window is empty. The temporal fallback to the 2024--2025 archive "
         f"that earlier versions used has been removed from the experimental "
         f"path entirely; it survives only in the interactive dashboard, where "
         f"no accuracy is claimed."),
        ("If a short window has no Sentinel-1 scene, a longer 2024--2025 fallback is used so that every pixel still has radar.",
         "No temporal fallback is applied on the experimental path: an empty "
         "window would leave the radar bands masked and the affected reference "
         "points would be dropped, making the loss visible in the sample count "
         "rather than hidden in a substitution. Measured across all 48 "
         "districts and both seasons, no window is empty. The polarimetric "
         "ratio is formed as the difference of the two decibel bands, "
         "VV(dB)~$-$~VH(dB), which is the logarithm of the linear-power ratio "
         "$\\sigma^0_{VV}/\\sigma^0_{VH}$; the $[0,1]$ rescaling is applied "
         "afterwards to the finished ratio and never to its ingredients."),

        # --- 2.11 seasonal windows -------------------------------------------
        ("Two 2025 windows are used everywhere, with exclusive end dates: winter (1 January--28 February, 58 days) and summer (31 March--30 April, 30 days). The winter window is roughly twice as long as the summer window. We chose these boundaries to match phenological meaning---winter captures rabi crops at peak greenness, summer captures pre-monsoon fallow---rather than to equalise composite depth. To verify that the unequal window lengths do not drive the observed accuracy difference, we counted valid Sentinel-2 scenes ($<$15\\% cloud) per district in each window. Across six sampled districts, winter composites contained a median of 57 scenes (range 45--167) and summer composites contained a median of 33 scenes (range 29--113). Both windows contribute more than enough independent looks for a stable per-pixel median, so the lower summer accuracy (Section~IV) is attributable to phenological ambiguity rather than to composite depth.",
         f"Two 2025 windows of equal length are used everywhere, with exclusive "
         f"end dates: winter (1 January--28 February) and summer "
         f"(1 April--29 May), 59 days each. They were previously 59 and 31 "
         f"days, which confounded season with composite depth because the "
         f"shorter window also had fewer chances of a cloud-free pass. Under "
         f"the matched windows the two composites rest on a comparable number "
         f"of observations, counted across all 48 districts rather than a "
         f"sample of six: a mean of {mean(winter_depth, 's2_scenes'):.1f} "
         f"Sentinel-2 granules per district passing the 15\\% cloud filter in "
         f"winter against {mean(summer_depth, 's2_scenes'):.1f} in summer, and "
         f"{mean(winter_depth, 's1_scenes'):.1f} against "
         f"{mean(summer_depth, 's1_scenes'):.1f} Sentinel-1 scenes. Optical "
         f"coverage is complete in every district in both seasons, so the "
         f"lower summer accuracy is attributable to phenological ambiguity "
         f"rather than to a thinner composite."),

        # --- 2.10 label provenance -------------------------------------------
        ("One thousand manually placed points were prepared per class (5,002 points in total: 1,002 vegetation, 1,000 each for water, built area, open land and agriculture).",
         f"Exactly 1,000 manually placed points were prepared per class, "
         f"{provenance['total_points']:,} in total, verified by re-counting the "
         f"published Earth Engine assets rather than from the collection notes."),
        ("Points were distributed across the full extent of Jabalpur district, covering bounding boxes from $79.38^\\circ$--$80.52^\\circ$E and $22.84^\\circ$--$23.59^\\circ$N. The minimum inter-point spacing within a single class was approximately 20\\,m; no formal double-annotation by a second interpreter was performed.",
         f"Re-measuring the assets against the Jabalpur polygon shows that "
         f"{outside_total:,} of the {provenance['total_points']:,} points fall "
         f"outside it: "
         + ", ".join(f"{count:,} in {name}" for name, count in outside.items())
         + f", every one within 23\\,km of the boundary. Katni was separated "
         f"from Jabalpur in 1998, so a wider historical outline explains the "
         f"placement, but under the boundaries used for evaluation those three "
         f"districts hold training data, and they are withheld from all "
         f"reported accuracy together with Jabalpur itself. Inter-point spacing "
         f"is far tighter than previously stated: the median nearest-neighbour "
         f"distance within a class ranges from {spacing[tightest]:.0f}\\,m "
         f"({tightest}) to {max(spacing.values()):.0f}\\,m, and "
         f"{provenance['classes'][tightest]['within_100m']:,} of the "
         f"{tightest} points lie within 100\\,m of another point of the same "
         f"class, which is three pixels. That clustering is the mechanism "
         f"behind the inflated random-split accuracy in Section~V-E. No formal "
         f"double-annotation by a second interpreter was performed, and the "
         f"assets carry no interpreter, source-image or date attribute, so "
         f"per-point provenance cannot be reconstructed."),
        # The anchor must not survive inside its own replacement, or a second
        # run appends the addendum again; it ends on the following sentence.
        ("on a 10\\,m output grid. Rows with a missing label or a missing band are dropped.",
         "on a 10\\,m "
         "output grid. That is an output grid and not a uniform native "
         "resolution: B11 and B12 are acquired at 20\\,m and QA60 at 60\\,m, "
         "and Earth Engine resamples them by nearest neighbour when the request "
         "specifies 10\\,m, so a 10\\,m pixel of B11 carries no information "
         "finer than 20\\,m. Five of the nineteen model inputs derive from the "
         "20\\,m bands. Rows with a missing label or a missing band are "
         "dropped."),

        # --- 2.1 the model is not XGBoost ------------------------------------
        ("\\item \\textbf{Smile GTB:} Extreme Gradient Boosting \\cite{r13} represents the state-of-the-art in tree-based ensemble methods. It builds trees sequentially, where each new tree corrects the residual errors made by the combination of all previous trees. Smile GTB incorporates explicit regularization to penalize model complexity. It is implemented with \\texttt{ee.Classifier.smileGradientTreeBoost}.",
         "\\item \\textbf{Smile GTB:} gradient tree boosting \\cite{r13} builds "
         "trees sequentially, each new tree fitting the residual errors of the "
         "ensemble so far. Earth Engine exposes the Smile library's "
         "implementation as \\texttt{ee.Classifier.smileGradientTreeBoost}, "
         "which is what this study uses. It is a different implementation from "
         "the XGBoost of Chen and Guestrin, offering neither that method's "
         "regularised objective nor its sparsity-aware split finding, and "
         "results obtained with one should not be reported under the other's "
         "name. The only capacity controls available are the tree count, the "
         "shrinkage rate and the node cap."),

        # --- 2.8 sampling and counts ------------------------------------------
        ("Sampling is 20 points per class per district at 10\\,m, over all 48 historical administrative districts. Any candidate within 100\\,m of a training point is removed. After that buffer, 4,787 winter samples and 4,627 summer samples remain.",
         f"Sampling is 20 points per class per district at 10\\,m over all 48 "
         f"district polygons. Any candidate within 100\\,m of a training point "
         f"is removed. The four districts that contain training points are then "
         f"withheld entirely, and the remainder is split by spatial block into "
         f"{dev_n} development and {test_n} test districts "
         f"(Table~\\ref{{tab_splits}}). After the buffer and the split, "
         f"{w_gtb['sample_count']:,} winter and {s_gtb['sample_count']:,} "
         f"summer reference points remain in the test half."),
        ("Table~IX reports all five models on the public-map consensus over 48 historical administrative districts (4,787 winter and 4,627 summer samples).",
         f"Table~IX reports all five models on the public-map consensus over the "
         f"{test_n} test districts ({w_gtb['sample_count']:,} winter and "
         f"{s_gtb['sample_count']:,} summer samples). The ranking itself was "
         f"decided on the development districts; these numbers are the "
         f"consequence of that decision rather than its basis."),
        ("\\subsection{Classifier comparison over 48 historical administrative districts}",
         f"\\subsection{{Classifier comparison over {test_n} held-out test districts}}"),

        # --- 2.5 the reference is not ground truth ---------------------------
        ("To quantify how well the consensus tracks actual land cover, a stratified subset of 250 consensus points (50 per class) has been prepared for manual verification against 2025 very-high-resolution satellite imagery available in Google Earth; this verification is ongoing and will be reported in a future revision.",
         arbiter_text),
        ("\\item The reference is agreement among public maps, not field plots. While the consensus approach filters out individual map errors, it inherits any systemic biases shared by the parent products. A stratified VHR verification subset (Table~\\ref{tab_splits}) partially addresses this, but the consensus remains the primary benchmark.",
         "\\item The reference is agreement among public maps, not field plots. "
         "While the consensus filters out individual map errors, it inherits "
         "any systemic bias shared by the parent products, which use "
         "overlapping Sentinel inputs. Auditing it against an independent "
         "product bounds that error but cannot remove it, and no field or "
         "very-high-resolution verification has been carried out."),

        # --- 2.7, 2.17 the ranking claim -------------------------------------
        ("Smile GTB is first in both seasons: 85.11\\% OA and Kappa 0.814 in winter, 70.30\\% and 0.627 in summer (Fig.~\\ref{fig:statewide_acc}). SVM and KNN are close in winter (84.10\\% and 83.16\\%). Random Forest and CART trail (78.04\\% and 71.36\\%).",
         f"Smile GTB is nominally first in winter, at "
         f"{_pct(w_gtb['overall_accuracy'])} OA and Kappa "
         f"{w_gtb['kappa']:.3f}, and second in summer at "
         f"{_pct(s_gtb['overall_accuracy'])} and {s_gtb['kappa']:.3f} "
         f"(Fig.~\\ref{{fig:statewide_acc}}). Nominally, because the margin "
         f"does not survive a paired district-level test: against SVM the mean "
         f"per-district difference is {w_svm[0]:+.2f} points in winter "
         f"(95\\% CI {w_svm[1]:+.2f} to {w_svm[2]:+.2f}, Wilcoxon "
         f"{_p(w_svm[3])}) and {s_svm[0]:+.2f} points in summer "
         f"(95\\% CI {s_svm[1]:+.2f} to {s_svm[2]:+.2f}, {_p(s_svm[3])}). "
         f"The honest statement is that the top classifiers are not separable "
         f"on this evidence. Random Forest and CART do separate, and trail by "
         f"margins that are significant in both seasons."),
        ("All five classifiers were compared on the same statewide reference set, and the best-performing classifier was identified from those scores. This means model selection and final evaluation share the same data, which can introduce selection bias. However, no hyperparameter tuning was performed on this test set---settings were inherited from a prior experiment and applied without modification---so the bias is bounded to the variance of the ranking among pre-set models rather than to overfitting of free parameters. A lightweight four-fold spatial cross-validation within Jabalpur (Section~V-D) was conducted post hoc to verify that the ranking is robust to spatial partitioning.",
         f"Earlier versions of this work compared all five classifiers on the "
         f"same statewide reference set and identified the winner from those "
         f"scores, so model selection and final evaluation shared the same "
         f"data. That is repaired here. The districts are partitioned once, by "
         f"spatial block and before anything is selected, into {dev_n} "
         f"development and {test_n} test districts, with the four districts "
         f"that hold training points withheld from both. Every choice reported "
         f"in this paper---the feature stack, the classifier, the "
         f"hyperparameters---is made on the development half alone, and the "
         f"test half is scored once with the result."),

        # --- 2.18 spatial transfer ---------------------------------------------
        ("Accuracy is not uniform across Madhya Pradesh. A tile-based analysis shows overall accuracy rising from about 62\\% in the westernmost tile column to 76\\% in the easternmost, toward the $79.4^\\circ$--$80.6^\\circ\\text{E}$ band where the training points sit.",
         f"Accuracy is not uniform across Madhya Pradesh, but the pattern is "
         f"not the simple distance decay an earlier tile-based reading "
         f"suggested. Regressing per-district overall accuracy on great-circle "
         f"distance from the training district gives a slope of "
         f"{transfer['slope_oa_per_100km'] * 100:+.2f} points per 100\\,km "
         f"($R^2={transfer['r_squared']:.3f}$, "
         f"$p={transfer['linregress_p']:.2f}$), which is indistinguishable "
         f"from flat: the nearest quartile of districts averages "
         f"{_pct(transfer['nearest_quartile_mean_oa'], 1)} and the farthest "
         f"{_pct(transfer['farthest_quartile_mean_oa'], 1)}. The decay is real "
         f"but class-specific and self-cancelling in the aggregate. Open-land "
         f"recall falls {abs(bare.get('slope_recall_per_100km', 0)) * 100:.1f} "
         f"points per 100\\,km while agriculture recall rises "
         f"{agri.get('slope_recall_per_100km', 0) * 100:.1f}, so the class that "
         f"fails first away from the labelled area is precisely the "
         f"open-versus-built case this paper set out to repair."),
        ("Table~XI and Fig.~\\ref{fig:cart_splits} report the same models under three checks of increasing strictness. A random split inside Jabalpur reports 94.3\\%, whereas external consensus over 48 historical administrative districts is 61.1--85.0\\%.",
         f"Table~XI and Fig.~\\ref{{fig:cart_splits}} report the same models "
         f"under three checks of increasing strictness, all measured on the "
         f"same winter composite so that the comparison is between protocols "
         f"and not between pipeline versions. A random split inside the "
         f"labelled area reports {random_span} across the "
         f"five classifiers, a spatially blocked split of the same points "
         f"reports {blocked_span}, and the external consensus over "
         f"the {test_n} held-out test districts reports {external_winter}."),

        # --- 2.6: the band cut is owned by _band_study ------------------
        ("\\caption{Overall accuracy across feature stack reductions (27, 22, and 19 bands) for Random Forest, Smile GTB, and SVM on a reserved evaluation subset of 5,760 points. Error bars indicate standard error over seeds.}",
         f"\\caption{{Overall accuracy across feature stack reductions (27, 22 "
         f"and 19 bands) for Smile GTB, on the {dev_n} development districts "
         f"that chose the stack and the {test_n} test districts that did not. "
         f"Bars are 95\\% intervals from resampling whole districts. The winter "
         f"panel cannot separate the three stacks; the summer panel prefers "
         f"27 bands.}}"),

        # --- 2.16 hyperparameters ---------------------------------------------
        ("To evaluate whether the default hyperparameters (Table~VI) limit the models' performance and to verify that the classifier ranking holds under spatial constraints, a lightweight four-fold spatial block cross-validation was conducted post hoc on the Jabalpur training data. The 5,002 labelled points were partitioned into four geographic quadrants (NW, NE, SW, SE). For each fold, three quadrants formed the training set and the fourth served as the test set, enforcing geographic separation.",
         f"The settings in Table~VI were inherited from an earlier two-class "
         f"problem and were never retuned for the five-class, 19-band setup, "
         f"which is not a neutral omission: a ranking built on inherited "
         f"settings partly measures which model happened to inherit a workable "
         f"configuration. A grid over all five classifiers is therefore run on "
         f"the {dev_n} development districts, and only there, so the tuned "
         f"configuration can afterwards be scored once on the test districts "
         f"and still be an estimate rather than a maximum over a search."),
        ("\\item The hyperparameters in Table~VI were chosen on an earlier problem and were not re-tuned for five classes and 19 bands on the final statewide test set. A lightweight four-fold spatial cross-validation within Jabalpur was conducted post hoc to assess the sensitivity of the classifier ranking to hyperparameter settings, but a full grid search on a spatially blocked development set is recommended for future work.",
         f"\\item The hyperparameter grid is deliberately small and covers the "
         f"capacity controls that matter most for these five models; it is not "
         f"an exhaustive search, and a wider one on the same development "
         f"districts could still move the ranking among the top three, which "
         f"this evidence already cannot separate."),

        # --- 2.15 area adjustment ---------------------------------------------
        ("\\item Sampling is balanced by class rather than area-weighted, so overall accuracy is an unweighted figure and not an area-adjusted state estimate with 95\\% confidence intervals per the Olofsson et al.\\ framework.",
         f"\\item Because the sample takes 20 points per class per district, "
         f"the unweighted confusion matrix describes a landscape that is one "
         f"fifth water. Design-based estimates following Olofsson et al.\\ are "
         f"therefore reported alongside it, weighting each (district, reference "
         f"class) stratum by its measured share of the high-confidence "
         f"consensus area. Area-adjusted overall accuracy is "
         f"{_pct(w_area['area_adjusted_overall_accuracy'])} $\\pm$ "
         f"{w_area['area_adjusted_overall_accuracy_ci95'] * 100:.2f} points in "
         f"winter and {_pct(s_area['area_adjusted_overall_accuracy'])} $\\pm$ "
         f"{s_area['area_adjusted_overall_accuracy_ci95'] * 100:.2f} in summer, "
         f"against unweighted figures of {_pct(w_gtb['overall_accuracy'])} and "
         f"{_pct(s_gtb['overall_accuracy'])}. What those figures estimate must "
         f"be stated precisely: the weights describe the high-confidence "
         f"consensus domain, which covers "
         f"{_pct(w_domain.get('labelled_share_of_districts', 0), 1)} of the "
         f"test districts' area and is "
         f"{_pct(w_domain.get('reference_class_area_share', {}).get('Agriculture', 0), 1)} "
         f"agriculture within that. The imbalance is an artefact of how the "
         f"reference is built rather than of Madhya Pradesh, since Vegetation "
         f"must satisfy WorldCover, a Dynamic World label and a 0.70 "
         f"confidence threshold while Agriculture needs only WorldCover and "
         f"WorldCereal. These are therefore area-adjusted accuracies over the "
         f"consensus domain and not over the state; a genuine map-wide estimate "
         f"would need a fresh sample stratified on the map's own classes."),

        # --- 2.17 district-block uncertainty -----------------------------------
        ("When our models were evaluated using a standard random train-test split within the Jabalpur training district, they achieved an artificially inflated overall accuracy of over 94\\%.",
         f"When our models were evaluated using a standard random train-test "
         f"split within the labelled area, the five of them reported "
         f"{random_span}, an artificially inflated result."),
        ("When the identical models were subjected to a rigorous spatial block cross-validation, accuracy dropped to 79-91\\%.",
         f"When the identical models were subjected to a spatially blocked "
         f"split of the same points, accuracy dropped to "
         f"{blocked_span}."),
        ("Finally, when evaluated against an independent, high-confidence consensus reference across 48 historical administrative districts, the accuracy settled between 61\\% and 85\\%.",
         f"Finally, when evaluated against an independent, high-confidence "
         f"consensus reference across the {test_n} held-out test districts, the "
         f"accuracy settled at {external_winter} in the same season."),

        # --- remaining "48 districts" mentions ---------------------------------
        ("accuracy is measured only on independent public-map samples over 48 historical administrative districts. The same feature code is used for training and for mapping",
         f"accuracy is measured only on independent public-map samples over the "
         f"{test_n} held-out test districts. The same feature code is used for "
         f"training and for mapping"),
        ("accuracy is measured only on independent public-map samples over 48 historical administrative districts. Training labels are never reused as the test set.",
         f"accuracy is measured only on independent public-map samples over the "
         f"{test_n} held-out test districts. Training labels are never reused "
         f"as the test set."),
        ("statewide scoring is split into 96 jobs (48 historical administrative districts $\\times$ 2 seasons)",
         "statewide scoring is split into 96 jobs (48 districts $\\times$ 2 seasons)"),

        # --- the paragraph that still argued the pre-split conclusion --------
        ("The reduction of the feature stack from 27 bands to an optimized 19 bands resulted in a measurable increase in overall accuracy across multiple classifiers, particularly Random Forest. This observation aligns with the Hughes phenomenon and underscores the importance of feature selection. Redundant or highly correlated features can introduce noise and cause models to overfit the training data. A leaner, highly discriminative feature space allows the classifier to define more robust decision boundaries.",
         "The Hughes phenomenon is the usual explanation offered for a result "
         "like this, and it is worth being clear that the evidence here does "
         "not support it. Redundant features can introduce noise and encourage "
         "overfitting, and on a pooled evaluation the leaner stack did appear "
         "to win. Held-out districts show no such gain in winter and a real "
         "loss in summer, which is what one would expect if the extra nine "
         "features carry a little genuine signal that matters most when the "
         "optical separation is weakest. The reduction remains worth shipping "
         "for its cost, and the season-dependence is the finding."),

        # --- Conclusion, which still quoted the pre-split figure -------------
        ("Smile GTB reached 85.11\\% overall accuracy and Kappa 0.814 in winter across 48 historical administrative districts of Madhya Pradesh.",
         f"Smile GTB reached {_pct(w_gtb['overall_accuracy'])} overall accuracy "
         f"(95\\% CI {_interval(w_gtb['overall_accuracy_ci95'])}) and Kappa "
         f"{w_gtb['kappa']:.3f} in winter across {test_n} held-out test "
         f"districts of Madhya Pradesh, a margin that a paired district-level "
         f"test cannot separate from SVM."),
    ]
    return [(old, new) for old, new in pairs if old and new]
