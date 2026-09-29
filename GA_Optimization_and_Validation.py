"""Reproduce the 2023 WEH genetic optimization and reviewer diagnostics.

Python 3.13.2; requires numpy, scipy, and matplotlib.
Run: python code/GA_Optimization_and_Validation.py
Inputs are the two English CSV files in data/.
All outputs are written to results/.

The cost values are scenario cost-equivalent coefficients, not observed spending.
This self-contained script retains the final internal-reallocation model and
historical overall adjustment-intensity limit from the July 2026 GA code.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data"
OUTPUT = ROOT / "results"
RAW_FILE = INPUT / "city_indicator_panel_2014_2023.csv"
WEIGHT_FILE = INPUT / "indicator_metadata.csv"
CITIES = ('Hangzhou', 'Shaoxing', 'Ningbo', 'Jiaxing', 'Zhoushan', 'Huzhou')
NAMES = ('Wastewater discharge (10,000 t)', 'COD discharge (t)', 'Total nitrogen discharge (t)', 'Total phosphorus discharge (t)', 'Wetland area (1,000 ha)', 'Industrial wastewater treatment capacity (10,000 t/day)', 'Ecological water replenishment (100 million m3)', 'Newly treated soil erosion area (1,000 ha)')

COST = np.array((0.00013, 0.000033, 0.000593, 0.001673,
                 1.4411568659448155, 0.04950967629665652,
                 0.68, 0.009629110462882954), dtype=float)
UPPER = np.array((0.20, 0.25, 0.25, 0.25, 0.10, 0.20, 0.30, 0.30))
V0 = 0.3818712777069848
SEEDS = tuple(range(1, 11))
POPULATION = 420
GENERATIONS = 750
ELITES = 36
CROSSOVER = 0.90
MUTATION = 0.10
TOL = 1e-8


def write_csv(name, header, rows):
    with (OUTPUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Model:
    indicators: list
    baseline: np.ndarray
    weights: np.ndarray
    minima: np.ndarray
    maxima: np.ndarray
    positive: np.ndarray
    groups: np.ndarray
    controlled: np.ndarray
    baseline_components: dict

    @classmethod
    def read(cls):
        metadata = list(csv.DictReader((INPUT / "indicator_metadata.csv").open(encoding="utf-8", newline="")))
        names = [row["indicator"] for row in metadata]
        groups = np.array([{"Pressure": 0, "State": 1, "Response": 2}[row["psr_group"]]
                           for row in metadata])
        positive = np.array([row["beneficial_direction"] == "True" for row in metadata])
        weights = np.array([float(row["weight"]) for row in metadata])
        data = np.empty((10, 6, 15), dtype=float)
        seen = np.zeros((10, 6, 15), dtype=bool)
        city_index = {name: i for i, name in enumerate(CITIES)}
        with (INPUT / "city_indicator_panel_2014_2023.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                y = 2023 - int(row["year"])
                i = city_index[row["city"]]
                j = int(row["indicator_id"])
                data[y, i, j] = float(row["value"])
                seen[y, i, j] = True
        if not seen.all():
            raise ValueError("The city indicator panel is incomplete")
        controlled = np.array([names.index(n) for n in NAMES], dtype=int)
        m = cls(names, data[0], weights, data.min(axis=(0, 1)),
                data.max(axis=(0, 1)), positive, groups, controlled, {})
        m.baseline_components = m.components(data[0])
        return m

    def components(self, raw):
        scores = np.empty_like(raw, dtype=float)
        for j in range(15):
            gap = self.maxima[j] - self.minima[j]
            if gap < 1e-12:
                scores[..., j] = 1.0
            elif self.positive[j]:
                scores[..., j] = (raw[..., j] - self.minima[j]) / gap
            else:
                scores[..., j] = (self.maxima[j] - raw[..., j]) / gap
        weighted = np.clip(scores, 0, 1) * self.weights
        contributions = [
            weighted[..., self.groups == group].sum(axis=-1)
            for group in range(3)
        ]
        group_weights = [
            self.weights[self.groups == group].sum()
            for group in range(3)
        ]
        p, s, r = [
            value / group_weight
            for value, group_weight in zip(contributions, group_weights)
        ]
        h = contributions[0] + contributions[1] + contributions[2]
        mean_psr = (p + s + r) / 3
        product = p * s * r
        c = np.zeros_like(mean_psr)
        valid = (mean_psr > 0) & (product > 0)
        c[valid] = (product[valid] / mean_psr[valid] ** 3) ** (1 / 3)
        return {"P": p, "S": s, "R": r, "H": h,
                "D": np.sqrt(np.maximum(c * h, 0))}

    def evaluate(self, flat, cost_factor=1.0, v=V0):
        x = np.asarray(flat, dtype=float).reshape((-1, 6, 8))
        raw = np.broadcast_to(self.baseline, (len(x), 6, 15)).copy()
        signs = np.array((-1, -1, -1, -1, 1, 1, 1, 1))
        raw[..., self.controlled] = self.baseline[:, self.controlled][None] * (1 + signs * x)
        comp = self.components(raw)
        base = self.baseline_components
        z = (comp["H"] - base["H"][None]).mean(axis=1)
        base_cost = self.baseline[:, self.controlled] * COST[None] * cost_factor
        cost_slack = -np.sum(base_cost[None] * signs * x, axis=(1, 2))
        intensity = np.mean((x / UPPER[None, None]) ** 2, axis=(1, 2))
        slacks = np.column_stack((cost_slack, v - intensity,
            *(comp[k] - base[k][None] for k in ("P", "S", "R", "D"))))
        # The GA uses the same normalized aggregate violation as the source code.
        norm = np.column_stack((np.maximum(0, -cost_slack) / max(base_cost.sum(), 1e-9),
            np.maximum(0, intensity - v) / v,
            *(np.maximum(0, base[k][None] - comp[k]) /
              np.maximum(base[k][None], 1e-9) for k in ("P", "S", "R", "D"))))
        violation = norm.sum(axis=1)
        return z, violation, slacks, comp, intensity

    def repair(self, flat, cost_factor=1.0, v=V0):
        x = np.asarray(flat, dtype=float).reshape(6, 8).copy()
        base_cost = self.baseline[:, self.controlled] * COST[None] * cost_factor
        released = np.sum(base_cost[:, :4] * x[:, :4])
        needed = np.sum(base_cost[:, 4:] * x[:, 4:])
        if needed > released + 1e-12:
            x[:, 4:] *= released / needed if released > 0 else 0.0
        intensity = np.mean((x / UPPER[None]) ** 2)
        if intensity > v:
            x *= np.sqrt(v / intensity)
        return x.ravel()


class CostCapModel(Model):
    """Hold cost coefficients and B0 fixed while changing only the cost cap."""

    def __init__(self, base: Model, delta_fraction: float):
        self.__dict__.update(base.__dict__)
        self.delta_fraction = delta_fraction
        self.base_cost = self.baseline[:, self.controlled] * COST[None]
        self.b0 = float(self.base_cost.sum())
        self.delta_abs = delta_fraction * self.b0
        self.signs = np.array((-1, -1, -1, -1, 1, 1, 1, 1))
        self.anchor = np.zeros((6, 8))
        if delta_fraction < 0:
            self.anchor[:, :4] = .8 * UPPER[:4]
            anchor_net = self.net_change(self.anchor)
            if anchor_net >= self.delta_abs:
                raise ValueError("The cost-saving anchor is not feasible for this cap")
            if super().evaluate([self.anchor.ravel()])[4][0] > V0:
                raise ValueError("The cost-saving anchor violates the V limit")

    def net_change(self, x):
        return float(np.sum(self.base_cost * self.signs * np.asarray(x).reshape(6, 8)))

    def evaluate(self, flat, cost_factor=1.0, v=V0):
        if not np.isscalar(cost_factor) or cost_factor != 1.0:
            raise ValueError("Cost-cap sensitivity holds all coefficients fixed")
        z, violation, slacks, comp, intensity = super().evaluate(flat, 1.0, v)
        old_budget_violation = np.maximum(0.0, -slacks[:, 0]) / self.b0
        slacks[:, 0] += self.delta_abs
        new_budget_violation = np.maximum(0.0, -slacks[:, 0]) / self.b0
        violation = np.maximum(0.0, violation - old_budget_violation + new_budget_violation)
        return z, violation, slacks, comp, intensity

    def repair(self, flat, cost_factor=1.0, v=V0):
        x = np.clip(np.asarray(flat, dtype=float).reshape(6, 8), 0, UPPER[None])
        intensity = np.mean((x / UPPER[None]) ** 2)
        if intensity > v:
            x = x * np.sqrt(v / intensity)
        net = self.net_change(x)
        if net > self.delta_abs:
            anchor_net = self.net_change(self.anchor)
            fraction = (net - self.delta_abs) / (net - anchor_net)
            x = (1 - fraction) * x + fraction * self.anchor
        return x.ravel()


def run_ga(model, seed, cost_factor=1.0, v=V0, generations=GENERATIONS):
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    upper = np.tile(UPPER, 6)
    pop = [model.repair(np.zeros(48), cost_factor, v),
           model.repair(upper, cost_factor, v)]
    pop += [model.repair(nrng.beta(1.6, 1.7, 48) * upper, cost_factor, v)
            for _ in range(POPULATION - 2)]
    history = []
    evaluated = infeasible = attempted = attempted_infeasible = 0
    champion, champion_z = None, -np.inf
    for generation in range(generations + 1):
        z, violations, _, _, _ = model.evaluate(pop, cost_factor, v)
        feasible = violations <= TOL
        evaluated += len(pop)
        infeasible += int((~feasible).sum())
        if feasible.any():
            idx = int(np.argmax(np.where(feasible, z, -np.inf)))
            if z[idx] > champion_z:
                champion, champion_z = pop[idx].copy(), float(z[idx])
        history.append((generation, champion_z if champion is not None else math.nan,
                        float(feasible.mean())))
        if generation == generations:
            break
        order = sorted(range(len(pop)), key=lambda i: (not feasible[i],
                       -z[i] if feasible[i] else violations[i]))
        evaluations = list(zip(z, violations))
        next_pop = [pop[i].copy() for i in order[:ELITES]]
        raw_children = []
        def tournament():
            ids = rng.sample(range(len(pop)), 3)
            return pop[min(ids, key=lambda i: (not feasible[i],
                   -evaluations[i][0] if feasible[i] else evaluations[i][1]))].copy()
        while len(next_pop) < POPULATION:
            a, b = tournament(), tournament()
            if rng.random() < CROSSOVER:
                alpha = nrng.uniform(-0.15, 1.15, 48)
                children = (alpha * a + (1 - alpha) * b,
                            alpha * b + (1 - alpha) * a)
            else:
                children = (a, b)
            for child in children:
                mask = nrng.random(48) < MUTATION
                child[mask] += nrng.normal(0, .07, mask.sum()) * upper[mask]
                reset = nrng.random(48) < .01
                child[reset] = nrng.random(reset.sum()) * upper[reset]
                child = np.clip(child, 0, upper)
                raw_children.append(child)
                next_pop.append(model.repair(child, cost_factor, v))
                if len(next_pop) == POPULATION:
                    break
        attempted += len(raw_children)
        attempted_infeasible += int((model.evaluate(raw_children, cost_factor, v)[1] > TOL).sum())
        pop = next_pop
    if champion is None:
        raise RuntimeError(f"No feasible solution for seed {seed}")
    return champion, history, dict(seed=seed, best_z=champion_z,
        evaluated=evaluated, infeasible=infeasible,
        violation_rate=infeasible / evaluated, attempted=attempted,
        attempted_infeasible=attempted_infeasible,
        raw_violation_rate=attempted_infeasible / attempted)


def slsqp_check(model, x0, cost_factor=1.0, v=V0):
    cache = {"x": None}
    def get(x):
        if cache["x"] is None or not np.array_equal(cache["x"], x):
            z, _, slacks, _, _ = model.evaluate([x], cost_factor, v)
            cache.update(x=x.copy(), z=float(z[0]), slacks=slacks[0])
        return cache
    result = minimize(lambda x: -get(x)["z"], x0, method="SLSQP",
        bounds=[(0.0, float(u)) for u in np.tile(UPPER, 6)],
        constraints=[{"type": "ineq", "fun": lambda x: get(x)["slacks"]}],
        options={"maxiter": 1500, "ftol": 1e-11, "disp": False})
    z, violation, slacks, _, _ = model.evaluate([result.x], cost_factor, v)
    return result, float(z[0]), float(violation[0]), float(slacks[0].min())


def city_metrics(model, flat, cost_factor=1.0, v=V0):
    z, viol, slack, comp, intensity = model.evaluate([flat], cost_factor, v)
    x = flat.reshape(6, 8)
    base = model.baseline_components
    rows = []
    for i, city in enumerate(CITIES):
        rows.append((city, float(base["H"][i]), float(comp["H"][0, i]),
                     float(comp["H"][0, i] - base["H"][i]),
                     float(base["D"][i]), float(comp["D"][0, i]),
                     float(comp["D"][0, i] - base["D"][i]),
                     float(x[i, :4].mean()), float(x[i, 4:].mean())))
    return rows, float(z[0]), float(viol[0]), float(intensity[0])


def measure_contributions(model, flat):
    """Mean regional H gain attributable to each adjustable indicator."""
    x = flat.reshape(6, 8)
    out = []
    for k, j in enumerate(model.controlled):
        sign = -1 if k < 4 else 1
        before = model.baseline[:, j]
        after = before * (1 + sign * x[:, k])
        gap = model.maxima[j] - model.minima[j]
        if gap < 1e-12:
            gain = 0.0
        else:
            if model.positive[j]:
                b = (before - model.minima[j]) / gap
                a = (after - model.minima[j]) / gap
            else:
                b = (model.maxima[j] - before) / gap
                a = (model.maxima[j] - after) / gap
            gain = float((np.clip(a, 0, 1) - np.clip(b, 0, 1)).mean() * model.weights[j])
        out.append(gain)
    return out


def plot_convergence(all_history):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8.4, 5.3), dpi=180)
    for seed, history in all_history.items():
        ax.plot([r[0] for r in history], [r[1] for r in history],
                lw=1.0, alpha=.58, label=f"Seed {seed}")
    ax.set(xlabel="Generation", ylabel="Best feasible objective Z (mean WEH gain)",
           title="GA convergence across 10 independent runs")
    ax.grid(alpha=.25)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "Fig_S1_Convergence.png")
    plt.close(fig)


def plot_sensitivity(records):
    vv = [r["v_limit"] for r in records]
    h = [r["mean_h_after"] for r in records]
    d = [r["mean_d_after"] for r in records]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.7), dpi=180)
    for a, values in zip(ax, (h, d)):
        a.plot(vv, values, color="#496c89", marker="o", lw=1.8, ms=6)
        a.scatter([V0], [values[2]], color="#d08345", s=65, zorder=3,
                  label="Base (V=0.381871)")
        span = max(values) - min(values)
        pad = max(span * .22, 0.00008)
        a.set_ylim(min(values) - pad, max(values) + pad)
        a.set_xticks(vv, ["0.25", "0.30", "0.3818", "0.45", "0.55"])
        a.set_xlabel("Adjustment-intensity limit V")
    ax[0].set(ylabel="Mean WEH index H", title="WEH after optimization")
    ax[1].set(ylabel="Mean coupling coordination D", title="CCD after optimization")
    for a in ax:
        a.grid(axis="y", alpha=.22)
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "Fig_S2_Sensitivity.png")
    plt.close(fig)


def coefficient_sweep():
    """Perturb each cost-equivalent coefficient separately and locally reoptimize."""
    model = Model.read()
    rows = list(csv.DictReader((OUTPUT / "Solver_Decisions.csv").open(encoding="utf-8-sig")))
    # Use the main GA solution as the reference for coefficient sensitivity.
    baseline_x = np.array([float(row["GA_uij"]) for row in rows])
    baseline_z = float(model.evaluate([baseline_x])[0][0])
    result_rows, decision_rows, endpoints = [], [], []
    short_names = ("Wastewater", "COD", "Total nitrogen", "Total phosphorus",
                   "Wetland", "Industrial capacity", "Water replenishment", "Soil conservation")
    for k, name in enumerate(NAMES):
        pair = []
        for perturbation in (-.1, .1):
            factor = np.ones(8)
            factor[k] += perturbation
            start = model.repair(baseline_x, factor)
            local, lz, lv, lslack = slsqp_check(model, start, factor)
            candidates = [(start, False, float(model.evaluate([start], factor)[0][0]),
                           float(model.evaluate([start], factor)[1][0]), 0, float("nan"))]
            candidates.append((local.x, bool(local.success), lz, lv, local.nit, lslack))
            if not local.success or lv > TOL:
                ga_x, _, _ = run_ga(model, 31, cost_factor=factor)
                ga_z, ga_v, _, _, _ = model.evaluate([ga_x], factor)
                candidates.append((ga_x, False, float(ga_z[0]), float(ga_v[0]), GENERATIONS, float("nan")))
                alt, az, av, aslack = slsqp_check(model, ga_x, factor)
                candidates.append((alt.x, bool(alt.success), az, av, alt.nit, aslack))
            valid = [candidate for candidate in candidates if candidate[3] <= TOL]
            highest = max(candidate[2] for candidate in valid)
            near_best = [candidate for candidate in valid if candidate[2] >= highest - 1e-10]
            solution, solver_success, _, _, solver_iterations, solver_slack = max(
                near_best, key=lambda c: (c[1], c[2]))
            _, z, violation, intensity = city_metrics(model, solution, factor)
            _, _, _, comp, _ = model.evaluate([solution], factor)
            contributions = measure_contributions(model, solution)
            city_data, _, _, _ = city_metrics(model, solution, factor)
            cities = "; ".join(row[0] for row in sorted(city_data, key=lambda r: r[3], reverse=True)[:2])
            measures = "; ".join(NAMES[j] for j in sorted(range(8), key=lambda j: contributions[j], reverse=True)[:3])
            label = f"{name} {'-10%' if perturbation < 0 else '+10%'}"
            result_rows.append((name, perturbation, z, z-baseline_z,
                float(comp["H"].mean()), float(comp["D"].mean()), intensity,
                violation, violation <= TOL, solver_slack, solver_success,
                solver_iterations, cities, measures))
            for i, city in enumerate(CITIES):
                for j, indicator in enumerate(NAMES):
                    decision_rows.append((label, city, indicator, float(solution.reshape(6,8)[i,j])))
            pair.append(z)
            print(f"Cost {k+1}/8 {perturbation:+.0%}: Z={z:.10f}", flush=True)
        endpoints.append(pair)
    write_csv("Table_S5_Individual_Cost_Coefficient_Sensitivity.csv",
        ("perturbed_coefficient", "relative_change", "best_z", "delta_z_from_base",
         "mean_h_after", "mean_d_after", "intensity", "normalized_violation",
         "feasible_tol_1e-8", "minimum_raw_constraint_slack", "slsqp_success",
         "slsqp_iterations", "top_two_cities", "top_three_measures"), result_rows)
    write_csv("Individual_Cost_Sensitivity_Decisions.csv",
              ("scenario", "city", "indicator", "uij"), decision_rows)
    fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=180)
    yy = np.arange(8)
    ends = np.array(endpoints)
    ax.hlines(yy, ends.min(axis=1), ends.max(axis=1), color="#5d829f", lw=4)
    ax.scatter(ends[:, 0], yy, marker="<", color="#377da6", label="Coefficient -10%")
    ax.scatter(ends[:, 1], yy, marker=">", color="#cc8044", label="Coefficient +10%")
    ax.axvline(baseline_z, color="#555555", ls="--", lw=1, label="Base")
    ax.set_yticks(yy, short_names)
    ax.invert_yaxis()
    ax.set(xlabel="Locally optimized mean WEH gain Z", title="Individual cost-equivalent coefficient sensitivity")
    ax.grid(axis="x", alpha=.2)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "Fig_S3_Individual_Cost_Sensitivity.png")
    plt.close(fig)

def cost_cap_sweep():
    """Vary the cost-equivalent cap with fixed coefficients and fixed B0."""
    base = Model.read()
    decision_rows = list(csv.DictReader((OUTPUT / "Solver_Decisions.csv").open(encoding="utf-8-sig")))
    # Use the main GA solution as the reference for cost-cap sensitivity.
    baseline_x = np.array([float(row["GA_uij"]) for row in decision_rows])
    deltas = (-.01, -.005, 0.0, .005, .01)
    summary, ga_runs, city_rows, u_rows = [], [], [], []
    for delta in deltas:
        model = CostCapModel(base, delta)
        label = f"{delta:+.1%}" if delta else "Base"
        candidates = []
        if delta == 0:
            candidates.append((baseline_x, "Baseline GA", True))
        else:
            for seed in range(1, 6):
                x, _, run = run_ga(model, seed)
                candidates.append((x, f"GA seed {seed}", True))
                ga_runs.append((label, delta, seed, run["best_z"],
                                run["violation_rate"], run["raw_violation_rate"]))
                print(f"Cost cap {label}, seed {seed}: Z={run['best_z']:.10f}", flush=True)
            best_ga = max(candidates, key=lambda item: model.evaluate([item[0]])[0][0])[0]
            for tag, start in (("SLSQP from GA", best_ga),
                               ("SLSQP from baseline", model.repair(baseline_x))):
                result, z, violation, _ = slsqp_check(model, start)
                if violation <= TOL:
                    candidates.append((result.x, tag, bool(result.success)))
        feasible_candidates = [item for item in candidates if model.evaluate([item[0]])[1][0] <= TOL]
        if not feasible_candidates:
            raise RuntimeError(f"No feasible cost-cap solution for {label}")
        highest = max(float(model.evaluate([item[0]])[0][0]) for item in feasible_candidates)
        near_best = [item for item in feasible_candidates
                     if float(model.evaluate([item[0]])[0][0]) >= highest - 1e-10]
        x, method, solver_success = max(near_best,
            key=lambda item: (item[2], float(model.evaluate([item[0]])[0][0])))
        bounds_ok = bool(np.all(x >= -TOL) and np.all(x <= np.tile(UPPER, 6) + TOL))
        rows, z, violation, intensity = city_metrics(model, x)
        _, _, slacks, comp, _ = model.evaluate([x])
        actual_cost = model.b0 + model.net_change(x)
        contributions = measure_contributions(base, x)
        city_order = sorted(rows, key=lambda row: row[3], reverse=True)
        measure_order = sorted(range(8), key=lambda k: contributions[k], reverse=True)
        top_cities = "; ".join(row[0] for row in city_order[:2])
        top_measures = "; ".join(NAMES[k] for k in measure_order[:3])
        record = dict(scenario=label, delta_fraction=delta, baseline_cost=model.b0,
            cost_cap=model.b0 + model.delta_abs, actual_cost=actual_cost,
            cap_slack=float(slacks[0, 0]), cap_utilization=actual_cost / (model.b0 + model.delta_abs),
            best_z=z, mean_h_after=float(comp["H"].mean()),
            mean_d_after=float(comp["D"].mean()), intensity=intensity,
            min_other_slack=float(slacks[0, 1:].min()),
            normalized_violation=violation, feasible=violation <= TOL and bounds_ok,
            selected_method=method, solver_success=solver_success,
            top_two_cities=top_cities, top_three_measures=top_measures)
        summary.append(record)
        for row in rows:
            city_rows.append((label, delta, *row))
        for i, city in enumerate(CITIES):
            for k, name in enumerate(NAMES):
                u_rows.append((label, delta, city, name, float(x.reshape(6, 8)[i, k]),
                               contributions[k]))
        print(f"Cost cap {label}: selected Z={z:.10f}, cap slack={record['cap_slack']:.3e}", flush=True)
    write_csv("Table_S6_Cost_Cap_Sensitivity.csv", tuple(summary[0]),
              ([row[key] for key in summary[0]] for row in summary))
    write_csv("Cost_Cap_GA_Run_Diagnostics.csv",
        ("scenario", "delta_fraction", "seed", "best_z", "post_repair_violation_rate",
         "pre_repair_violation_rate"), ga_runs)
    write_csv("Cost_Cap_City_Results.csv",
        ("scenario", "delta_fraction", "city", "h_before", "h_after", "h_gain",
         "d_before", "d_after", "d_gain", "mean_pollution_reduction_u",
         "mean_positive_measure_u"), city_rows)
    write_csv("Cost_Cap_Decisions.csv",
        ("scenario", "delta_fraction", "city", "indicator", "uij",
         "regional_h_contribution_of_indicator"), u_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), dpi=180)
    xx = [100 * row["delta_fraction"] for row in summary]
    for ax, key in zip(axes, ("mean_h_after", "mean_d_after")):
        yy = [row[key] for row in summary]
        ax.plot(xx, yy, color="#496c89", marker="o", lw=1.8, ms=6)
        ax.scatter([0], [yy[2]], color="#d08345", s=65, zorder=3)
        ax.set_xticks(xx, [f"{value:+g}%" if value else "Base" for value in xx])
        ax.set_xlabel("Change in cost-equivalent cap from B0")
        ax.grid(axis="y", alpha=.22)
        span = max(yy) - min(yy)
        ax.set_ylim(min(yy) - .18 * span, max(yy) + .18 * span)
    axes[0].set(ylabel="Mean WEH index H", title="WEH after optimization")
    axes[1].set(ylabel="Mean coupling coordination D", title="CCD after optimization")
    fig.tight_layout()
    fig.savefig(OUTPUT / "Fig_S4_Cost_Cap_Sensitivity.png")
    plt.close(fig)

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = Model.read()
    solutions, runs, histories = {}, [], {}
    for seed in SEEDS:
        sol, history, summary = run_ga(model, seed)
        solutions[seed], histories[seed] = sol, history
        runs.append(summary)
        print(f"GA seed {seed}: Z={summary['best_z']:.10f}, violation={summary['violation_rate']:.5%}", flush=True)
    best = max(runs, key=lambda r: r["best_z"])
    best_seed, best_x = best["seed"], solutions[best["seed"]]
    vals = np.array([r["best_z"] for r in runs])
    total_eval = sum(r["evaluated"] for r in runs)
    total_bad = sum(r["infeasible"] for r in runs)
    perf = [["run", r["seed"], r["best_z"], r["violation_rate"],
             r["infeasible"], r["evaluated"], r["raw_violation_rate"]] for r in runs]
    perf += [["summary", "best", vals.max(), total_bad / total_eval, total_bad, total_eval, ""],
             ["summary", "median", np.median(vals), "", "", "", ""],
             ["summary", "worst", vals.min(), "", "", "", ""],
             ["summary", "mean", vals.mean(), "", "", "", ""],
             ["summary", "SD_sample", vals.std(ddof=1), "", "", "", ""]]
    write_csv("Table_S2_GA_Performance.csv",
        ("record_type", "seed_or_statistic", "best_z", "violation_rate_post_repair",
         "infeasible_evaluations", "total_evaluations", "violation_rate_pre_repair"), perf)
    write_csv("GA_Convergence_Trace.csv", ("seed", "generation", "best_feasible_z", "feasible_fraction"),
              ((seed, *row) for seed, h in histories.items() for row in h))
    plot_convergence(histories)
    result, sz, sv, smin = slsqp_check(model, best_x)
    gz, gv, gslack, _, _ = model.evaluate([best_x])
    solver = [dict(solver="GA", z=float(gz[0]), violation=float(gv[0]),
                   min_slack=float(gslack[0].min()), success=True,
                   message=f"Best of 10 runs, seed {best_seed}", nit=GENERATIONS),
              dict(solver="SLSQP", z=sz, violation=sv, min_slack=smin,
                   success=bool(result.success), message=str(result.message), nit=result.nit)]
    write_csv("Table_S3_Solver_Check.csv",
              ("solver", "best_z", "normalized_violation", "minimum_raw_constraint_slack",
               "feasible_tol_1e-8", "reported_success", "iterations", "message"),
              ((r["solver"], r["z"], r["violation"], r["min_slack"],
                r["violation"] <= TOL, r["success"], r["nit"], r["message"]) for r in solver))
    write_csv("Solver_Decisions.csv", ("city", "indicator", "GA_uij", "SLSQP_uij", "difference"),
              ((city, name, best_x.reshape(6, 8)[i, k], result.x.reshape(6, 8)[i, k],
                result.x.reshape(6, 8)[i, k] - best_x.reshape(6, 8)[i, k])
               for i, city in enumerate(CITIES) for k, name in enumerate(NAMES)))
    # Keep the main GA solution as the sensitivity-analysis baseline.
    # The SLSQP result is reported only as a local-solver validation check.
    base_x = best_x
    scenarios = []
    for label, v in (("V=0.25", .25), ("V=0.30", .30),
                     ("Base", V0), ("V=0.45", .45), ("V=0.55", .55)):
        if label == "Base":
            scenarios.append((label, v, base_x))
            continue
        candidates = []
        for seed in (1, 2, 3):
            x, _, summary = run_ga(model, seed, v=v, generations=GENERATIONS)
            candidates.append(x)
            print(f"{label}, seed {seed}: Z={summary['best_z']:.10f}", flush=True)
        if model.evaluate([base_x], v=v)[1][0] <= TOL:
            candidates.append(base_x)
        x = max(candidates, key=lambda a: model.evaluate([a], v=v)[0][0])
        local, lz, lv, _ = slsqp_check(model, x, v=v)
        if lv <= TOL and lz >= model.evaluate([x], v=v)[0][0]:
            x = local.x
        scenarios.append((label, v, x))
    sens, city_rows, decision_rows, structure = [], [], [], []
    for label, v, x in scenarios:
        rows, z, viol, intensity = city_metrics(model, x, v=v)
        _, _, slacks, comp, _ = model.evaluate([x], v=v)
        record = dict(scenario=label, v_limit=v, best_z=z,
            mean_h_after=float(comp["H"].mean()), mean_d_after=float(comp["D"].mean()),
            intensity=intensity, normalized_violation=viol,
            min_raw_slack=float(slacks.min()), feasible=viol <= TOL)
        sens.append(record)
        city_rows += [[label, v, *row] for row in rows]
        contributions = measure_contributions(model, x)
        city_order = sorted(rows, key=lambda row: row[3], reverse=True)
        measure_order = sorted(range(8), key=lambda k: contributions[k], reverse=True)
        structure.append(dict(scenario=label,
            top_cities="; ".join(row[0] for row in city_order[:2]),
            top_measures="; ".join(NAMES[k] for k in measure_order[:3])))
        for i, city in enumerate(CITIES):
            for k, name in enumerate(NAMES):
                decision_rows.append((label, city, name, float(x.reshape(6, 8)[i, k]),
                                      float(x.reshape(6, 8)[i, k] / UPPER[k]), contributions[k]))
    write_csv("Table_S4_Sensitivity.csv",
        ("scenario", "v_limit", "city", "h_before", "h_after", "h_gain",
         "d_before", "d_after", "d_gain", "mean_pollution_reduction_u",
         "mean_positive_measure_u"), city_rows)
    write_csv("Sensitivity_Summary.csv", tuple(sens[0]), ([r[k] for k in sens[0]] for r in sens))
    write_csv("Sensitivity_Decisions.csv",
        ("scenario", "city", "indicator", "uij", "fraction_of_upper_rate",
         "regional_h_contribution_of_indicator"), decision_rows)
    write_csv("Strategy_Ranking.csv", ("scenario", "top_two_cities_by_h_gain",
        "top_three_measures_by_regional_h_contribution"),
        ((r["scenario"], r["top_cities"], r["top_measures"]) for r in structure))
    plot_sensitivity(sens)
    manifest = {"python": sys.version, "numpy": np.__version__,
                "scipy": __import__("scipy").__version__,
                "matplotlib": matplotlib.__version__,
                "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in (RAW_FILE, WEIGHT_FILE)},
                "source_model": "Corrected manuscript-consistent PSR/CCD model with within-subsystem weight renormalization",
                "seeds": list(SEEDS), "population": POPULATION,
                "generations": GENERATIONS, "v": V0, "cost_coefficients": COST.tolist(),
                "upper_rates": UPPER.tolist(), "violation_tolerance": TOL}
    (OUTPUT / "Reproducibility_Manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    coefficient_sweep()
    cost_cap_sweep()
    print(f"Finished: {OUTPUT}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--smoke-test":
        m = Model.read()
        print(f"Baseline WEH={m.baseline_components['H'].mean():.12f}; "
              f"baseline CCD={m.baseline_components['D'].mean():.12f}; "
              f"zero-adjustment Z={m.evaluate([np.zeros(48)])[0][0]:.12f}")
    elif len(sys.argv) == 2 and sys.argv[1] == "--cost-sweep":
        coefficient_sweep()
    elif len(sys.argv) == 2 and sys.argv[1] == "--budget-sweep":
        cost_cap_sweep()
    else:
        main()
