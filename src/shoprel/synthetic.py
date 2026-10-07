"""Generate a small ESCI-shaped dataset for demos, tests and CI.

The real ESCI data is ~2.6M judgements and needs a download, so this module
builds a toy catalog with the same columns and the same failure modes that
make product search hard:

* Complements whose titles repeat the query words ("case for galaxy phone"),
  which fool pure keyword matching.
* Substitutes that share the product type but miss a brand, color or size
  the shopper asked for.
* Queries that use synonyms ("sneakers") the title never mentions.

Numbers measured on this data are only a sanity check. Real results come
from running the pipeline on the actual dataset (see the README).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shoprel.data import prepare

# product type -> (category, query synonyms, complement types, attributes)
CATALOG: dict[str, dict[str, list[str] | str]] = {
    "running shoes": {
        "cat": "footwear",
        "syn": ["sneakers", "running trainers", "jogging shoes"],
        "comp": ["running socks", "shoe insoles"],
        "attr": ["size 9", "size 10", "size 11", "wide fit", "lightweight"],
    },
    "hiking boots": {
        "cat": "footwear",
        "syn": ["trekking boots", "trail boots"],
        "comp": ["running socks", "shoe insoles"],
        "attr": ["waterproof", "size 9", "size 10", "ankle support"],
    },
    "running socks": {
        "cat": "footwear",
        "syn": ["athletic socks", "sport socks"],
        "comp": ["running shoes"],
        "attr": ["6 pack", "cushioned", "no show"],
    },
    "shoe insoles": {
        "cat": "footwear",
        "syn": ["foot inserts", "arch support inserts"],
        "comp": ["running shoes", "hiking boots"],
        "attr": ["gel", "memory foam", "orthotic"],
    },
    "phone case": {
        "cat": "electronics",
        "syn": ["phone cover", "protective cover"],
        "comp": ["smartphone", "screen protector"],
        "attr": ["shockproof", "slim", "with card holder"],
    },
    "smartphone": {
        "cat": "electronics",
        "syn": ["cell phone", "mobile phone", "android phone"],
        "comp": ["phone case", "screen protector", "usb c charger"],
        "attr": ["128gb", "256gb", "unlocked", "dual sim"],
    },
    "screen protector": {
        "cat": "electronics",
        "syn": ["tempered glass", "screen guard"],
        "comp": ["smartphone", "phone case"],
        "attr": ["2 pack", "anti glare", "privacy"],
    },
    "usb c charger": {
        "cat": "electronics",
        "syn": ["wall charger", "fast charger", "power adapter"],
        "comp": ["smartphone", "laptop"],
        "attr": ["65w", "20w", "2 port"],
    },
    "laptop": {
        "cat": "electronics",
        "syn": ["notebook computer", "ultrabook"],
        "comp": ["laptop sleeve", "usb c charger", "wireless mouse"],
        "attr": ["16gb ram", "512gb ssd", "14 inch", "15.6 inch"],
    },
    "laptop sleeve": {
        "cat": "electronics",
        "syn": ["laptop bag", "notebook case"],
        "comp": ["laptop"],
        "attr": ["14 inch", "15.6 inch", "water resistant"],
    },
    "wireless mouse": {
        "cat": "electronics",
        "syn": ["bluetooth mouse", "cordless mouse"],
        "comp": ["laptop", "mouse pad"],
        "attr": ["silent click", "ergonomic", "rechargeable"],
    },
    "mouse pad": {
        "cat": "electronics",
        "syn": ["mouse mat", "desk mat"],
        "comp": ["wireless mouse"],
        "attr": ["extended", "non slip", "xl"],
    },
    "coffee maker": {
        "cat": "kitchen",
        "syn": ["coffee machine", "drip brewer"],
        "comp": ["coffee filters", "coffee beans"],
        "attr": ["12 cup", "programmable", "single serve"],
    },
    "coffee filters": {
        "cat": "kitchen",
        "syn": ["paper filters", "basket filters"],
        "comp": ["coffee maker"],
        "attr": ["200 count", "unbleached", "cone"],
    },
    "coffee beans": {
        "cat": "kitchen",
        "syn": ["whole bean coffee", "espresso beans"],
        "comp": ["coffee maker", "coffee grinder"],
        "attr": ["dark roast", "medium roast", "2 lb"],
    },
    "coffee grinder": {
        "cat": "kitchen",
        "syn": ["burr grinder", "bean grinder"],
        "comp": ["coffee beans"],
        "attr": ["electric", "manual", "adjustable"],
    },
    "yoga mat": {
        "cat": "fitness",
        "syn": ["exercise mat", "workout mat"],
        "comp": ["yoga blocks", "water bottle"],
        "attr": ["6mm", "non slip", "extra thick"],
    },
    "yoga blocks": {
        "cat": "fitness",
        "syn": ["foam blocks", "yoga bricks"],
        "comp": ["yoga mat"],
        "attr": ["2 pack", "cork", "high density"],
    },
    "water bottle": {
        "cat": "fitness",
        "syn": ["insulated bottle", "sports bottle"],
        "comp": ["yoga mat"],
        "attr": ["32 oz", "stainless steel", "bpa free"],
    },
    "dumbbells": {
        "cat": "fitness",
        "syn": ["hand weights", "free weights"],
        "comp": ["dumbbell rack", "workout gloves"],
        "attr": ["adjustable", "10 lb", "pair"],
    },
    "dumbbell rack": {
        "cat": "fitness",
        "syn": ["weight rack", "weight stand"],
        "comp": ["dumbbells"],
        "attr": ["3 tier", "steel"],
    },
    "workout gloves": {
        "cat": "fitness",
        "syn": ["gym gloves", "lifting gloves"],
        "comp": ["dumbbells"],
        "attr": ["padded", "full finger", "breathable"],
    },
}

BRANDS = {
    "footwear": ["Stridex", "Trailmark", "Nimbus", "Pacer", "Ridgeway"],
    "electronics": ["Voltix", "Galaxa", "Pixelon", "Corelink", "Ampere"],
    "kitchen": ["Brewhaus", "Morning Co", "Kettleworks", "Roastly"],
    "fitness": ["Flexa", "Ironpeak", "Zenfit", "Corestrong"],
}
COLORS = ["black", "white", "blue", "red", "grey", "green", "pink"]
ADJECTIVES = ["premium", "classic", "pro", "everyday", "ultra", "new", "deluxe"]


def _make_catalog(rng: np.random.Generator, per_type: int) -> pd.DataFrame:
    rows = []
    pid = 0
    for ptype, spec in CATALOG.items():
        brands = BRANDS[spec["cat"]]
        for _ in range(per_type):
            brand = str(rng.choice(brands))
            color = str(rng.choice(COLORS))
            attr = str(rng.choice(spec["attr"]))
            adj = str(rng.choice(ADJECTIVES))
            # Complements often name what they go with ("case for laptop").
            for_part = ""
            if rng.random() < 0.6:
                for_part = f" for {rng.choice(spec['comp'])}"
            title = f"{brand} {adj} {ptype}{for_part} {attr} - {color}"
            syn = str(rng.choice(spec["syn"]))
            desc = f"A {adj} {syn} from {brand}. {attr.capitalize()}, {color} finish."
            bullets = f"{syn} | {attr} | {spec['cat']}"
            rows.append(
                {
                    "product_id": f"P{pid:06d}",
                    "ptype": ptype,
                    "category": spec["cat"],
                    "product_title": title,
                    "product_description": desc,
                    "product_bullet_point": bullets,
                    "product_brand": brand,
                    "product_color": color,
                    "attr": attr,
                    "product_locale": "us",
                }
            )
            pid += 1
    return pd.DataFrame(rows)


_FIELD_COLS = {"brand": "product_brand", "color": "product_color", "attr": "attr"}


def _label(target: dict, product: pd.Series) -> str:
    """ESCI label of a product for a query intent ``target``."""
    if product["ptype"] == target["ptype"]:
        for field, col in _FIELD_COLS.items():
            want = target.get(field)
            if want is not None and want != product[col]:
                return "S"
        return "E"
    if product["ptype"] in CATALOG[target["ptype"]]["comp"]:
        return "C"
    return "I"


def make_synthetic_esci(
    n_queries: int = 400,
    candidates_per_query: int = 16,
    products_per_type: int = 40,
    label_noise: float = 0.05,
    seed: int = 7,
) -> pd.DataFrame:
    """Return a DataFrame with the same columns as :func:`shoprel.data.load_esci`."""
    rng = np.random.default_rng(seed)
    catalog = _make_catalog(rng, products_per_type)
    by_type = {t: g for t, g in catalog.groupby("ptype")}
    labels = np.array(["E", "S", "C", "I"])

    rows = []
    for qid in range(n_queries):
        anchor = catalog.iloc[int(rng.integers(len(catalog)))]
        ptype = anchor["ptype"]
        target = {"ptype": ptype}
        parts = []
        if rng.random() < 0.5:
            target["brand"] = anchor["product_brand"]
            parts.append(anchor["product_brand"].lower())
        if rng.random() < 0.35:
            target["color"] = anchor["product_color"]
            parts.append(anchor["product_color"])
        # Shoppers often use a synonym the title never contains.
        type_words = str(rng.choice(CATALOG[ptype]["syn"])) if rng.random() < 0.4 else ptype
        parts.append(type_words)
        if rng.random() < 0.3:
            target["attr"] = anchor["attr"]
            parts.append(anchor["attr"])
        query = " ".join(parts)

        # Candidate pool: mostly same type, some complements, some random.
        same = by_type[ptype].sample(
            n=candidates_per_query // 2, random_state=int(rng.integers(1e9))
        )
        comp_types = CATALOG[ptype]["comp"]
        comp = pd.concat([by_type[t] for t in comp_types]).sample(
            n=candidates_per_query // 4, random_state=int(rng.integers(1e9))
        )
        rest = catalog[~catalog["ptype"].isin([ptype, *comp_types])].sample(
            n=candidates_per_query - len(same) - len(comp), random_state=int(rng.integers(1e9))
        )
        pool = pd.concat([same, comp, rest]).drop_duplicates("product_id")
        # Make sure each list has at least one exact match.
        pool = pd.concat([anchor.to_frame().T, pool]).drop_duplicates("product_id")

        for _, prod in pool.iterrows():
            label = _label(target, prod)
            if rng.random() < label_noise:
                label = str(rng.choice(labels))
            rows.append({"query_id": qid, "query": query, "esci_label": label, **prod.to_dict()})

    df = pd.DataFrame(rows)
    df["example_id"] = np.arange(len(df))
    df["small_version"] = 1
    df["large_version"] = 1
    df = df.drop(columns=["ptype", "category", "attr"])
    return prepare(df)
