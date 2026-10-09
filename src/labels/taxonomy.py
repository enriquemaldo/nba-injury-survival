"""Step 4: rule-based injury taxonomy applied to transaction notes.

Every rule lives in this file so the label definition is documented and
reproducible. Tiers are NESTED: strict ⊂ proxy ⊂ broad.

  strict      Muscle/tendon site (hamstring, calf, Achilles, groin/adductor,
              quad, hip flexor) + explicit strain/pull/tear/rupture word,
              and no contact language in the note.
  proxy       Same sites + any soft-tissue complaint (adds sore, tight,
              tendinitis, spasm...), no contact language. Likely PRIMARY label.
  broad       Any lower-extremity musculoskeletal injury (adds knee, ankle,
              foot, hip, toe, etc.), contact or not.
  other_injury  Injury outside the lower extremity (back, shoulder, hand, head...).
  uncertain   Looks injury-related but no identifiable body site.
  non_injury  Rest, illness, personal, suspension, protocols, etc.

Edit the regexes, re-run, and record each change in your update notes.
"""
import re

import pandas as pd

def _rx(words):
    return re.compile(r"\b(" + "|".join(words) + r")", re.IGNORECASE)

NON_INJURY = _rx([
    r"rest", r"illness", r"ill\b", r"flu", r"virus", r"covid", r"health and safety",
    r"protocol", r"personal", r"family", r"suspen", r"conditioning", r"g league",
    r"coach'?s decision", r"not with team", r"birth", r"bereavement", r"migraine",
    r"dental", r"tooth", r"tonsil", r"append", r"food poisoning", r"gastro", r"gasro",
    r"nausea", r"cold\b", r"strep", r"pneumonia",
    # added after auditing uncertain_notes.csv (2016-20 Kaggle run):
    r"respiratory", r"infection", r"stomach", r"sinus", r"mononucleosis", r"mono\b",
    r"blood", r"clot", r"embolism", r"kidney", r"heart", r"arrhythm", r"dehydra",
    r"fatigue", r"quarantine", r"medical test", r"thrombocyt", r"dizz", r"vestibular",
    r"neuropath", r"abscess", r"oral surgery", r"throat", r"ailment",
])

MUSCLE_SITES = {
    "hamstring": _rx([r"hamstring"]),
    "calf": _rx([r"calf", r"calves", r"soleus", r"gastrocnemius"]),
    "achilles": _rx([r"achilles"]),
    # Sports hernia / athletic pubalgia: groin-region soft-tissue injury.
    "groin_adductor": _rx([r"groin", r"adductor", r"sports hernia", r"pubalgia"]),
    "quad": _rx([r"quad", r"thigh"]),
    "hip_flexor": _rx([r"hip flexor"]),
}

OTHER_LOWER_SITES = {
    "knee": _rx([r"knee", r"patell", r"menisc", r"acl\b", r"mcl\b", r"pcl\b", r"lcl\b"]),
    "ankle": _rx([r"ankle"]),
    "foot": _rx([r"foot", r"feet", r"plantar", r"heel", r"toe", r"metatars", r"navicular", r"arch\b"]),
    "hip": _rx([r"hip", r"glute", r"pelvi", r"sacro", r"si joint"]),
    "lower_leg": _rx([r"shin", r"tibia", r"fibula", r"lower leg", r"leg\b"]),
}

UPPER_SITES = _rx([
    r"back", r"spine", r"lumbar", r"neck", r"shoulder", r"elbow", r"wrist",
    r"hand", r"finger", r"thumb", r"head\b", r"concussion", r"face", r"nose",
    r"eye", r"jaw", r"rib", r"chest", r"abdom", r"oblique", r"core", r"arm",
    # added after auditing uncertain_notes.csv:
    r"facial", r"orbital", r"cheek", r"sinus fractur", r"cornea", r"mouth",
    r"sternum", r"pectoral", r"tailbone", r"coccyx", r"collarbone", r"clavicle",
    r"forearm", r"rotator", r"tricep", r"bicep", r"thoracic", r"cervical", r"ulnar",
])

STRAIN = _rx([r"strain", r"pull", r"tear", r"torn", r"ruptur"])
SOFT_TISSUE = _rx([
    r"strain", r"pull", r"tear", r"torn", r"ruptur", r"sore", r"tight",
    r"tendin", r"tendon", r"spasm", r"cramp", r"injur",
])
# Contusions/bruises come from a blow, so they count as contact language
# (a "bruised left quad" is broad-tier, never strict/proxy).
CONTACT = _rx([
    r"bruis", r"contusion",
    r"collision", r"collid", r"fell\b", r"fall\b", r"landed on", r"stepped on",
    r"kicked", r"knee[- ]to", r"hit\b", r"elbow(ed)? to", r"contact",
    r"inadvertent", r"fractur", r"broken", r"dislocat",
])
INJURY_HINT = _rx([r"injur", r"sore", r"strain", r"sprain", r"surgery", r"pain",
                   r"tendin", r"\bdtd\b", r"\bil\b", r"out for season", r"out indefinitely"])

TIER_ORDER = ["strict", "proxy", "broad", "other_injury", "uncertain", "non_injury"]


def classify(note: str) -> dict:
    """Return tier + body site for one transaction note."""
    n = note or ""
    muscle = [k for k, rx in MUSCLE_SITES.items() if rx.search(n)]
    lower = [k for k, rx in OTHER_LOWER_SITES.items() if rx.search(n)]
    contact = bool(CONTACT.search(n))

    if muscle and STRAIN.search(n) and not contact:
        return {"tier": "strict", "site": muscle[0], "contact_language": False}
    if muscle and SOFT_TISSUE.search(n) and not contact:
        return {"tier": "proxy", "site": muscle[0], "contact_language": False}
    if muscle or lower:
        return {"tier": "broad", "site": (muscle or lower)[0], "contact_language": contact}
    if UPPER_SITES.search(n):
        return {"tier": "other_injury", "site": "upper_or_trunk", "contact_language": contact}
    if NON_INJURY.search(n):
        return {"tier": "non_injury", "site": None, "contact_language": False}
    if INJURY_HINT.search(n):
        return {"tier": "uncertain", "site": None, "contact_language": contact}
    return {"tier": "non_injury", "site": None, "contact_language": False}


def label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.join(pd.DataFrame([classify(x) for x in df["Notes"].fillna("")], index=df.index))
    # Nested flags: a strict row also counts as proxy and broad.
    rank = {t: i for i, t in enumerate(TIER_ORDER)}
    r = out["tier"].map(rank)
    out["is_strict"] = r <= rank["strict"]
    out["is_proxy"] = r <= rank["proxy"]
    out["is_broad"] = r <= rank["broad"]
    return out
