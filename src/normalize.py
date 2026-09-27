"""Text normalisation for business names and addresses.

Every record is mapped to a handful of canonical string fields that the
blocking and feature stages consume.  Nothing here is country specific except
general-knowledge abbreviation tables (US/India state codes, street types,
legal forms incl. French ones), so unseen country labels still work.
"""
import re
import unicodedata

from unidecode import unidecode

# ---------------------------------------------------------------- scripts
_TAMIL = re.compile("[\u0B80-\u0BFF]")
_MAL_FIX = {"ൽ": "ല്", "ൻ": "ന്", "ർ": "ര്", "ൺ": "ണ്", "ൾ": "ള്", "ൿ": "ക്", "റ്റ": "ട്ട"}
_NONLATIN = re.compile(r"[^\x00-\x7FÀ-ɏ -⁯]")


def to_ascii(s):
    if not s:
        return "", False
    nonlat = bool(_NONLATIN.search(s))
    if nonlat:
        for k, v in _MAL_FIX.items():
            s = s.replace(k, v)
        tamil = bool(_TAMIL.search(s))
        s = unidecode(s)
        if tamil:
            s = s.replace("c", "s")
        s = s.replace("oN", "o").replace("N", "n")
    else:
        s = unidecode(s)
    return s.lower(), nonlat


# ---------------------------------------------------------------- phonetic skeleton
# Coarse consonant skeleton.  Voiced/unvoiced pairs are merged because several
# Indic scripts (e.g. Tamil) do not distinguish them, so "Baba" written in Tamil
# transliterates to "paapaa".
_SK_PRE = [("ght", "t"), ("ng", "n"), ("x", "ks"), ("gi", "ji"), ("ge", "je"), ("ce", "se"), ("ci", "si"), ("cy", "sy"), ("sch", "s"), ("tion", "sn"),
           ("ph", "f"), ("sh", "s"), ("ch", "k"), ("ck", "k"), ("kh", "k"), ("gh", "g"), ("th", "t"),
           ("dh", "d"), ("bh", "b"), ("jh", "j"), ("wh", "v"), ("qu", "k")]
_SK_TABLE = str.maketrans({"b": "p", "v": "p", "w": "p", "f": "p", "d": "t", "g": "k", "c": "k", "q": "k",
                           "x": "k", "j": "s", "z": "s", "m": "n",
                           "a": None, "e": None, "i": None, "o": None, "u": None, "y": None, "h": None})


def skel(w):
    """Consonant skeleton robust to vowel typos and Indic transliteration."""
    for a, b in _SK_PRE:
        if a in w:
            w = w.replace(a, b)
    first = w[:1]
    w = w.translate(_SK_TABLE)
    out = []
    for ch in w:
        if not out or out[-1] != ch:
            out.append(ch)
    s = "".join(out)
    if not s:
        s = first
    return s


# ---------------------------------------------------------------- names
LEGAL_MAP = {
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp", "limited": "ltd", "ltd": "ltd",
    "private": "pvt", "pvt": "pvt", "pvte": "pvt", "company": "co", "co": "co", "llc": "llc", "lp": "lp",
    "llp": "llp", "pllc": "pllc", "pc": "pc", "plc": "plc", "pa": "pa", "sarl": "sarl", "sas": "sas",
    "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci", "snc": "snc", "gmbh": "gmbh", "opc": "opc",
    "public": "public", "cie": "cie", "ltda": "ltd", "lc": "llc", "l": None,
}
LEGAL = {v for v in LEGAL_MAP.values() if v}
LEGAL_SKEL = {"prpt": "pvt", "lntt": "ltd", "lnt": "ltd", "nkrprtt": "inc", "krprsn": "corp", "lp": "llp", "llp": "llp"}
NAME_STOP = {"the", "and", "ms", "m", "of", "dba", "a"}
_DBA = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|d\.b\.a\.?|dba|doing business as|trading as|t/a|a\.k\.a\.?|aka)\b")
_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|in|net|org|co|fr|us|biz|info|io)(?:\.[a-z]{2})?/?$")
_PHONE = re.compile(r"\d{7,}")
_DOTS_IN_ABBR = re.compile(r"\b[a-z](?:\.[a-z])+\b\.?")
_MS = re.compile(r"^\s*m\s*/\s*s\.?\s+|\bm/s\b")
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def _name_tokens(s, nonlat):
    s = _DOTS_IN_ABBR.sub(lambda m: m.group(0).replace(".", ""), s)
    s = s.replace("&", " and ").replace("+", " and ").replace("'", "").replace("`", "")
    s = _PUNCT.sub(" ", s)
    toks = s.split()
    out = []
    for t in toks:
        if t in LEGAL_MAP:
            v = LEGAL_MAP[t]
            if v:
                out.append(v)
            continue
        if nonlat and t in ("praa", "pra"):
            out.append("pvt"); continue
        if nonlat and t in ("li", "li"):
            out.append("ltd"); continue
        sk = skel(t)
        if sk in LEGAL_SKEL and len(t) >= 5:
            out.append(LEGAL_SKEL[sk]); continue
        out.append(t)
    return out


def norm_name(raw):
    """Returns dict of name fields."""
    s, nonlat = to_ascii(raw or "")
    s = s.split("|")[0]
    s = _MS.sub(" ", s)
    s = _PHONE.sub(" ", s)
    s = s.strip(" -<>#*~_=.,;:!?\"'[](){}")
    s = s.strip()
    dom = 0
    m = _DOMAIN.match(s.replace(" ", ""))
    if m and " " not in s.strip():
        dom = 1
        s = m.group(1).replace("-", " ")
    alt = ""
    parts = _DBA.split(s)
    if len(parts) > 1:
        alt = parts[0]
        s = " ".join(parts[1:])
    toks = _name_tokens(s, nonlat)
    core = [t for t in toks if t not in LEGAL and t not in NAME_STOP]
    if not core:
        core = [t for t in toks if t not in NAME_STOP] or toks
    legal = sorted({t for t in toks if t in LEGAL})
    alt_core = []
    if alt:
        at = _name_tokens(alt, nonlat)
        alt_core = [t for t in at if t not in LEGAL and t not in NAME_STOP]
    return {
        "nm": " ".join(toks),
        "core": " ".join(core),
        "skel": " ".join(skel(t) for t in core),
        "alt": " ".join(alt_core),
        "legal": " ".join(legal),
        "dom": dom,
        "nonlat": int(nonlat),
    }


# ---------------------------------------------------------------- addresses
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia", "pr": "puerto rico",
}
IN_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar", "cg": "chhattisgarh",
    "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh",
    "jh": "jharkhand", "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh", "mh": "maharashtra",
    "mn": "manipur", "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland", "od": "odisha", "or": "odisha",
    "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu", "ts": "telangana", "tg": "telangana",
    "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand", "ut": "uttarakhand", "wb": "west bengal",
    "dl": "delhi", "jk": "jammu and kashmir", "ch": "chandigarh", "py": "puducherry", "an": "andaman and nicobar",
    "la": "ladakh", "dn": "dadra and nagar haveli", "dd": "daman and diu", "ld": "lakshadweep",
}
STATE_NAMES = {}
for _d in (US_STATES, IN_STATES):
    for _code, _nm in _d.items():
        STATE_NAMES[_nm] = _nm.replace(" ", "_")
STATE_NAMES.update({"orissa": "odisha", "new delhi": None, "pondicherry": "puducherry", "uttaranchal": "uttarakhand"})

ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "dr": "drive", "drv": "drive", "ln": "lane", "ct": "court", "crt": "court", "blvd": "boulevard",
    "bd": "boulevard", "bld": "boulevard", "pl": "place", "trl": "trail", "tr": "trail", "cir": "circle",
    "pkwy": "parkway", "pky": "parkway", "hwy": "highway", "sq": "square", "ter": "terrace", "terr": "terrace",
    "pt": "point", "mt": "mount", "ft": "fort", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest", "apt": "apartment",
    "ste": "suite", "fl": "floor", "flr": "floor", "bldg": "building", "nr": "near", "opp": "opposite",
    "r": "rue", "imp": "impasse", "ch": "chemin", "rte": "route", "fbg": "faubourg", "cres": "crescent",
    "xing": "crossing", "expy": "expressway", "fwy": "freeway", "jct": "junction", "hts": "heights",
    "mtn": "mountain", "vly": "valley", "cyn": "canyon", "spg": "spring", "sta": "station", "ctr": "center",
    "centre": "center", "wy": "way", "aly": "alley", "cv": "cove", "pass": "pass", "pk": "park",
    "rd.": "road", "mg": "marg", "chowk": "chowk", "ngr": "nagar", "clny": "colony",
}
ADDR_STOP = {"city", "of", "village", "town", "township", "twp", "cdp", "the", "and", "null", "na", "none",
             "no", "number", "h", "hno", "house", "unit", "apartment", "suite", "floor", "po", "box", "near",
             "opposite", "behind", "de", "du", "des", "la", "le", "les", "d", "l", "c", "o", "nd", "rd", "th",
             "st", "tk", "cty", "india", "usa", "us", "france", "flat", "plot", "door", "shop", "office", "room",
             "bis", "ter", "b", "a"}
_NUM = re.compile(r"\d+")
_POBOX = re.compile(r"\bp\.?\s*o\.?\s*box\s*#?\s*\d+")


def norm_addr(raw, country_codes=None):
    s, nonlat = to_ascii(raw or "")
    if not s.strip():
        return {"ad": "", "nums": "", "pnum": "", "st": "", "city": "", "ad_empty": 1}
    s = _POBOX.sub(" ", s)
    comps = [c.strip() for c in s.split(",")]
    comps = [c for c in comps if c and c not in ("null", "n/a", "na", "none", "-")]
    states = []
    words_all = []
    nums_all = []
    pnum = ""
    city = ""
    for c in comps:
        c2 = _PUNCT.sub(" ", c.replace("'", "").replace(".", " ").replace("/", " ").replace("-", " "))
        c2 = _WS.sub(" ", c2).strip()
        if not c2:
            continue
        # whole component is a state?
        if c2 in STATE_NAMES:
            v = STATE_NAMES[c2]
            if v:
                states.append(v)
            else:
                words_all.extend(c2.split())
            continue
        toks = c2.split()
        if len(toks) == 1 and len(toks[0]) == 2 and not toks[0].isdigit():
            code = toks[0]
            if country_codes is not None:
                v = country_codes.get(code)
            else:
                v = US_STATES.get(code) or IN_STATES.get(code)
            if v:
                states.append(v.replace(" ", "_"))
                continue
        nums = []
        for t in _NUM.findall(c):
            t = t.lstrip("0") or "0"
            nums.append(t)
        if nums and not pnum:
            # primary number: first number of first component that starts with a number-ish token
            if re.match(r"^\W*(?:no\.?\s*|#\s*|h\.?\s*no\.?\s*)?\(?\d", c):
                pnum = nums[0]
        nums_all.extend(nums)
        ws = []
        for t in toks:
            if t.isdigit():
                continue
            if any(ch.isdigit() for ch in t):
                t = re.sub(r"\d+", "", t)
                if len(t) <= 1 or t in ("st", "nd", "rd", "th"):
                    continue
            t = ADDR_ABBR.get(t, t)
            if t in ADDR_STOP:
                continue
            ws.append(t)
        if ws and not nums and not city:
            city = " ".join(ws)
        words_all.extend(ws)
    if not pnum and nums_all:
        pnum = nums_all[0]
    return {
        "ad": " ".join(words_all),
        "nums": " ".join(nums_all),
        "pnum": pnum,
        "st": " ".join(sorted(set(states))),
        "city": city,
        "ad_empty": 0,
    }


STATE_TABLES = {"us": US_STATES, "usa": US_STATES, "united states": US_STATES, "india": IN_STATES, "in": IN_STATES}


def normalize_record(name, addr, country=""):
    d = norm_name(name)
    d.update(norm_addr(addr, STATE_TABLES.get((country or "").strip().lower(), {})))
    return d


if __name__ == "__main__":
    tests = [
        ("Xylogild doing business as Beacon Biotechnologies LLC", "127 Hungerford Avenue, Fl 0, Haysville, KS"),
        ("BEACONBIOTECHNOLOGIES.COM", "127 HUNGERFORD AVE, HAYSVILE, KS"),
        ("M/s Darsh Trading Pirvet Límited", "Rectangle No. 1, Behind Marriot Hotel Saket Commercial Complex D4, Saket, दिल्ली, South Delhi, New Delhi, 4Th Floor"),
        ("DARSH TRADING PRIVATE LIMITED - 8144132946", "4T FLOOR, RECTANGLE NO. 1, BEHIND MARRIOT HOTEL, SAKET, NEW DELHI, दिल्ली"),
        ("HEARTLAND LEAGUE, L.L.C.", "09616 NORTHRIDGE CT, RICHMOND, VA"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi"),
        ("അൽ കൺസ്ട്രക്ഷൻസ് ഫുഡ്സ് പ്രൈവറ്റ് ലിമിറ്റഡ്", "XVIII/C-37, FIRST FLOOR, THRISSUR, കേരളം"),
        ("Europ & Frères Distribution S.A.", "(41) Rue Des Thuyas, Lège-cap-ferret, Gironde"),
        ("SHIVSHAKTI VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com", "H.NO 204 C ROAD HOSHIARPUR, PUNJAB, Punjab"),
        ("-- Holloway Peak Inc Seafood", "5183 NORWALDO AVE, null, INDIANAPOLIS, IN"),
        ("Oller Vanguard Fuel [Corp]", "380 CIVIC CENTER DR, PO BOX 6599, AUGUSTA, ME"),
    ]
    tests.append(("Kelly Advisory, Inc", "301 1st Street, Chokio, MN"))
    for n, a in tests:
        print(n, "|", a)
        print("   ", normalize_record(n, a, "US"))
