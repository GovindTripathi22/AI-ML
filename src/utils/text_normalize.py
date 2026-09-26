"""
Text Normalization Module for Business Entity Resolution.

Provides pure Python / regex-based normalization for business names and addresses.
No external APIs or internet lookups are used.
"""

from typing import Any, NamedTuple, Optional, Set, Tuple
import re

# -----------------------------------------------------------------------------
# Legal Suffix Dictionary
# -----------------------------------------------------------------------------
LEGAL_SUFFIXES = {
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "inc": "incorporated",
    "co": "company",
    "llc": "llc",
    "llp": "llp",
    "plc": "plc",
}

# -----------------------------------------------------------------------------
# Address Abbreviation Dictionary
# -----------------------------------------------------------------------------
ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "apt": "apartment",
    "no": "number",
    "bldg": "building",
    "fl": "floor",
    "ste": "suite",
    "dist": "district",
    "nr": "near",
}

# -----------------------------------------------------------------------------
# Common Stopwords (for token set extraction)
# -----------------------------------------------------------------------------
STOPWORDS = {
    "the", "and", "of", "in", "on", "at", "for", "to", "a", "an", "is",
    "by", "with", "from", "or", "as", "into", "it",
}

# Landmark detection regex pattern
LANDMARK_PATTERN = re.compile(
    r"\b(near|nr|opposite|opp|behind|next\s+to)\b",
    re.IGNORECASE,
)

# Landmark phrase removal pattern (delimited by comma, semicolon, or line boundary)
LANDMARK_PHRASE_REMOVE = re.compile(
    r"(?:^|[,;]|\s+)\b(?:near|nr|opposite|opp|behind|next\s+to)\b.*?(?=[,;]|$)",
    re.IGNORECASE,
)

# PIN / Postal code regex (5 to 6 continuous digits)
PINCODE_PATTERN = re.compile(r"\b(\d{5,6})\b")


class AddressNormResult(tuple):
    """
    Tuple containing (normalized_address, has_landmark, pincode).
    
    Supports indexing, 3-element unpacking, and named attribute access.
    """
    def __new__(cls, address: str, has_landmark: bool, pincode: Optional[str] = None):
        return super().__new__(cls, (address, has_landmark, pincode))

    @property
    def normalized_address(self) -> str:
        return self[0]

    @property
    def has_landmark(self) -> bool:
        return self[1]

    @property
    def pincode(self) -> Optional[str]:
        return self[2]


def normalize_business_name(name: Any) -> str:
    """
    Normalize a business name:
    - Lowercase, strip
    - Replace '&' with 'and'
    - Remove punctuation except alphanumerics and spaces
    - Expand common legal suffix abbreviations using dictionary
    - Remove filler words like 'the' only at the start
    - Collapse multiple spaces

    Args:
        name: Raw business name (str or None).

    Returns:
        Clean normalized business name string.
    """
    if name is None:
        return ""
    s = str(name).strip().lower()
    if not s:
        return ""

    # Replace '&' with ' and ' before punctuation removal
    s = re.sub(r"&", " and ", s)

    # Remove punctuation except alphanumerics and spaces
    s = re.sub(r"[^a-z0-9\s]", " ", s)

    # Expand legal suffix abbreviations token by token
    tokens = [LEGAL_SUFFIXES.get(token, token) for token in s.split()]
    s = " ".join(tokens)

    # Remove filler words like 'the' only at the beginning
    s = re.sub(r"^the\s+", "", s)

    # Collapse multiple whitespace characters
    return re.sub(r"\s+", " ", s).strip()


def detect_landmark(address: Any) -> bool:
    """
    Detect whether landmark indicators exist in an address string.

    Args:
        address: Raw address string.

    Returns:
        True if a landmark keyword is detected, else False.
    """
    if address is None:
        return False
    s = str(address).strip()
    return bool(LANDMARK_PATTERN.search(s))


def extract_pincode(address: Any) -> Optional[str]:
    """
    Extract a 5 or 6 digit postal/PIN code from an address string.

    Args:
        address: Raw address string.

    Returns:
        PIN code string if found, otherwise None.
    """
    if address is None:
        return None
    s = str(address).strip()
    match = PINCODE_PATTERN.search(s)
    return match.group(1) if match else None


def normalize_address(address: Any) -> AddressNormResult:
    """
    Normalize an address string:
    - Lowercase, strip
    - Extract and separate postal/PIN code (regex for 5-6 digit numbers)
    - Detect landmark presence flag and remove landmark noise phrases
    - Remove punctuation while preserving numbers
    - Expand common address abbreviations (rd->road, st->street, ave->avenue, etc.)
    - Collapse whitespace

    Args:
        address: Raw address string.

    Returns:
        AddressNormResult tuple: (normalized_address, has_landmark, pincode)
    """
    if address is None:
        return AddressNormResult("", False, None)
    raw = str(address).strip().lower()
    if not raw:
        return AddressNormResult("", False, None)

    # 1. Detect landmark presence before removal
    has_landmark = bool(LANDMARK_PATTERN.search(raw))

    # 2. Extract PIN code (5 to 6 digits)
    pin_match = PINCODE_PATTERN.search(raw)
    pincode = pin_match.group(1) if pin_match else None

    # 3. Separate PIN code by removing it from the street text
    s = PINCODE_PATTERN.sub(" ", raw)

    # 4. Remove landmark noise phrases
    s = LANDMARK_PHRASE_REMOVE.sub(" ", s)

    # 5. Remove punctuation but preserve alphanumerics
    s = re.sub(r"[^a-z0-9\s]", " ", s)

    # 6. Expand address abbreviations
    tokens = [ADDRESS_ABBREVIATIONS.get(token, token) for token in s.split()]
    clean_address = re.sub(r"\s+", " ", " ".join(tokens)).strip()

    return AddressNormResult(clean_address, has_landmark, pincode)


def normalize_address_text(address: Any) -> str:
    """Helper returning only the cleaned address text."""
    return normalize_address(address).normalized_address


def extract_tokens(text: Any) -> Set[str]:
    """
    Tokenize text into a set of lowercase words for Jaccard comparisons,
    excluding common stopwords.

    Args:
        text: Input text string.

    Returns:
        Set of clean token strings.
    """
    if text is None:
        return set()
    s = str(text).lower()
    words = re.findall(r"\b[a-z0-9]+\b", s)
    return {w for w in words if w not in STOPWORDS}


def normalize_dataframe(df: Any) -> Any:
    """
    Apply text normalization across a source entity DataFrame.

    Appends the 4 normalized feature columns:
    - business_name_norm
    - business_address_norm
    - address_pincode
    - has_landmark

    Args:
        df: Pandas DataFrame with 'business_name' and 'business_address' columns.

    Returns:
        DataFrame with original and normalized columns.
    """
    df_out = df.copy()
    df_out["business_name_norm"] = df_out["business_name"].apply(normalize_business_name)

    norm_results = df_out["business_address"].apply(normalize_address)
    df_out["business_address_norm"] = norm_results.apply(lambda r: r.normalized_address)
    df_out["has_landmark"] = norm_results.apply(lambda r: r.has_landmark)
    df_out["address_pincode"] = norm_results.apply(lambda r: r.pincode if r.pincode else "")

    return df_out

