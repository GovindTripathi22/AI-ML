"""
Unit tests for text normalization module (src/utils/text_normalize.py).
"""

import pytest
from src.utils.text_normalize import (
    detect_landmark,
    extract_pincode,
    extract_tokens,
    normalize_address,
    normalize_address_text,
    normalize_business_name,
)


class TestBusinessNameNormalization:
    """Tests for normalize_business_name."""

    def test_legal_abbreviation_expansion(self):
        assert normalize_business_name("Infosys Pvt. Ltd.") == "infosys private limited"
        assert normalize_business_name("Acme Corp.") == "acme corporation"
        assert normalize_business_name("Apple Inc.") == "apple incorporated"
        assert normalize_business_name("Ford Motor Co.") == "ford motor company"
        assert normalize_business_name("Apex LLC") == "apex llc"
        assert normalize_business_name("Alpha LLP") == "alpha llp"
        assert normalize_business_name("British Gas PLC") == "british gas plc"
        assert normalize_business_name("Barnes & Noble") == "barnes and noble"

    def test_leading_the_removal(self):
        assert normalize_business_name("The Boeing Company") == "boeing company"
        assert normalize_business_name("The Goldman Sachs Group") == "goldman sachs group"
        # Inner 'the' should not be removed
        assert normalize_business_name("Over The Moon Ltd") == "over the moon limited"

    def test_punctuation_and_whitespace(self):
        assert normalize_business_name("  Johnson & Johnson,   Inc.!  ") == "johnson and johnson incorporated"
        assert normalize_business_name("L'Oreal (Paris) Ltd.") == "l oreal paris limited"

    def test_edge_cases(self):
        assert normalize_business_name("") == ""
        assert normalize_business_name(None) == ""
        assert normalize_business_name("   ") == ""
        assert normalize_business_name("12345") == "12345"
        assert normalize_business_name("404 Technologies") == "404 technologies"


class TestAddressNormalization:
    """Tests for normalize_address and related helpers."""

    def test_address_abbreviation_expansion(self):
        addr, has_lm, pin = normalize_address("100 Main St, Ste 400, New York")
        assert "street" in addr
        assert "suite" in addr
        assert not has_lm

        addr2, _, _ = normalize_address("Park Ave, 3rd Fl, Bldg A, Apt 2B, dist 9")
        assert "avenue" in addr2
        assert "floor" in addr2
        assert "building" in addr2
        assert "apartment" in addr2
        assert "district" in addr2

        addr3, _, _ = normalize_address("Sunset Blvd, No 45, Ring Rd")
        assert "boulevard" in addr3
        assert "number" in addr3
        assert "road" in addr3

    def test_landmark_detection_and_removal(self):
        # Near
        addr1, has_lm1, pin1 = normalize_address("12 MG Road, Near SBI Bank, Bangalore 560001")
        assert has_lm1 is True
        assert "near" not in addr1
        assert "sbi" not in addr1
        assert pin1 == "560001"
        assert "12 mg road bangalore" in addr1

        # Opposite / Opp
        addr2, has_lm2, _ = normalize_address("Plot 7, Opposite City Hospital, Pune")
        assert has_lm2 is True
        assert "opposite" not in addr2
        assert "hospital" not in addr2

        # Behind
        addr3, has_lm3, _ = normalize_address("Behind Post Office, Ring Road")
        assert has_lm3 is True
        assert "behind" not in addr3
        assert "post office" not in addr3

        # Next to
        addr4, has_lm4, _ = normalize_address("Next to Metro Station, Connaught Place")
        assert has_lm4 is True
        assert "metro" not in addr4

    def test_pincode_extraction(self):
        # 5-digit US ZIP code
        assert extract_pincode("123 Main St, New York, NY 10001") == "10001"
        # 6-digit Indian PIN code
        assert extract_pincode("Outer Ring Rd, Bangalore 560103") == "560103"
        # No pincode
        assert extract_pincode("Baker Street, London") is None

        # Verify separation in normalize_address
        clean_addr, _, pin = normalize_address("742 Evergreen Terrace, Springfield 97477")
        assert pin == "97477"
        assert "97477" not in clean_addr

    def test_edge_cases(self):
        res_empty = normalize_address("")
        assert res_empty.normalized_address == ""
        assert res_empty.has_landmark is False
        assert res_empty.pincode is None

        res_none = normalize_address(None)
        assert res_none.normalized_address == ""
        assert res_none.has_landmark is False
        assert res_none.pincode is None

        res_pin_only = normalize_address("560001")
        assert res_pin_only.normalized_address == ""
        assert res_pin_only.has_landmark is False
        assert res_pin_only.pincode == "560001"


class TestTokenExtraction:
    """Tests for extract_tokens."""

    def test_token_extraction_and_stopwords(self):
        tokens = extract_tokens("The quick brown fox and the lazy dog of Wall Street")
        assert "the" not in tokens
        assert "and" not in tokens
        assert "of" not in tokens
        assert "quick" in tokens
        assert "fox" in tokens
        assert "wall" in tokens
        assert "street" in tokens

    def test_edge_cases(self):
        assert extract_tokens("") == set()
        assert extract_tokens(None) == set()
        assert extract_tokens("The and of") == set()
