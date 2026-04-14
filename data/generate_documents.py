"""
FintechCID -- Synthetic Evidence Generation Engine
=====================================================
Reads the 1M-row Parquet dataset and generates two artefacts per
transaction (sampled):

  1. A synthetic PDF invoice / KYC document  →  data/sample_docs/<tid>_invoice.pdf
  2. A Vectorless RAG JSON tree              →  data/sample_docs/<tid>_tree.json

Fraud injection
---------------
For transactions flagged is_fraud == 1, deliberate discrepancies are
injected into the JSON tree to simulate real-world evidence tampering:
  • Amount spoofed:   tree amount ≠ actual transaction amount
  • Merchant spoofed: one character replaced with a lookalike (l33t-style)

Scope
-----
Generating PDFs for all 50,000 fraud rows in the 1M dataset would take
too long for a demo environment.  This script caps at:
  FRAUD_SAMPLE  = 50  fraud transactions
  LEGIT_SAMPLE  = 10  legitimate transactions
Total output: 60 PDF/JSON pairs in data/sample_docs/.

Usage:
    python data/generate_documents.py          (from project root)
"""

import json
import pathlib
import random

import pandas as pd
from fpdf import FPDF

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR    = pathlib.Path(__file__).parent.parent
DATA_PATH   = BASE_DIR / "data" / "raw" / "transactions.parquet"
DOCS_DIR    = BASE_DIR / "data" / "sample_docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

FRAUD_SAMPLE = 50
LEGIT_SAMPLE = 10
RANDOM_SEED  = 42

random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Fake personal / merchant data pools
# ---------------------------------------------------------------------------
FAKE_NAMES = [
    "Alistair Pemberton", "Cecilia Hawthorne", "Desmond Okafor",
    "Fatima Al-Rashid",   "Gonzalo Reyes",     "Hiroshi Tanaka",
    "Ingrid Svensson",    "Joaquin Varela",     "Katerina Novak",
    "Liam O'Sullivan",    "Mei-Ling Zhao",      "Nadia Petrov",
    "Omar Farouk",        "Priya Sharma",       "Rafael Costa",
]
FAKE_ADDRESSES = [
    "14 Whitechapel Rd, London E1 1EW, UK",
    "7 Rue de Rivoli, 75001 Paris, France",
    "221B Baker Street, London NW1 6XE, UK",
    "500 Orchard Road, Singapore 238880",
    "1 Infinite Loop, Cupertino CA 95014, USA",
    "88 Queen St, Auckland 1010, New Zealand",
    "Königsallee 92, 40212 Düsseldorf, Germany",
    "Via Condotti 12, 00187 Roma, Italy",
]
MERCHANT_DISPLAY = {
    "grocery":        "FreshMart Superstore",
    "restaurant":     "The Golden Fork Brasserie",
    "online_retail":  "ShopNow Global Ltd.",
    "travel":         "SkyRoute Travel Agency",
    "gas_station":    "PetroPlus Service Station",
    "electronics":    "TechZone Electronics",
    "healthcare":     "MediCare Pharmacy Group",
    "entertainment":  "StarPlex Entertainment",
}
# Lookalike spoofs for fraud injection (one char swapped)
MERCHANT_SPOOF = {
    "grocery":        "Fr3shMart Superstore",
    "restaurant":     "The G0lden Fork Brasserie",
    "online_retail":  "Sh0pNow Global Ltd.",
    "travel":         "SkyR0ute Travel Agency",
    "gas_station":    "Petr0Plus Service Station",
    "electronics":    "T3chZone Electronics",
    "healthcare":     "MediC4re Pharmacy Group",
    "entertainment":  "St4rPlex Entertainment",
}


# ---------------------------------------------------------------------------
# PDF generator
# ---------------------------------------------------------------------------

class InvoicePDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_fill_color(30, 60, 114)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, "FINTECHCID -- SYNTHETIC INVOICE", align="C", fill=True)
        self.ln(4)
        self.set_text_color(180, 0, 0)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 5, "SYNTHETIC DATA --NOT A REAL DOCUMENT --FOR DEMO PURPOSES ONLY",
                  align="C")
        self.ln(6)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()} | FintechCID Synthetic Evidence",
                  align="C")


def _make_pdf(tid: str, row: pd.Series, is_fraud: bool,
              display_amount: float, merchant_display_name: str) -> None:
    """Render a single synthetic invoice PDF."""
    name    = random.choice(FAKE_NAMES)
    address = random.choice(FAKE_ADDRESSES)
    ts_str  = str(row["timestamp"])[:19].replace("T", " ")

    pdf = InvoicePDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---- Merchant block ----
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, merchant_display_name, fill=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, random.choice(FAKE_ADDRESSES))
    pdf.ln(8)
    pdf.set_text_color(0, 0, 0)

    # ---- Invoice meta ----
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(50, 6, "Invoice No:")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"INV-{tid[:8].upper()}")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(50, 6, "Transaction ID:")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, tid)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(50, 6, "Date / Time:")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, ts_str)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(50, 6, "Customer:")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, name)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(50, 6, "Billing Address:")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, address)
    pdf.ln(10)

    # ---- Line items ----
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(110, 7, "Description", fill=True)
    pdf.cell(40,  7, "Qty", align="C", fill=True)
    pdf.cell(40,  7, "Amount (USD)", align="R", fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    item_desc = f"{merchant_display_name} --Service/Product"
    pdf.cell(110, 7, item_desc)
    pdf.cell(40,  7, "1", align="C")
    pdf.cell(40,  7, f"${display_amount:,.2f}", align="R")
    pdf.ln()

    # ---- Totals ----
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(150, 7, "TOTAL DUE:", align="R")
    if is_fraud:
        pdf.set_text_color(180, 0, 0)
    pdf.cell(40, 7, f"${display_amount:,.2f}", align="R")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    # ---- Fraud watermark ----
    if is_fraud:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(180, 0, 0)
        pdf.cell(0, 6,
                 "[!] FORENSIC FLAG: Amount and/or merchant may not match transaction record.",
                 align="C")
        pdf.ln(4)
        pdf.set_text_color(0, 0, 0)

    out_path = DOCS_DIR / f"{tid}_invoice.pdf"
    pdf.output(str(out_path))


# ---------------------------------------------------------------------------
# JSON RAG tree generator
# ---------------------------------------------------------------------------

def _make_json_tree(tid: str, row: pd.Series, is_fraud: bool,
                    display_amount: float, merchant_display_name: str) -> dict:
    """Build a hierarchical document tree that mirrors the PDF content."""
    ts_str = str(row["timestamp"])[:19]

    tree = {
        "DocumentRoot": {
            "Header": {
                "InvoiceNumber":  f"INV-{tid[:8].upper()}",
                "TransactionID":  tid,
                "DateTime":       ts_str,
                "DocumentType":   "Commercial Invoice",
            },
            "MerchantInfo": {
                "DisplayName":    merchant_display_name,
                "Category":       row["merchant_category"],
                "Address":        random.choice(FAKE_ADDRESSES),
            },
            "CustomerInfo": {
                "Name":           random.choice(FAKE_NAMES),
                "BillingAddress": random.choice(FAKE_ADDRESSES),
            },
            "Body": {
                "LineItems": [
                    {
                        "Description": f"{merchant_display_name} --Service/Product",
                        "Quantity":    1,
                        "UnitPrice":   round(display_amount, 2),
                    }
                ],
                "Subtotal":       round(display_amount, 2),
                "Tax":            round(display_amount * 0.08, 2),
                "TotalDue":       round(display_amount * 1.08, 2),
            },
            "Metadata": {
                "is_fraud_label":   int(row["is_fraud"]),
                "actual_tx_amount": round(float(row["amount"]), 2),
                "discrepancy_injected": is_fraud,
            },
        }
    }
    return tree


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[DocGen] Reading parquet from {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    print(f"[DocGen] Loaded {len(df):,} rows")

    fraud_df = df[df["is_fraud"] == 1].head(FRAUD_SAMPLE)
    legit_df = df[df["is_fraud"] == 0].head(LEGIT_SAMPLE)
    sample   = pd.concat([fraud_df, legit_df], ignore_index=True)

    print(f"[DocGen] Generating documents for {len(fraud_df)} fraud + "
          f"{len(legit_df)} legit transactions ...")

    generated = []
    for _, row in sample.iterrows():
        tid      = str(row["transaction_id"])
        is_fraud = bool(row["is_fraud"])
        merchant = str(row["merchant_category"])
        actual_amount = float(row["amount"])

        # --- Fraud injection ---
        if is_fraud:
            # Spoof display amount: multiply by a random factor (5-15x) so the
            # invoice amount clearly differs from the transaction amount.
            spoof_factor   = random.uniform(5, 15)
            display_amount = round(actual_amount * spoof_factor, 2)
            merchant_name  = MERCHANT_SPOOF.get(merchant, merchant)
        else:
            display_amount = actual_amount
            merchant_name  = MERCHANT_DISPLAY.get(merchant, merchant)

        # PDF
        _make_pdf(tid, row, is_fraud, display_amount, merchant_name)

        # JSON tree
        tree = _make_json_tree(tid, row, is_fraud, display_amount, merchant_name)
        json_path = DOCS_DIR / f"{tid}_tree.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(tree, fh, indent=2, default=str)

        generated.append(tid)

    print(f"[DocGen] Done. {len(generated)} PDF + JSON pairs written to {DOCS_DIR}")

    # Emit the first fraud transaction_id for use in tests
    first_fraud_tid = str(fraud_df.iloc[0]["transaction_id"])
    print(f"[DocGen] Sample fraud transaction_id for testing: {first_fraud_tid}")

    # Save the index of generated IDs for downstream use
    index_path = DOCS_DIR / "generated_index.json"
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump({
            "fraud_tids": [str(r["transaction_id"]) for _, r in fraud_df.iterrows()],
            "legit_tids": [str(r["transaction_id"]) for _, r in legit_df.iterrows()],
        }, fh, indent=2)
    print(f"[DocGen] Index written to {index_path}")


if __name__ == "__main__":
    main()
