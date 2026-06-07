"""
generate_sample_docs.py
-----------------------
Generates realistic synthetic financial documents for demo/testing.
Creates fund factsheets, analyst reports, and compliance snippets.

Run: python generate_sample_docs.py
Output: data/sample_docs/*.txt
"""

import os
from pathlib import Path

DOCS = {
    "horizon_growth_fund_factsheet.txt": """
HORIZON GROWTH FUND — FACTSHEET Q1 2024

Fund Overview
-------------
Fund Name: Horizon Growth Fund
Fund Type: Open-ended equity mutual fund
Benchmark: Nifty 500 TRI
Fund Manager: Priya Mehta, CFA (12 years experience)
Inception Date: March 15, 2018
AUM: ₹4,820 Crores (as of March 31, 2024)

Expense Ratio
-------------
Direct Plan: 0.42% per annum
Regular Plan: 1.15% per annum

Investment Objective
--------------------
The fund seeks long-term capital appreciation by investing predominantly in
equity and equity-related instruments of companies with strong growth potential
across market capitalisation segments.

Portfolio Composition (Top 10 Holdings)
----------------------------------------
1. Reliance Industries Ltd         — 8.4%
2. HDFC Bank Ltd                   — 7.1%
3. Infosys Ltd                     — 5.9%
4. ICICI Bank Ltd                  — 5.2%
5. Tata Consultancy Services Ltd   — 4.8%
6. Bharti Airtel Ltd               — 3.7%
7. Larsen & Toubro Ltd             — 3.4%
8. Kotak Mahindra Bank Ltd         — 3.1%
9. Axis Bank Ltd                   — 2.9%
10. Maruti Suzuki India Ltd        — 2.6%

Asset Allocation
----------------
Equity — Large Cap:  52.3%
Equity — Mid Cap:    28.7%
Equity — Small Cap:  14.2%
Cash & Equivalents:   4.8%

Performance (Annualised Returns)
----------------------------------
            1Y      3Y      5Y    Since Inception
Fund       18.4%   14.2%   16.8%     15.1%
Benchmark  15.2%   11.8%   14.3%     13.2%
Alpha       3.2%    2.4%    2.5%      1.9%

Risk Metrics
------------
Standard Deviation (3Y): 14.2%
Sharpe Ratio (3Y): 0.82
Beta: 0.94
Portfolio Turnover: 34% per annum

Exit Load
---------
1% if redeemed within 1 year of allotment. Nil thereafter.

Minimum Investment
------------------
Lumpsum: ₹5,000
SIP: ₹500 per month

Riskometer: Very High Risk
""",

    "techvision_q4_analyst_report.txt": """
TECHVISION SOLUTIONS LTD — EQUITY ANALYST REPORT
Q4 FY2024 Results Review

Rating: BUY (maintained)
Target Price: ₹1,480 (12-month horizon)
Current Market Price: ₹1,215 (as of April 5, 2024)
Upside: 21.8%

Analyst: Rahul Sharma | rahul.sharma@alpharesearch.in
Date: April 8, 2024

Q4 FY2024 Financial Summary
-----------------------------
Revenue:          ₹2,840 Cr  (+18.3% YoY, +4.1% QoQ)
EBITDA:           ₹620 Cr    (+22.4% YoY)
EBITDA Margin:    21.8%       (+70 bps YoY)
PAT:              ₹445 Cr    (+19.1% YoY)
EPS (diluted):    ₹37.2      (+18.8% YoY)
Free Cash Flow:   ₹380 Cr    (conversion: 85.4%)

Full Year FY2024 Performance
------------------------------
Revenue:          ₹10,420 Cr  (+16.7% YoY)
EBITDA:           ₹2,180 Cr   (+20.3% YoY)
PAT:              ₹1,560 Cr   (+18.9% YoY)
ROE:              24.3%
ROCE:             28.7%
Debt/Equity:      0.12x (near debt-free)

Key Business Highlights
-----------------------
- Cloud and SaaS revenue grew 34% YoY, now contributing 38% of total revenue
- Signed 3 large enterprise deals worth ₹280 Cr TCV in Q4 alone
- Attrition declined to 12.4% from 16.8% a year ago — retention improving
- R&D spend increased to 8.2% of revenue, focused on GenAI product suite
- Order backlog stands at ₹6,840 Cr (1.6x annual revenue) — strong visibility

Management Guidance FY2025
---------------------------
Revenue growth: 18–20% YoY
EBITDA margin: 22–23% (100–150 bps expansion expected)
Capex: ₹320 Cr (primarily data centre and AI infrastructure)

Valuation
---------
FY2025E EPS:  ₹44.5
Target P/E:   33x (sector median: 28x, premium justified by GenAI traction)
Target Price: ₹1,480

Key Risks
---------
1. Macro slowdown impacting IT spend by BFSI clients (35% of revenue)
2. Currency headwinds if USD/INR appreciates materially
3. Competition from global SIs in large-deal pursuits
4. Execution risk on GenAI product monetisation timeline

Recommendation: BUY with a 12-month target of ₹1,480.
""",

    "compliance_aml_policy_extract.txt": """
ALPHA ASSET MANAGEMENT — AML & KYC COMPLIANCE POLICY
Section 4: Customer Due Diligence (CDD) Requirements
Effective Date: January 1, 2024 | Version: 3.2

4.1 Standard CDD Requirements
------------------------------
All new clients must undergo Standard Customer Due Diligence before account
activation. The following documents are mandatory:

Individual Clients:
  - Government-issued photo ID (Aadhaar, Passport, Driving Licence, Voter ID)
  - PAN Card (mandatory for investments above ₹50,000)
  - Address proof not older than 3 months
  - Recent passport-size photograph
  - Bank account proof (cancelled cheque or bank statement)

Non-Individual Clients (Corporates, Trusts, HUF):
  - Certificate of Incorporation / Registration
  - Memorandum & Articles of Association
  - Board resolution authorising investments
  - PAN of entity
  - KYC of all beneficial owners holding >25% stake
  - Beneficial Ownership Declaration (Form BO-1)

4.2 Enhanced Due Diligence (EDD)
----------------------------------
EDD is mandatory for:
  a) Politically Exposed Persons (PEPs) and their immediate family members
  b) Clients from FATF non-compliant jurisdictions
  c) Transactions above ₹10,00,000 (Ten Lakhs) in a single day
  d) Non-face-to-face clients
  e) High-risk business categories (shell companies, cash-intensive businesses)

EDD Additional Requirements:
  - Source of funds declaration
  - Source of wealth declaration
  - Senior management approval for account opening
  - Enhanced ongoing monitoring (monthly transaction review)

4.3 Transaction Monitoring Thresholds
--------------------------------------
Automatic flagging is triggered for:
  - Cash transactions above ₹2,00,000
  - Aggregate transactions above ₹10,00,000 in a calendar month
  - Wire transfers above ₹5,00,000 to/from high-risk jurisdictions
  - Structuring patterns (multiple transactions just below thresholds)
  - Sudden spike of >300% vs. 6-month average transaction value

4.4 Suspicious Transaction Reporting
--------------------------------------
STRs must be filed with the Financial Intelligence Unit — India (FIU-IND)
within 7 working days of suspicion being identified. Filing must be done
electronically via the FIU-IND reporting portal.

Failure to file STRs is subject to penalties under the Prevention of Money
Laundering Act (PMLA) 2002, including fines up to ₹1 Crore and potential
criminal prosecution of responsible officers.
""",
}

EVAL_QUESTIONS = [
    {"question": "What is the expense ratio of the Horizon Growth Fund direct plan?"},
    {"question": "Who is the fund manager of the Horizon Growth Fund?"},
    {"question": "What is the AUM of the Horizon Growth Fund as of March 2024?"},
    {"question": "What is the analyst's target price for TechVision Solutions?"},
    {"question": "What was TechVision's revenue growth in Q4 FY2024?"},
    {"question": "What documents are required for corporate KYC onboarding?"},
    {"question": "What is the STR filing deadline under the AML policy?"},
    {"question": "What is the exit load on the Horizon Growth Fund?"},
    {"question": "What is TechVision's FY2025 revenue growth guidance?"},
    {"question": "At what transaction amount is EDD mandatory?"},
]

if __name__ == "__main__":
    import json

    out_dir = Path("data/sample_docs")
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname, content in DOCS.items():
        (out_dir / fname).write_text(content.strip())
        print(f"✓ Created {fname}")

    eval_path = Path("data/eval_questions.json")
    eval_path.write_text(json.dumps(EVAL_QUESTIONS, indent=2))
    print(f"✓ Created eval_questions.json ({len(EVAL_QUESTIONS)} questions)")

    print(f"\n✅ Sample docs ready in {out_dir}/")
    print("   Run: python rag_pipeline.py --docs data/sample_docs/")
    print("   Eval: python rag_pipeline.py --docs data/sample_docs/ --eval")
