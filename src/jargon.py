"""
jargon.py
---------
Makes the output understandable for a common person (a "layman").

Two simple, rule-based tools:
    1. SHORT_FORMS      -> abbreviations replaced directly inside the summary
                           (e.g. "YoY" -> "compared to last year")
    2. FINANCIAL_TERMS  -> difficult words found in the report are listed in a
                           "Difficult words explained" glossary

To support more words, just add them to the dictionaries below.
"""

import re

# ---------------------------------------------------------------------------
# 1. Abbreviations: replaced INSIDE the summary text
# ---------------------------------------------------------------------------
SHORT_FORMS = {
    "YoY": "compared to the same period last year",
    "y-o-y": "compared to the same period last year",
    "QoQ": "compared to the previous quarter",
    "q-o-q": "compared to the previous quarter",
    "bps": "basis points (100 bps = 1%)",
    "FY": "financial year",
    "H1": "first half of the year",
    "H2": "second half of the year",
    "Q1": "first quarter (3 months)",
    "Q2": "second quarter (3 months)",
    "Q3": "third quarter (3 months)",
    "Q4": "fourth quarter (3 months)",
    "mn": "million",
    "bn": "billion",
}

# ---------------------------------------------------------------------------
# 2. Financial terms: explained in the glossary
# ---------------------------------------------------------------------------
FINANCIAL_TERMS = {
    "revenue": "Total money earned from selling products/services, before any costs are removed.",
    "net sales": "Money from sales after removing returns and discounts.",
    "net profit": "The final profit left after paying ALL costs, interest and taxes.",
    "net income": "The final profit left after paying ALL costs, interest and taxes.",
    "net loss": "The company spent more than it earned in this period.",
    "gross margin": "Out of every 100 earned from sales, how much is left after paying for making the product.",
    "operating profit": "Profit from the company's main business, before interest and taxes.",
    "operating margin": "Out of every 100 earned, how much remains as profit from the main business.",
    "ebitda": "Profit before interest, taxes and wear-and-tear of assets. Shows how well the core business runs.",
    "ebit": "Profit before interest and taxes are paid.",
    "eps": "Earnings Per Share - the company's profit divided by the number of shares. Higher is better.",
    "earnings per share": "The company's profit divided by the number of shares. Higher is better.",
    "dividend": "Part of the profit paid out in cash to shareholders.",
    "buyback": "The company buys back its own shares, usually a sign it thinks the shares are cheap.",
    "liquidity": "How easily the company can pay its short-term bills with available cash.",
    "cash flow": "Money actually coming into and going out of the company.",
    "free cash flow": "Cash left after running the business and investing in equipment. Can be used for dividends or paying debt.",
    "capex": "Capital Expenditure - money spent on long-term things like buildings and machines.",
    "capital expenditure": "Money spent on long-term things like buildings and machines.",
    "debt": "Money the company has borrowed and must pay back.",
    "leverage": "How much the company relies on borrowed money. High leverage = higher risk.",
    "debt-to-equity": "Borrowed money compared to owners' money. Higher = more risky.",
    "balance sheet": "A snapshot of what the company owns (assets) and owes (liabilities).",
    "assets": "Things the company owns that have value (cash, buildings, machines).",
    "liabilities": "Money the company owes to others (loans, unpaid bills).",
    "equity": "The owners' share of the company = assets minus liabilities.",
    "impairment": "The company admits something it owns is now worth less than it thought - a loss.",
    "write-down": "Reducing the recorded value of something because it lost value.",
    "write-off": "Removing something from the books because it's worth nothing now (e.g. a loan that won't be repaid).",
    "depreciation": "Spreading the cost of a machine/building over its useful life as it wears out.",
    "amortization": "Like depreciation, but for non-physical things like patents or software.",
    "provision": "Money set aside now for an expected future loss or expense.",
    "restructuring": "Big reorganisation of the company (closing units, job cuts) to reduce costs.",
    "guidance": "The company's own forecast of how it will perform in the future.",
    "outlook": "What the company expects for the coming months/year.",
    "headwinds": "Problems or conditions that slow the business down.",
    "tailwinds": "Conditions that help the business grow.",
    "market share": "The company's portion of total sales in its industry.",
    "inventory": "Goods the company has stored but not yet sold.",
    "working capital": "Money available for day-to-day operations (short-term assets minus short-term debts).",
    "interest expense": "Cost of borrowing money (interest paid on loans).",
    "credit rating": "A grade showing how safe it is to lend money to the company.",
    "volatility": "How much and how quickly prices go up and down. High = unpredictable.",
    "basis points": "1 basis point = 0.01%. So 100 basis points = 1%.",
    "fiscal year": "The company's 12-month accounting year (may not start in January).",
    "year-over-year": "Compared to the same period last year.",
    "quarter": "A 3-month period. A year has 4 quarters.",
    "margin": "The percentage of sales that remains as profit.",
    "shareholders": "People/institutions who own shares (part ownership) of the company.",
    "market capitalisation": "Total value of all the company's shares on the stock market.",
    "market capitalization": "Total value of all the company's shares on the stock market.",
    "valuation": "How much the company (or an asset) is estimated to be worth.",
    "acquisition": "When one company buys another company.",
    "merger": "When two companies join together to become one.",
    "goodwill": "Extra amount paid when buying a company, above the value of its assets (for brand, customers, etc.).",
    "hedging": "Taking steps to protect against losses from price or currency changes.",
    "foreign exchange": "Changes in currency rates (e.g. dollar vs rupee) that affect profits.",
    "inflation": "General rise in prices, which reduces what money can buy.",
    "default": "Failing to repay a loan on time.",
    "covenant": "Rules a borrower must follow as part of a loan agreement.",
    "refinancing": "Replacing an old loan with a new one, usually with better terms.",
    "order backlog": "Orders received from customers but not yet delivered.",
    "backlog": "Orders received from customers but not yet delivered.",
    "like-for-like": "Comparison that excludes new or closed stores/units, to compare fairly.",
    "organic growth": "Growth from the company's own business, NOT from buying other companies.",
    "headcount": "Number of employees.",
    "cost of goods sold": "Direct cost of making the products that were sold.",
    "operating expenses": "Regular costs of running the business (salaries, rent, marketing).",
    "non-performing": "Loans where the borrower has stopped paying.",
}


def simplify_summary(bullets: list[str]) -> list[str]:
    """Replace abbreviations (YoY, Q3, bps...) with plain words inside each bullet."""
    simplified = []
    for bullet in bullets:
        # "modest YoY improvements" -> "modest yearly improvements" (used as an adjective)
        bullet = re.sub(r"\b(YoY|y-o-y)\s+(?=[a-z])(?!(to|in|and|at|from|on|for|while)\b)",
                        "yearly ", bullet)
        for short, plain in SHORT_FORMS.items():
            # \b = whole word only, so "FY" won't match inside "FYI"
            bullet = re.sub(rf"\b{re.escape(short)}\b", plain, bullet)
        simplified.append(bullet)
    return simplified


def find_jargon(text: str) -> dict[str, str]:
    """Return {term: simple meaning} for every financial term found in the text."""
    lower = text.lower()
    found = {}
    # check longer terms first so "free cash flow" is matched before "cash flow"
    for term in sorted(FINANCIAL_TERMS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(term)}\b", lower):
            # skip a short term if it's only part of a longer term already found
            if any(term in bigger for bigger in found):
                continue
            found[term] = FINANCIAL_TERMS[term]
    return dict(sorted(found.items()))
