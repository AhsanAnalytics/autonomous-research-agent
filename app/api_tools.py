import requests
from langchain_core.tools import tool


@tool
def get_exchange_rate(base: str, quote: str) -> str:
    """Get the current currency exchange rate between two currencies.

    Use for currency conversion or FX questions (e.g. "USD to PKR").
    Args:
        base: 3-letter ISO currency code to convert FROM, e.g. "USD".
        quote: 3-letter ISO currency code to convert TO, e.g. "PKR".
    Returns a short string like "1 USD = 278.4 PKR", or an error message.
    """
    base, quote = base.upper().strip(), quote.upper().strip()
    try:
        r = requests.get(f"https://open.er-api.com/v6/latest/{base}", timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("result") != "success":
            return f"error: could not get rates for {base}"
        rate = data.get("rates", {}).get(quote)
        if rate is None:
            return f"error: no rate found for {base}->{quote}"
        return f"1 {base} = {rate} {quote}"
    except requests.Timeout:
        return "error: exchange-rate API timed out"
    except Exception as e:
        return f"error: could not fetch exchange rate ({e})"