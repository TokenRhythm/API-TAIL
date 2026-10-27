import math
from numbers import Integral, Real

CREDENTIAL_KEYS = {"api_key", "apikey", "authorization", "password", "secret",
                   "token", "access_token", "headers", "api-key"}

def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)

def number(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} outside its allowed range")
    return float(value)

def reject_credentials(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in CREDENTIAL_KEYS:
                raise ValueError("Credentials and arbitrary headers must not be embedded in config")
            reject_credentials(item)
    elif isinstance(value, list):
        for item in value:
            reject_credentials(item)
