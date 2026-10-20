"""The example of the SAP export that the Datos screen shows: the same columns as a real one, with invented plantas, materials and tonnes."""

from __future__ import annotations

# Row 1 of a real long export. The two untitled columns hold the customer name and, right after "Material", the description of the SKU.
HEADER = ["Customer", None, "Material", None, "Day", "TO"]
# planta, customer name, material code, SKU description, day as text (dd.mm.yyyy), tonnes
ROWS = [
    ("DEMO1", "CLIENTE DE EJEMPLO UNO", "900001", "K/01/125gsm/2000mm/1200-1400", "02.03.2026", 12.345),
    ("DEMO1", "CLIENTE DE EJEMPLO UNO", "900001", "K/01/125gsm/2000mm/1200-1400", "04.03.2026", 8.1),
    ("DEMO1", "CLIENTE DE EJEMPLO UNO", "900001", "K/01/125gsm/2000mm/1200-1400", "05.03.2026", 15.72),
    ("DEMO1", "CLIENTE DE EJEMPLO UNO", "900002", "TSL/01/80gsm/2450mm/1200-1450", "03.03.2026", 6.4),
    ("DEMO1", "CLIENTE DE EJEMPLO UNO", "900002", "TSL/01/80gsm/2450mm/1200-1450", "05.03.2026", 9.875),
    ("DEMO2", "CLIENTE DE EJEMPLO DOS", "900003", "HP2/01/105gsm/2440mm/1200-1450", "02.03.2026", 21.0),
    ("DEMO2", "CLIENTE DE EJEMPLO DOS", "900003", "HP2/01/105gsm/2440mm/1200-1450", "06.03.2026", 17.55),
]


def example_table() -> dict:
    r"""The example as the screen draws it: the letters and titles of the columns, and the rows."""
    return {
        "columnas": [{"letra": letter, "titulo": title or ""} for letter, title in zip("ABCDEF", HEADER)],
        "filas": [list(row) for row in ROWS],
    }
