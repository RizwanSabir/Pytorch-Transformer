import re

class RomanError(Exception): ...
class OutOfRangeError(RomanError): ...
class NotIntegerError(RomanError): ...
class InvalidRomanNumeralError(RomanError): ...

romanNumeralMap: tuple[tuple[str, int], ...]

def toRoman(n: int) -> str: ...

romanNumeralPattern: re.Pattern[str]

def fromRoman(s: str) -> int: ...
