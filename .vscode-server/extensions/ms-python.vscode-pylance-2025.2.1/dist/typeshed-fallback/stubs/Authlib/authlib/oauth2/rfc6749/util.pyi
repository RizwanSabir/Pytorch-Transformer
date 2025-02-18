from collections.abc import Collection

def list_to_scope(scope: Collection[str] | str | None) -> str: ...
def scope_to_list(scope: Collection[str] | str | None) -> list[str]: ...
def extract_basic_authorization(headers: dict[str, str]) -> tuple[str, str]: ...
