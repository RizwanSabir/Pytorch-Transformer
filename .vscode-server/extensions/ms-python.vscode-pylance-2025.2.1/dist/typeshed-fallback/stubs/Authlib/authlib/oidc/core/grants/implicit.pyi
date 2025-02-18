from _typeshed import Incomplete

from authlib.oauth2.rfc6749 import ImplicitGrant

class OpenIDImplicitGrant(ImplicitGrant):
    RESPONSE_TYPES: Incomplete
    DEFAULT_RESPONSE_MODE: str
    def exists_nonce(self, nonce, request) -> None: ...
    def get_jwt_config(self) -> None: ...
    def generate_user_info(self, user, scope) -> None: ...
    def get_audiences(self, request): ...
    def validate_authorization_request(self): ...
    def validate_consent_request(self) -> None: ...
    def create_authorization_response(self, redirect_uri, grant_user): ...
    def create_granted_params(self, grant_user): ...
    def process_implicit_token(self, token, code: Incomplete | None = None): ...
