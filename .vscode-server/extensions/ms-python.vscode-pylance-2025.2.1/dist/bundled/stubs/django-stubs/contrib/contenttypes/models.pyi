from typing import Any, ClassVar

from django.contrib.auth.models import Permission
from django.db import models
from django.db.models import Manager
from django.db.models.base import Model
from django.db.models.manager import BaseManager
from django.db.models.query import QuerySet

class ContentTypeManager(BaseManager[ContentType]):
    def get_by_natural_key(self, app_label: str, model: str) -> ContentType: ...
    def get_for_model(
        self, model: type[Model] | Model, for_concrete_model: bool = ...
    ) -> ContentType: ...
    def get_for_models(
        self, *models: Any, for_concrete_models: bool = ...
    ) -> dict[type[Model], ContentType]: ...
    def get_for_id(self, id: int) -> ContentType: ...
    def clear_cache(self) -> None: ...

class ContentType(models.Model):
    id: int
    app_label: models.CharField[Any] = ...
    model: models.CharField[Any] = ...
    objects: ClassVar[  # pyright: ignore[reportIncompatibleVariableOverride]
        ContentTypeManager
    ] = ...
    permission_set: Manager[Permission]
    @property
    def name(self) -> str: ...
    def model_class(self) -> type[Model] | None: ...
    def get_object_for_this_type(self, **kwargs: Any) -> Model: ...
    def get_all_objects_for_this_type(self, **kwargs: Any) -> QuerySet[Any]: ...
    def natural_key(self) -> tuple[str, str]: ...
