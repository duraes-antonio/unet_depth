from typing import TypeVar, TypedDict

T = TypeVar('T')


class Entity(TypedDict):
    id: str


class NamedEntity(Entity):
    name: str
