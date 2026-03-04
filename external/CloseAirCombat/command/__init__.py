__all__ = ["KoreaAirCommanderSystem", "CommanderCombatDB"]


def __getattr__(name):
    if name == "KoreaAirCommanderSystem":
        from .air_commander_system import KoreaAirCommanderSystem
        return KoreaAirCommanderSystem
    if name == "CommanderCombatDB":
        from .commander_db import CommanderCombatDB
        return CommanderCombatDB
    raise AttributeError(name)
