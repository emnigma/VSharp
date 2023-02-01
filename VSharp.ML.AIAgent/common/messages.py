from dataclasses_json import dataclass_json, config
from dataclasses import dataclass, field
import json
from enum import Enum

from .game import GameMap, GameState, Reward


class ClientMessageType(str, Enum):
    START = "start"
    STEP = "step"
    GETALLMAPS = "getallmaps"


@dataclass_json
@dataclass
class ClientMessageBody:
    def type(self) -> ClientMessageType:
        pass


@dataclass_json
@dataclass
class GetAllMapsMessageBody(ClientMessageBody):
    def type(self) -> ClientMessageType:
        return ClientMessageType.GETALLMAPS


@dataclass_json
@dataclass
class StartMessageBody(ClientMessageBody):
    MapId: int
    StepsToPlay: int

    def type(self) -> ClientMessageType:
        return ClientMessageType.START


@dataclass_json
@dataclass
class StepMessageBody(ClientMessageBody):
    StateId: int
    PredictedStateUsefulness: float

    def type(self) -> ClientMessageType:
        return ClientMessageType.STEP


@dataclass_json
@dataclass
class ClientMessage:
    MessageType: str = field(init=False)
    MessageBody: ClientMessageBody = field(
        metadata=config(
            encoder=lambda x: x.to_json()
            if issubclass(type(x), ClientMessageBody)
            else json.dumps(x)
        )
    )

    def __post_init__(self):
        self.MessageType = self.MessageBody.type()


@dataclass_json
@dataclass
class MapsMessageBody:
    Maps: list[GameMap]


class ServerMessageType(str, Enum):
    MAPS = "Maps"
    READY_FOR_NEXT_STEP = "ReadyForNextStep"
    MOVE_REVARD = "MoveReward"
    GAMEOVER = "GameOver"
    INCORRECT_PREDICTED_STATEID = "IncorrectPredictedStateId"


@dataclass_json
@dataclass
class ServerMessage:
    def decode(x):
        if x == "" or x == {}:
            return x

        for MessageBodyType in [MapsMessageBody, Reward, GameState]:
            try:
                return MessageBodyType.from_dict(x)
            except (KeyError, ValueError, AttributeError):
                pass

        raise RuntimeError(f"Can't decode msg: {x}")

    MessageType: ServerMessageType
    MessageBody: MapsMessageBody | Reward | GameState = field(
        metadata=config(decoder=decode)
    )