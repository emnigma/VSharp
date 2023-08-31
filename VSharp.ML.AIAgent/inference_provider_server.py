import pathlib

import torch
from aiohttp import web

from common.game import GameState
from config import GeneralConfig
from ml.data_loader_compact import ServerDataloaderHeteroVector
from ml.predict_state_vector_hetero import PredictStateVectorHetGNN

routes = web.RouteTableDef()


MODEL_PATH = pathlib.Path("test_model.pth")
MODEL = GeneralConfig.EXPORT_MODEL_INIT()
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))


def run_inference(model: torch.nn.Module, game_state: GameState) -> int:
    data, state_map = ServerDataloaderHeteroVector.convert_input_to_tensor(game_state)
    predicted_state = PredictStateVectorHetGNN.predict_state_single_out(
        model, data, state_map
    )
    return predicted_state


@routes.post("/run_inference")
async def dequeue_instance(request):
    game_state_raw = await request.read()
    game_state = GameState.from_json(game_state_raw.decode("utf-8"))

    predicted_state = run_inference(MODEL, game_state)
    return web.Response(text=str(predicted_state))


def main():
    app = web.Application()
    app.add_routes(routes)
    web.run_app(app, port=8080)


if __name__ == "__main__":
    main()
