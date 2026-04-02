# server/my_environment.py
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import MyAction, MyObservation

class MyEnvironment(Environment):
    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self) -> MyObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        return MyObservation(result="Ready", success=True, done=False, reward=0.0)

    def step(self, action: MyAction) -> MyObservation:
        # Implement your logic here
        self._state.step_count += 1
        result = self._execute_command(action.command)
        return MyObservation(result=result, success=True, done=False, reward=1.0)

    @property
    def state(self) -> State:
        return self._state