from gym.envs.registration import register


register(
    id="drawobjects-v0",
    entry_point="gym_drawobjects.envs:DrawObjectsEnv")
