import carla
client = carla.Client('localhost', 2000)
client.load_world('Town10HD_Opt')