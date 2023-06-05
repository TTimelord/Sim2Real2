from src.third_party.ConvONets.config import get_model as ConvONets


def create_network(mode_opt):
    ori = eval(mode_opt.network_type)
    network = ori(mode_opt)
    return network
