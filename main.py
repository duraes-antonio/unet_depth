from typing import List

from domain.models.network import Optimizers, KerasBackbone, Networks, NetworkConfig
from infra.services.test_case_service_mongodb import TestCaseServiceMongoDB


def build_db_name(config: NetworkConfig) -> str:
    return f"{config['size']}_{config['filter_min']}-{config['filter_max']}_pool-{config['pool']}_unpool-{config['unpool']}"


def get_all_config() -> List[NetworkConfig]:
    sizes = [256, 512]
    filter_max = [512, 1024]
    bools = [True]
    configs = []

    for size in sizes:
        for f_max in filter_max:
            for pool in bools:
                for unpool in bools:
                    config: NetworkConfig = {
                        'size': size,
                        'filter_min': 64,
                        'filter_max': f_max,
                        'pool': pool,
                        'unpool': unpool,
                    }
                    configs.append(config)
    return configs


def populate_config_remote():
    for config in get_all_config():
        db_name = build_db_name(config)
        test_case_serv = TestCaseServiceMongoDB(db_name)

        networks = [Networks.UNet, Networks.AttentionUNet, Networks.TransUNet]
        optimizers = [Optimizers.Adam]
        backbones = [KerasBackbone.ResNet50, KerasBackbone.ResNet101]

        test_case_serv.remove_all()
        test_case_serv.populate(networks, backbones, optimizers)


def main():
    populate_config_remote()
    return 0


main()
