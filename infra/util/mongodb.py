from domain.models.network import NetworkConfig


def build_db_name(config: NetworkConfig) -> str:
    return f"{config['size']}_{config['filter_min']}-{config['filter_max']}_pool-{config['pool']}_unpool-{config['unpool']}"
