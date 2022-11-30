from domain.models.network import Optimizers, KerasBackbone, Networks
from domain.models.test_case.test_case import InputReadMode, TestCaseConfig
from infra.services.test_case_service_mongodb import TestCaseServiceMongoDB

fixed_params = TestCaseConfig(
    network=Networks.UNet,
    backbone=KerasBackbone.ResNet50,
    optimizer=Optimizers.Adam,
    filter_min=64,
    use_imagenet_weights=True,
)


def populate_config_remote(db_name: str):
    test_case_serv = TestCaseServiceMongoDB(db_name)
    test_case_serv.remove_all()
    test_case_serv.populate(
        networks=[fixed_params['network']],
        backbones=[fixed_params['backbone']],
        optimizers=[fixed_params['optimizer']],
        read_modes=[InputReadMode.ANY_DEPTH, InputReadMode.BGR2GRAY],
        use_imagenet_weights=[fixed_params['use_imagenet_weights']],
        filters_min=[fixed_params['filter_min']],
        filters_max=[512, 1024],
        sizes=[256, 512]
    )


def get_db_name(params: TestCaseConfig):
    network = params['network'].value
    optimizer = str(params['optimizer'].value).lower()
    backbone = str(params['backbone'].value).lower()
    return f'{network}_{optimizer}_{backbone}'


def run():
    populate_config_remote(get_db_name(fixed_params))
    return 0


run()
