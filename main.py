from domain.models.network import Optimizers, KerasBackbone, Networks
from infra.services.test_case_service_mongodb import TestCaseServiceMongoDB


def main():
    test_case_serv = TestCaseServiceMongoDB('unet_depth')

    networks = [Networks.UNet, Networks.AttentionUNet, Networks.TransUNet]
    optimizers = [Optimizers.Adam]
    backbones = [KerasBackbone.ResNet50, KerasBackbone.ResNet101]

    test_case_serv.remove_all()
    test_case_serv.populate(networks, backbones, optimizers)
    return 0


main()
